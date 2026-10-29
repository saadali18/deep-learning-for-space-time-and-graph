"""Assemble report.html from the section fragments and result JSON files, then print it to report.pdf.

Placeholders of the form {{name}} in the fragments are replaced by tables generated here from
Question2/results/*.json, so every number in the report comes straight from the experiment outputs.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "Question2", "results")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def fmt(v, sci_below=1e-3):
    """Three significant figures; scientific notation below 1e-3."""
    if v is None:
        return "—"
    if v == 0:
        return "0"
    if abs(v) < sci_below:
        m, e = f"{v:.2e}".split("e")
        return f"{m}×10<sup>{int(e)}</sup>"
    return f"{v:#.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def ms(rows, base, key, fmt_fn=fmt):
    """'mean (seed 0, seed 1)' for a learned model, or the single value for a deterministic one."""
    if f"{base}_s0" in rows:
        a, b = rows[f"{base}_s0"][key], rows[f"{base}_s1"][key]
        return f"{fmt_fn((a + b) / 2)} <span class='seeds'>({fmt_fn(a)}, {fmt_fn(b)})</span>"
    return fmt_fn(rows[base][key])


def html_table(header, body, cls="res"):
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
    return f"<table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def secs(v):  # three significant figures
    return f"{v:#.3g} s"


def table1():
    rows = load("table1.json")
    sup = lambda t: re.sub(r"\^(−?\d)", r"<sup>\1</sup>", t)
    return html_table(["Check", "Expected", "Yours"], [[sup(r[0]), r[1], r[2]] for r in rows])


def table2():
    t = load("table2.json")
    t = {**{k: v for k, v in t.items()}}
    body = []
    for label, base in [("Model A", "A"), ("Blank", "Blank"), ("Copy-last", "Copy-last")]:
        body.append([label] + [ms2(t, base, f"MSE{h}") for h in range(1, 6)] + [ms2(t, base, "E5"),
                     f"{t[base + '_s0' if base == 'A' else base]['params']:,}"])
    return html_table(["Predictor", "MSE<sub>1</sub>", "MSE<sub>2</sub>", "MSE<sub>3</sub>", "MSE<sub>4</sub>",
                       "MSE<sub>5</sub>", "E<sup>(5)</sup>", "Params"], body)


def table3():
    t = load("table3_4.json")
    body = [[lab] + [ms(t, b, k) for k in ("E5_1x", "CM", "EM", "CR", "ER")]
            for lab, b in [("Model A", "A"), ("Blank", "Blank"), ("Copy-last", "Copy-last")]]
    return html_table(["Predictor", "E<sup>(5)</sup>", "C<sub>M</sub>", "E<sub>M</sub>", "C<sub>R</sub>", "E<sub>R</sub>"], body)


def table4():
    t = load("table3_4.json")
    body = [[lab] + [ms(t, b, k) for k in ("E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x")]
            for lab, b in [("Model A", "A"), ("Copy-last", "Copy-last")]]
    return html_table(["Predictor", "E<sup>(2)</sup> 1×", "E<sup>(2)</sup> 2×", "E<sup>(2)</sup> 3×", "E<sup>(5)</sup> 1×",
                       "E<sup>(5)</sup> 2×"], body)


T56_A = ["Model", "E<sup>(5)</sup> 1×", "C<sub>M</sub>", "E<sub>M</sub>", "C<sub>R</sub>", "E<sub>R</sub>"]
T56_B = ["Model", "E<sup>(2)</sup> 1×", "E<sup>(2)</sup> 2×", "E<sup>(2)</sup> 3×", "Params", "Train s/epoch",
         "Inference (1,000 videos)"]


def ms2(rows, base, key, fmt_fn=fmt):
    """Mean on the first line, (seed 0, seed 1) on the second."""
    if f"{base}_s0" not in rows:
        return fmt_fn(rows[base][key])
    a, b = rows[f"{base}_s0"][key], rows[f"{base}_s1"][key]
    return f"{fmt_fn((a + b) / 2)}<br><span class='seeds'>({fmt_fn(a)}, {fmt_fn(b)})</span>"


def t56(t, models):
    part_a = [[lab] + [ms2(t, b, k) for k in ("E5_1x", "CM", "EM", "CR", "ER")] for lab, b in models]
    part_b = [[lab] + [ms2(t, b, k) for k in ("E2_1x", "E2_2x", "E2_3x")]
              + [f"{t[b + '_s0']['params']:,}", ms2(t, b, "train_s_per_epoch", secs), ms2(t, b, "inference_s", secs)]
              for lab, b in models]
    return ("<p class='tcap'>(a) Accuracy and consistency at 1×</p>" + html_table(T56_A, part_a, "res") +
            "<p class='tcap'>(b) Speed generalisation and cost</p>" + html_table(T56_B, part_b, "res"))


def table5():
    t = load("table5.json")
    models = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")] + ([("D+B", "DB")] if "DB_s0" in t else [])
    return t56(t, models)


def table6():
    t = load("table6.json")
    rows = t["rows"]
    names = {"sym:B": "Symmetrised B", "sym:D": "Symmetrised D", "sym:DB": "Symmetrised D+B"}
    models = [(names[n] + (" (final)" if n == t["final"] else ""), n)
              for n in ("sym:B", "sym:D", "sym:DB") if f"{n}_s0" in rows]
    return t56(rows, models)


def table_dsel():
    t = load("model_D_selection.json")
    desc = {"dil12": ("layer 2 dilated ×2 (3×3)", "1 + 2 cells = 12 px"),
            "k55h46": ("5×5 kernels, 46 hidden", "2 + 2 cells = 16 px"),
            "p8": ("8×8 patches (8×8 grid)", "2 cells × 8 px = 16 px")}
    body = [[n + (" ✓ chosen" if n == t["chosen"] else ""), desc[n][0], desc[n][1], f"{r['params']:,}", fmt(r["val_E5_1x"]),
             fmt(r["val_E2_1x"]), fmt(r["val_E2_2x"]), fmt(r["val_E2_3x"]), fmt(r["val_E2_mean"])]
            for n, r in t["candidates"].items()]
    return html_table(["Candidate", "Change vs. A", "Reach", "Params", "val E<sup>(5)</sup> 1×", "val E<sup>(2)</sup> 1×",
                       "val E<sup>(2)</sup> 2×", "val E<sup>(2)</sup> 3×", "mean val E<sup>(2)</sup>"], body, "res small")


TABLES = {"T1": table1, "T2": table2, "T3": table3, "T4": table4, "T5": table5, "T6": table6, "TDSEL": table_dsel}


def render(text):
    def rep(m):
        key = m.group(1)
        if key in TABLES:
            try:
                return TABLES[key]()
            except Exception as ex:  # missing results while drafting
                return f"<p class='todo'>[{key} pending: {ex}]</p>"
        return m.group(0)
    return re.sub(r"\{\{(\w+)\}\}", rep, text)


def main():
    parts = []
    for name in ("front.html", "q1.html", "q2_setup.html", "q2_baseline.html", "q2_testing.html", "q2_data.html", "q2_final.html", "ai_usage.html"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            with open(p) as fh:
                parts.append(render(fh.read()))
    with open(os.path.join(HERE, "template.html")) as fh:
        tpl = fh.read()
    body = "\n".join(parts)
    # LaTeX-style caption labels: "<b>Figure 3.</b>" -> "Figure 3:", "<b>Table 5.</b>" -> "Table 5:"
    body = re.sub(r"<b>((?:Figure|Table) [^<]{1,12}?)\.</b>", r"\1:", body)
    out = tpl.replace("<!--BODY-->", body)
    html_path = os.path.join(HERE, "report.html")
    with open(html_path, "w") as fh:
        fh.write(out)
    if "--no-pdf" not in sys.argv:
        pdf = os.path.join(HERE, "report.pdf")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=60000", "--run-all-compositor-stages-before-draw",
                        f"--print-to-pdf={pdf}", "file://" + html_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(pdf)


if __name__ == "__main__":
    main()
