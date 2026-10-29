"""Moving MNIST loading, symmetry transformations and space/time windows.

Conventions follow the handout exactly:
  * videos are uint8 tensors of shape (20, 1, 64, 64); frames are numbered 1..20
  * split by dataset index: 0-7999 train, 8000-8999 val, 9000-9999 test
  * intensities are converted to float32 in [0, 1] (divide by 255) per batch
"""
import os

import torch
import torchvision

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N_FRAMES = 20
SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}


def load_videos(root=ROOT):
    """All 10,000 videos as a uint8 tensor (10000, 20, 1, 64, 64)."""
    ds = torchvision.datasets.MovingMNIST(root=root, split=None, download=True)
    data = ds.data
    assert data.shape == (10000, 20, 1, 64, 64), data.shape
    return data


def split(videos, name):
    lo, hi = SPLITS[name]
    return videos[lo:hi]


def to_float(x):
    return x.float().div_(255.0)


# ---------------------------------------------------------------- symmetries
def mirror(x):
    """(Mx)[i, j] = x[i, 63 - j]; acts on the last two (row, column) axes."""
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    """R^a: a quarter turns anticlockwise."""
    return torch.rot90(x, k=a, dims=(-2, -1))


# G = {R^a, MR^a : a = 0..3}, an element is (m, a) meaning M^m R^a (rotate first, then mirror)
G = [(m, a) for m in (0, 1) for a in range(4)]


def g_name(g):
    m, a = g
    return ("M" if m else "") + (f"R{a}" if a or not m else "")


def apply_g(x, g):
    m, a = g
    y = rotate(x, a) if a else x
    return mirror(y) if m else y


def apply_g_inv(x, g):
    m, a = g
    if m:  # every MR^a is its own inverse
        return apply_g(x, g)
    return rotate(x, -a) if a else x


# ---------------------------------------------------------------- windows
def frame_indices(t, k):
    """1-based frame numbers of the 5 input and 5 target frames of a window."""
    inp = [t + k * i for i in range(5)]
    tgt = [t + k * (5 + i) for i in range(5)]
    return inp, tgt


def window(video, t, k=1):
    """Window of one video (20, ...) starting at frame t with speed k.

    Returns X (5, ...), Y (n, ...) with only the target frames that exist (<= 20).
    """
    inp, tgt = frame_indices(t, k)
    assert inp[-1] <= N_FRAMES, "all five input frames must exist"
    tgt = [f for f in tgt if f <= N_FRAMES]
    return video[[f - 1 for f in inp]], video[[f - 1 for f in tgt]]


def batch_window(videos, t, k):
    """Batched windows with per-video start t (B,) and speed k (B,) (LongTensors or ints).

    Returns X (B, 5, ...), Y (B, 5, ...) padded with zeros where the target frame
    does not exist, and mask (B, 5) bool marking existing target frames.
    """
    B = videos.shape[0]
    t = torch.as_tensor(t, dtype=torch.long).cpu().expand(B)
    k = torch.as_tensor(k, dtype=torch.long).cpu().expand(B)
    steps = torch.arange(10)
    frames = t[:, None] + k[:, None] * steps[None, :]  # 1-based, (B, 10)
    exists = frames <= N_FRAMES
    idx = (frames.clamp(max=N_FRAMES) - 1).to(videos.device)
    gathered = videos[torch.arange(B, device=videos.device)[:, None], idx]
    mask = exists[:, 5:].to(videos.device)
    X, Y = gathered[:, :5], gathered[:, 5:].clone()
    Y[~mask] = 0
    return X, Y, mask


def allowed_starts(k, speed_aug):
    """Allowed 1-based start frames for speed k under the two training rules."""
    if not speed_aug:
        assert k == 1
        return list(range(1, 12))  # all five target frames exist
    # all five inputs and at least two targets exist: t + 6k <= 20
    return [t for t in range(1, N_FRAMES + 1) if t + 6 * k <= N_FRAMES]
