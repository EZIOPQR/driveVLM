"""Load precomputed optical-flow sidecars (.npz) for 5-channel vision input."""

import os

import numpy as np
import torch


def flow_npz_path_for_image(image_path: str, flow_root: str) -> str:
    """Map .../samples/CAM/foo.jpg -> flow_root/CAM/foo.npz."""
    cam = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(flow_root, cam, f"{stem}.npz")


def load_flow_uv_tensor(
    image_path: str,
    flow_root: str,
    height: int,
    width: int,
    flow_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return [2, H, W] flow (u,v) scaled by ``1/flow_scale``, float tensor."""
    if float(flow_scale) == 0.0:
        raise ValueError("flow_scale must be non-zero")
    path = flow_npz_path_for_image(image_path, flow_root)
    u = np.zeros((height, width), dtype=np.float32)
    v = np.zeros((height, width), dtype=np.float32)
    if os.path.isfile(path):
        z = np.load(path)
        raw_v = z.get("valid", np.array(True))
        if isinstance(raw_v, np.ndarray):
            valid = bool(raw_v.item()) if raw_v.size else True
        else:
            valid = bool(raw_v)
        if valid:
            u = np.asarray(z["u"], dtype=np.float32)
            v = np.asarray(z["v"], dtype=np.float32)
            if u.shape[0] != height or u.shape[1] != width:
                raise ValueError(
                    f"Flow shape {u.shape} != ({height},{width}) for {path}"
                )
    u = u / float(flow_scale)
    v = v / float(flow_scale)
    t = torch.from_numpy(np.stack([u, v], axis=0)).to(device=device, dtype=dtype)
    return t  # [2,H,W]
