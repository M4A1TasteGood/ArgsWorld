"""Inference artifacts: direct DiT video and FlashWorld Gaussian exports."""
from importlib import import_module
import math
from pathlib import Path

import numpy as np
import torch


def add_export_arguments(parser):
    from argparse import BooleanOptionalAction
    parser.add_argument('--save_spz', '--spz', action=BooleanOptionalAction, default=False,
                        help='Save final world.spz using the optional spz package')
    parser.add_argument('--spz_opacity_threshold', type=float, default=0.00025,
                        help='Filter low-opacity splats in the exported SPZ file')


def check_spz_export(args):
    if not 0 <= args.spz_opacity_threshold < 1:
        raise ValueError('spz_opacity_threshold must be in [0, 1)')
    if args.save_spz:
        try:
            import_module("spz")
        except ImportError as exc:
            raise ImportError('SPZ export requires the optional spz Python bindings') from exc


@torch.no_grad()
def export_spz(path, gaussians, opacity_threshold=0.00025):
    """FlashWorld layout [xyz, alpha, scales, wxyz, SH], in the stored world coordinate frame.

    Use log scales, logit alpha and coefficient-major SH, as in FlashWorld/utils.py.
    Convert gsplat's scalar-first (wxyz) rotations to SPZ's scalar-last (xyzw).
    SPZ and PLY use the same fixed session frame.
    """
    import spz
    if not 0 <= opacity_threshold < 1:
        raise ValueError('opacity_threshold must be in [0, 1)')
    if gaussians.ndim != 2:
        raise ValueError('Expected gaussians [N,11+3*(degree+1)**2]')
    channels = gaussians.shape[1] - 11
    degree = math.isqrt(max(channels // 3, 0)) - 1
    if degree not in range(4) or channels != 3 * (degree + 1)**2:
        raise ValueError('Invalid Gaussian SH layout (SPZ supports degrees 0–3)')
    g = gaussians.detach().float().cpu()
    mask = g[:, 3] >= opacity_threshold
    if not bool(mask.all()):
        g = g[mask]
    if not len(g):
        raise ValueError('No Gaussians remain for SPZ export; lower --spz_opacity_threshold')
    if not torch.isfinite(g).all() or (g[:, 4:7] <= 0).any():
        raise ValueError('SPZ requires finite Gaussian parameters and positive scales')
    rotation = g[:, 7:11]
    if (rotation.norm(dim=-1) == 0).any():
        raise ValueError('SPZ requires nonzero rotation quaternions')
    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    cloud = spz.GaussianCloud()
    cloud.sh_degree = degree
    cloud.positions = g[:, :3].contiguous().numpy().ravel()
    cloud.scales = (g[:, 4:7] + 1e-8).log().contiguous().numpy().ravel()
    cloud.rotations = rotation[:, [1, 2, 3, 0]].contiguous().numpy().ravel()
    cloud.alphas = torch.logit(g[:, 3].clamp(1e-6, 1-1e-6)).numpy()
    cloud.colors = g[:, 11:14].contiguous().numpy().ravel()
    cloud.sh = g[:, 14:].contiguous().numpy().ravel()
    result = spz.save_spz(cloud, spz.PackOptions(), str(path))
    if result is False or not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise OSError(f'Failed to save SPZ: {path}')
    return len(g)


@torch.no_grad()
def export_ply(path, gaussians, chunk_size=100_000):
    """Write standard 3DGS PLY: three DC fields and channel-major higher SH."""
    if gaussians.ndim != 2 or not len(gaussians):
        raise ValueError('Expected nonempty gaussians [N,D]')
    channels = gaussians.shape[1] - 11
    degree = math.isqrt(max(channels // 3, 0)) - 1
    if degree not in range(4) or channels != 3 * (degree + 1)**2:
        raise ValueError('Invalid Gaussian SH layout')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    names = (['x', 'y', 'z'] + [f'f_dc_{i}' for i in range(3)] +
             [f'f_rest_{i}' for i in range(channels - 3)] +
             ['opacity'] + [f'scale_{i}' for i in range(3)] +
             [f'rot_{i}' for i in range(4)])
    header = ('ply\nformat binary_little_endian 1.0\n' +
              f'element vertex {len(gaussians)}\n' +
              ''.join(f'property float {name}\n' for name in names) +
              'end_header\n')
    with Path(path).open('wb') as file:
        file.write(header.encode('ascii'))
        for offset in range(0, len(gaussians), chunk_size):
            g = gaussians[offset:offset + chunk_size].detach().float().cpu().numpy()
            if not np.isfinite(g).all() or (g[:, 4:7] <= 0).any():
                raise ValueError('PLY requires finite Gaussian parameters and positive scales')
            rows = np.empty((len(g), len(names)), dtype='<f4')
            rows[:, :3] = g[:, :3]
            rows[:, 3:6] = g[:, 11:14]
            # The model/SPZ use [coefficient, RGB]; 3DGS PLY uses [RGB, coefficient].
            rows[:, 6:3 + channels] = g[:, 14:].reshape(len(g), -1, 3).transpose(0, 2, 1).reshape(len(g), channels - 3)
            alpha = np.clip(g[:, 3], 1e-6, 1 - 1e-6)
            rows[:, 3 + channels] = np.log(alpha / (1 - alpha))
            rows[:, 4 + channels:7 + channels] = np.log(g[:, 4:7] + 1e-8)
            rows[:, 7 + channels:] = g[:, 7:11]
            file.write(rows.tobytes())
    return len(gaussians)


@torch.no_grad()
def direct_video_frames(codec, latents, frames_per_view=4):
    """Decode direct x0 latents, without CUDA autocast; hold each generated view for playback."""
    if frames_per_view < 1:
        raise ValueError('frames_per_view must be positive')
    with torch.autocast(device_type=latents.device.type, enabled=False):
        rgb = codec.decode(latents.float())[0]
    frames = ((rgb.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return np.repeat(frames, frames_per_view, axis=0)
