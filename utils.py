"""Camera geometry and depth regularization."""
import torch
import torch.nn.functional as F


def create_raymaps(cameras, h, w):
    rays_o, rays_d = create_rays(cameras, h, w)
    raymaps = torch.cat([rays_d, rays_o - (rays_o * rays_d).sum(dim=-1, keepdim=True) * rays_d], dim=-1)
    return raymaps


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# Quaternion conversion utilities adapted from PyTorch3D.
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return standardize_quaternion(out)


@torch.amp.autocast(device_type="cuda", enabled=False)
def normalize_cameras(cameras, return_meta=False, ref_w2c=None, T_norm=None, n_frame=None):
    B, N = cameras.shape[:2]
     
    c2ws = torch.zeros(B, N, 3, 4, device=cameras.device)

    c2ws[..., :3, :3] = quaternion_to_matrix(cameras[..., 0:4])
    c2ws[..., :, 3] = cameras[..., 4:7]

    _c2ws = c2ws

    ref_w2c = torch.inverse(matrix_to_square(_c2ws[:, :1])) if ref_w2c is None else ref_w2c
    _c2ws = (ref_w2c.repeat(1, N, 1, 1) @ matrix_to_square(_c2ws))[..., :3, :]

    if n_frame is not None:
        T_norm = _c2ws[..., :n_frame, :3, 3].norm(dim=-1).max(dim=1)[0][..., None, None] if T_norm is None else T_norm
    else:
        T_norm = _c2ws[..., :3, 3].norm(dim=-1).max(dim=1)[0][..., None, None] if T_norm is None else T_norm

    _c2ws[..., :3, 3] = _c2ws[..., :3, 3] / (T_norm + 1e-2)

    R = matrix_to_quaternion(_c2ws[..., :3, :3])
    T = _c2ws[..., :3, 3]
    cameras = torch.cat([R.float(), T.float(), cameras[..., 7:]], dim=-1)

    if return_meta:
        return cameras, ref_w2c, T_norm
    else:
        return cameras


def create_rays(cameras, h, w, uv_offset=None):
    prefix_shape = cameras.shape[:-1]
    cameras = cameras.flatten(0, -2)
    device = cameras.device
    N = cameras.shape[0]

    c2w = torch.eye(4, device=device)[None].repeat(N, 1, 1)
    c2w[:, :3, :3] = quaternion_to_matrix(cameras[:, :4])
    c2w[:, :3, 3] = cameras[:, 4:7]

    # fx, fy, cx, cy should be divided by original H, W
    fx, fy, cx, cy = cameras[:, 7:].chunk(4, -1)

    fx, cx = fx * w, cx * w
    fy, cy = fy * h, cy * h

    inds = torch.arange(0, h*w, device=device).expand(N, h*w)
        
    i = inds % w + 0.5
    j = torch.div(inds, w, rounding_mode='floor') + 0.5

    du = uv_offset[..., 0].reshape(N, h*w) * cx if uv_offset is not None else 0
    dv = uv_offset[..., 1].reshape(N, h*w) * cy if uv_offset is not None else 0
    zs = -torch.ones_like(i)
    xs = (i - cx + du) / fx
    ys = -(j - cy + dv) / fy
    directions = torch.stack((xs, ys, zs), dim=-1)

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)

    rays_o = c2w[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)

    rays_o = rays_o.reshape(*prefix_shape, h, w, 3)
    rays_d = rays_d.reshape(*prefix_shape, h, w, 3)

    return rays_o, rays_d


def matrix_to_square(mat):
    l = len(mat.shape)
    if l==3:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],1,1).to(mat.device)],dim=1)
    elif l==4:
        return torch.cat([mat, torch.tensor([0,0,0,1]).repeat(mat.shape[0],mat.shape[1],1,1).to(mat.device)],dim=2)


def depth_reg_fn(depths): 
    depths = depths / (depths.flatten(1, -1).max(dim=-1)[0][:, None, None, None].detach() + 1e-3)

    # Compute the difference between neighboring pixels in each direction.
    depths_dx = depths.diff(dim=-1)
    depths_dy = depths.diff(dim=-2)

    return (depths_dx.abs().mean() + depths_dy.abs().mean())


def sample_from_dense_cameras(cameras, t):
    """Interpolate poses and intrinsics for visualization at fractions t in [0, 1]."""
    if len(cameras) == 1:
        return cameras.expand(len(t), -1)
    index = t * (len(cameras) - 1)
    left = index.floor().long().clamp(max=len(cameras) - 2)
    weight = (index - left)[:, None]
    a, b = cameras[left], cameras[left + 1]
    qa, qb = a[:, :4], b[:, :4]
    dot = (qa * qb).sum(-1, keepdim=True)
    qb = torch.where(dot < 0, -qb, qb)
    angle = dot.abs().clamp(max=1).acos()
    denom = angle.sin().clamp_min(1e-8)
    spherical = ((1 - weight) * angle).sin() / denom * qa + (weight * angle).sin() / denom * qb
    linear = torch.lerp(qa, qb, weight)
    rotation = F.normalize(torch.where(angle < 1e-4, linear, spherical), dim=-1)
    return torch.cat((rotation, torch.lerp(a[:, 4:], b[:, 4:], weight)), dim=-1)
