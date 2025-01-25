import torch

def hamilton_product(q1, q2):
    """
    Compute the Hamilton product of two quaternions.

    q1 and q2 are tensors of shape [B, 4, H, W]
    """
    r1, i1, j1, k1 = q1[:, 0:1, :, :], q1[:, 1:2, :, :], q1[:, 2:3, :, :], q1[:, 3:4, :, :]
    r2, i2, j2, k2 = q2[:, 0:1, :, :], q2[:, 1:2, :, :], q2[:, 2:3, :, :], q2[:, 3:4, :, :]

    r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
    i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
    j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
    k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2

    return torch.cat([r, i, j, k], dim=1)  # [B, 4, H, W]
