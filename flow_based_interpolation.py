import torch
import torch.nn.functional as F

_KERNEL_CACHE = {}
_GRID_CACHE = {}


def _device_key(device) -> str:
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def _get_flow_kernels(device):
    key = _device_key(device)
    if key not in _KERNEL_CACHE:
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
        ).view(1, 1, 3, 3)
        laplacian_kernel = torch.tensor(
            [[1 / 12, 1 / 6, 1 / 12],
             [1 / 6, 0.0, 1 / 6],
             [1 / 12, 1 / 6, 1 / 12]],
            device=device,
        ).view(1, 1, 3, 3)
        _KERNEL_CACHE[key] = (kernel_x, kernel_y, laplacian_kernel)
    return _KERNEL_CACHE[key]


def _base_grid(B: int, H: int, W: int, device):
    key = (H, W, _device_key(device))
    if key not in _GRID_CACHE:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        _GRID_CACHE[key] = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    return _GRID_CACHE[key].expand(B, -1, -1, -1)


def compute_traditional_flow(x0, x2, num_iters=5, alpha=1.0):
    kernel_x, kernel_y, laplacian_kernel = _get_flow_kernels(x0.device)

    Ix = F.conv2d(x0, kernel_x, padding=1)
    Iy = F.conv2d(x0, kernel_y, padding=1)
    It = x2 - x0

    u = torch.zeros_like(x0)
    v = torch.zeros_like(x0)

    for _ in range(num_iters):
        u_avg = F.conv2d(u, laplacian_kernel, padding=1)
        v_avg = F.conv2d(v, laplacian_kernel, padding=1)

        der = (Ix * u_avg + Iy * v_avg + It) / (
            alpha**2 + Ix**2 + Iy**2 + 1e-6
        )

        u = u_avg - Ix * der
        v = v_avg - Iy * der

    return torch.cat([u, v], dim=1)


def warp_tensor(x, flow_field):
    B, _, H, W = x.size()
    base_grid = _base_grid(B, H, W, x.device)

    scaled_flow = torch.stack(
        (
            flow_field[:, 0, :, :] / ((W - 1) / 2.0),
            flow_field[:, 1, :, :] / ((H - 1) / 2.0),
        ),
        dim=-1,
    )

    sampling_grid = base_grid + scaled_flow
    return F.grid_sample(x, sampling_grid, mode="bilinear", align_corners=True)


def of_interpolation(x0, x2, num_iters=15):
    computed_flow = compute_traditional_flow(x0, x2, num_iters=num_iters)
    flow_0_to_1 = computed_flow * 0.5
    flow_2_to_1 = -computed_flow * 0.5
    warped_I0 = warp_tensor(x0, flow_0_to_1)
    warped_I2 = warp_tensor(x2, flow_2_to_1)
    return 0.5 * (warped_I0 + warped_I2)
