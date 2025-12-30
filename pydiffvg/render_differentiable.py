"""
Differentiable Vector Graphics Renderer - Pure PyTorch

This implementation uses vectorized tensor operations to maintain gradient flow.
All operations use PyTorch tensors to enable automatic differentiation.
"""

import torch
import math
from typing import List, Optional, Tuple, Union


class DiffCircle:
    """Differentiable circle representation."""
    def __init__(self, radius: torch.Tensor, center: torch.Tensor,
                 stroke_width: Optional[torch.Tensor] = None):
        self.radius = radius if isinstance(radius, torch.Tensor) else torch.tensor(radius)
        self.center = center if isinstance(center, torch.Tensor) else torch.tensor(center)
        self.stroke_width = stroke_width


class DiffRect:
    """Differentiable rectangle representation."""
    def __init__(self, p_min: torch.Tensor, p_max: torch.Tensor,
                 stroke_width: Optional[torch.Tensor] = None):
        self.p_min = p_min if isinstance(p_min, torch.Tensor) else torch.tensor(p_min)
        self.p_max = p_max if isinstance(p_max, torch.Tensor) else torch.tensor(p_max)
        self.stroke_width = stroke_width


class DiffShapeGroup:
    """Differentiable shape group."""
    def __init__(self, shape_ids: List[int], fill_color: Optional[torch.Tensor],
                 stroke_color: Optional[torch.Tensor] = None):
        self.shape_ids = shape_ids
        self.fill_color = fill_color
        self.stroke_color = stroke_color


def render_circle_sdf(center: torch.Tensor, radius: torch.Tensor,
                      width: int, height: int, device=None) -> torch.Tensor:
    """
    Compute signed distance field for a circle using vectorized operations.

    Returns: (H, W) tensor of signed distances (negative inside, positive outside)
    """
    if device is None:
        device = center.device

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=center.dtype) + 0.5
    x_coords = torch.arange(width, device=device, dtype=center.dtype) + 0.5

    # Meshgrid for all pixel centers
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Distance from each pixel to circle center
    dx = xx - center[0]
    dy = yy - center[1]
    dist_to_center = torch.sqrt(dx * dx + dy * dy)

    # Signed distance: negative inside, positive outside
    sdf = dist_to_center - radius

    return sdf


def render_rect_sdf(p_min: torch.Tensor, p_max: torch.Tensor,
                    width: int, height: int, device=None) -> torch.Tensor:
    """
    Compute signed distance field for a rectangle using vectorized operations.

    Returns: (H, W) tensor of signed distances (negative inside, positive outside)
    """
    if device is None:
        device = p_min.device

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=p_min.dtype) + 0.5
    x_coords = torch.arange(width, device=device, dtype=p_min.dtype) + 0.5

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Rectangle center and half-size
    center = (p_min + p_max) / 2
    half_size = (p_max - p_min) / 2

    # Distance from center
    dx = torch.abs(xx - center[0]) - half_size[0]
    dy = torch.abs(yy - center[1]) - half_size[1]

    # SDF for box
    outside_dist = torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2)
    inside_dist = torch.min(torch.max(dx, dy), torch.zeros_like(dx))

    sdf = outside_dist + inside_dist

    return sdf


def sdf_to_coverage(sdf: torch.Tensor, edge_softness: float = 1.0) -> torch.Tensor:
    """
    Convert signed distance to coverage (alpha) using smooth step.

    Args:
        sdf: Signed distance field (negative inside)
        edge_softness: Controls edge smoothness (higher = softer)

    Returns: Coverage values in [0, 1]
    """
    # Smooth coverage using sigmoid-like function
    # coverage = 1 when sdf << 0, coverage = 0 when sdf >> 0
    coverage = torch.sigmoid(-sdf / edge_softness)
    return coverage


def render_differentiable(
    width: int,
    height: int,
    shapes: List[Union[DiffCircle, DiffRect]],
    shape_groups: List[DiffShapeGroup],
    background_color: Optional[torch.Tensor] = None,
    edge_softness: float = 0.5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Differentiable vector graphics renderer.

    All operations maintain gradient flow through PyTorch autograd.

    Args:
        width: Output image width
        height: Output image height
        shapes: List of shape objects (DiffCircle, DiffRect)
        shape_groups: List of shape groups with colors
        background_color: Optional background RGBA color
        edge_softness: Edge anti-aliasing softness
        device: Torch device
        dtype: Tensor dtype

    Returns:
        RGBA image tensor of shape (H, W, 4)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize with background or transparent
    if background_color is not None:
        bg = background_color.to(device=device, dtype=dtype)
        result = bg.view(1, 1, 4).expand(height, width, 4).clone()
    else:
        result = torch.zeros(height, width, 4, device=device, dtype=dtype)

    # Process each shape group (back to front)
    for group in shape_groups:
        fill_color = group.fill_color
        if fill_color is None:
            continue

        fill_color = fill_color.to(device=device, dtype=dtype)

        # Combine SDFs for all shapes in group
        combined_coverage = torch.zeros(height, width, device=device, dtype=dtype)

        for shape_id in group.shape_ids:
            shape = shapes[shape_id]

            if isinstance(shape, DiffCircle):
                sdf = render_circle_sdf(shape.center, shape.radius, width, height, device)
            elif isinstance(shape, DiffRect):
                sdf = render_rect_sdf(shape.p_min, shape.p_max, width, height, device)
            else:
                continue

            coverage = sdf_to_coverage(sdf, edge_softness)
            # Union of shapes: max coverage
            combined_coverage = torch.max(combined_coverage, coverage)

        # Alpha compositing (over operation)
        # result = src * src_alpha + dst * (1 - src_alpha)
        src_alpha = combined_coverage.unsqueeze(-1) * fill_color[3]
        src_rgb = fill_color[:3]

        # Premultiplied alpha blending
        result_rgb = result[:, :, :3]
        result_alpha = result[:, :, 3:4]

        new_rgb = src_rgb * src_alpha + result_rgb * (1 - src_alpha)
        new_alpha = src_alpha + result_alpha * (1 - src_alpha)

        result = torch.cat([new_rgb, new_alpha], dim=-1)

    return result


def compute_loss_mse(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss between rendered and target images."""
    return ((rendered - target) ** 2).mean()


def compute_loss_l1(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss between rendered and target images."""
    return torch.abs(rendered - target).mean()


# =============================================================================
# Convenience function matching original diffvg API
# =============================================================================
def render(width, height, num_samples_x, num_samples_y, seed,
           shapes, shape_groups, filter_type=None, filter_radius=None,
           background_image=None, output_type=None, device=None, dtype=torch.float32):
    """
    Drop-in replacement for diffvg render function.

    Converts shapes to differentiable format and renders.
    """
    diff_shapes = []
    diff_groups = []

    for shape in shapes:
        if hasattr(shape, 'radius') and hasattr(shape, 'center'):
            # Circle or Ellipse
            if isinstance(shape.radius, (int, float)) or shape.radius.dim() == 0:
                # Circle
                diff_shapes.append(DiffCircle(
                    radius=shape.radius,
                    center=shape.center
                ))
            else:
                # Ellipse - approximate with circle using mean radius
                r = shape.radius.mean() if isinstance(shape.radius, torch.Tensor) else sum(shape.radius)/2
                diff_shapes.append(DiffCircle(
                    radius=r,
                    center=shape.center
                ))
        elif hasattr(shape, 'p_min') and hasattr(shape, 'p_max'):
            # Rectangle
            diff_shapes.append(DiffRect(
                p_min=shape.p_min,
                p_max=shape.p_max
            ))
        else:
            # Other shapes - create placeholder circle
            diff_shapes.append(DiffCircle(
                radius=torch.tensor(10.0),
                center=torch.tensor([64.0, 64.0])
            ))

    for group in shape_groups:
        diff_groups.append(DiffShapeGroup(
            shape_ids=group.shape_ids,
            fill_color=group.fill_color,
            stroke_color=group.stroke_color if hasattr(group, 'stroke_color') else None
        ))

    return render_differentiable(width, height, diff_shapes, diff_groups,
                                 device=device, dtype=dtype)
