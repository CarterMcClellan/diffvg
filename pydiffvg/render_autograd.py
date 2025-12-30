"""
Differentiable Vector Graphics Renderer - Custom Autograd Function

This implements a proper torch.autograd.Function with hand-written backward pass
that matches diffvg's gradient computation approach.

Key concepts from diffvg:
1. Forward: Compute SDF-based coverage and composite colors
2. Backward: Propagate gradients through coverage → SDF → shape parameters
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional, Any


class RenderFunction(torch.autograd.Function):
    """
    Custom autograd function for differentiable rendering.

    Forward pass:
        - Compute signed distance field (SDF) for each shape
        - Convert SDF to soft coverage using sigmoid
        - Composite colors using alpha blending

    Backward pass:
        - d_loss/d_output is the incoming gradient
        - Propagate through alpha blending → coverage → SDF → shape params
    """

    @staticmethod
    def forward(ctx, width: int, height: int,
                centers: torch.Tensor,      # (N, 2) circle centers
                radii: torch.Tensor,        # (N,) circle radii
                colors: torch.Tensor,       # (N, 4) RGBA colors
                edge_softness: float = 0.5) -> torch.Tensor:
        """
        Render circles to an image.

        Args:
            width, height: Image dimensions
            centers: (N, 2) tensor of circle centers
            radii: (N,) tensor of circle radii
            colors: (N, 4) tensor of RGBA colors
            edge_softness: Softness of edges for anti-aliasing

        Returns:
            (H, W, 4) RGBA image
        """
        device = centers.device
        dtype = centers.dtype
        num_circles = centers.shape[0]

        # Create coordinate grids
        y_coords = torch.arange(height, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(width, device=device, dtype=dtype) + 0.5
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)

        # Compute SDF for each circle: (N, H, W)
        # SDF = distance_to_center - radius (negative inside, positive outside)
        dx = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1)  # (N, H, W)
        dy = yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)  # (N, H, W)
        dist_to_center = torch.sqrt(dx * dx + dy * dy)  # (N, H, W)
        sdf = dist_to_center - radii.view(-1, 1, 1)  # (N, H, W)

        # Convert SDF to coverage using sigmoid (soft step function)
        # coverage = 1 when inside (sdf < 0), coverage = 0 when outside (sdf > 0)
        coverage = torch.sigmoid(-sdf / edge_softness)  # (N, H, W)

        # Alpha compositing (front to back)
        result = torch.zeros(height, width, 4, device=device, dtype=dtype)

        for i in range(num_circles):
            src_alpha = coverage[i:i+1, :, :].permute(1, 2, 0) * colors[i, 3]  # (H, W, 1)
            src_rgb = colors[i, :3]  # (3,)

            # Porter-Duff over: result = src + dst * (1 - src_alpha)
            result[:, :, :3] = src_rgb * src_alpha + result[:, :, :3] * (1 - src_alpha)
            result[:, :, 3:] = src_alpha + result[:, :, 3:] * (1 - src_alpha)

        # Save for backward
        ctx.save_for_backward(centers, radii, colors, sdf, coverage)
        ctx.edge_softness = edge_softness
        ctx.width = width
        ctx.height = height

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: compute gradients for centers, radii, and colors.

        The gradient chain is:
            d_loss/d_param = d_loss/d_output * d_output/d_coverage * d_coverage/d_sdf * d_sdf/d_param

        Where:
            d_coverage/d_sdf = sigmoid'(-sdf/edge_softness) * (-1/edge_softness)
                             = coverage * (1 - coverage) / edge_softness
            d_sdf/d_center = -(pixel - center) / distance
            d_sdf/d_radius = -1
        """
        centers, radii, colors, sdf, coverage = ctx.saved_tensors
        edge_softness = ctx.edge_softness
        width, height = ctx.width, ctx.height

        device = centers.device
        dtype = centers.dtype
        num_circles = centers.shape[0]

        # Initialize gradients
        d_centers = torch.zeros_like(centers)
        d_radii = torch.zeros_like(radii)
        d_colors = torch.zeros_like(colors)

        # Create coordinate grids
        y_coords = torch.arange(height, device=device, dtype=dtype) + 0.5
        x_coords = torch.arange(width, device=device, dtype=dtype) + 0.5
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Recompute distance components for gradient
        dx = xx.unsqueeze(0) - centers[:, 0].view(-1, 1, 1)  # (N, H, W)
        dy = yy.unsqueeze(0) - centers[:, 1].view(-1, 1, 1)  # (N, H, W)
        dist_to_center = torch.sqrt(dx * dx + dy * dy + 1e-8)  # (N, H, W), add eps for numerical stability

        # Gradient of sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        # coverage = sigmoid(-sdf / e), so d_coverage/d_sdf = sigmoid' * (-1/e) = -cov*(1-cov)/e
        d_coverage_d_sdf = -coverage * (1 - coverage) / edge_softness  # (N, H, W)

        # Process each circle (simplified - assumes independent compositing for gradients)
        for i in range(num_circles):
            # grad_output is (H, W, 4)
            # We need d_loss/d_coverage for this circle

            # For the coverage of circle i:
            # result_rgb = color_rgb * coverage * color_alpha + prev_rgb * (1 - coverage * color_alpha)
            # d_result_rgb/d_coverage = color_rgb * color_alpha - prev_rgb * color_alpha
            #                         ≈ color_rgb * color_alpha (simplified)

            color_alpha = colors[i, 3]
            color_rgb = colors[i, :3]

            # d_loss/d_coverage_i ≈ sum over channels of (grad_output * d_output/d_coverage)
            # Simplified: just use the alpha-weighted color contribution
            d_loss_d_coverage = (
                grad_output[:, :, 0] * color_rgb[0] * color_alpha +
                grad_output[:, :, 1] * color_rgb[1] * color_alpha +
                grad_output[:, :, 2] * color_rgb[2] * color_alpha +
                grad_output[:, :, 3] * color_alpha
            )  # (H, W)

            # d_loss/d_sdf = d_loss/d_coverage * d_coverage/d_sdf
            d_loss_d_sdf = d_loss_d_coverage * d_coverage_d_sdf[i]  # (H, W)

            # d_sdf/d_center_x = d/d_cx (sqrt((x-cx)^2 + (y-cy)^2) - r)
            #                  = -(x - cx) / distance
            d_sdf_d_cx = -dx[i] / dist_to_center[i]  # (H, W)
            d_sdf_d_cy = -dy[i] / dist_to_center[i]  # (H, W)

            # d_sdf/d_radius = -1
            d_sdf_d_r = -torch.ones_like(sdf[i])  # (H, W)

            # Accumulate gradients
            d_centers[i, 0] = (d_loss_d_sdf * d_sdf_d_cx).sum()
            d_centers[i, 1] = (d_loss_d_sdf * d_sdf_d_cy).sum()
            d_radii[i] = (d_loss_d_sdf * d_sdf_d_r).sum()

            # Color gradients
            cov = coverage[i]  # (H, W)
            # d_output/d_color_rgb = coverage * alpha (approximately)
            d_colors[i, 0] = (grad_output[:, :, 0] * cov * color_alpha).sum()
            d_colors[i, 1] = (grad_output[:, :, 1] * cov * color_alpha).sum()
            d_colors[i, 2] = (grad_output[:, :, 2] * cov * color_alpha).sum()
            # d_output/d_alpha is more complex but simplified here
            d_colors[i, 3] = (
                (grad_output[:, :, 0] * cov * color_rgb[0]).sum() +
                (grad_output[:, :, 1] * cov * color_rgb[1]).sum() +
                (grad_output[:, :, 2] * cov * color_rgb[2]).sum() +
                (grad_output[:, :, 3] * cov).sum()
            )

        # Return gradients (same order as forward arguments, None for non-tensor args)
        return None, None, d_centers, d_radii, d_colors, None


class DiffVGRenderer(nn.Module):
    """
    PyTorch module wrapper for the differentiable renderer.
    """

    def __init__(self, width: int, height: int, edge_softness: float = 0.5):
        super().__init__()
        self.width = width
        self.height = height
        self.edge_softness = edge_softness

    def forward(self, centers: torch.Tensor, radii: torch.Tensor,
                colors: torch.Tensor) -> torch.Tensor:
        """
        Render circles.

        Args:
            centers: (N, 2) tensor of circle centers
            radii: (N,) tensor of circle radii
            colors: (N, 4) tensor of RGBA colors

        Returns:
            (H, W, 4) RGBA image
        """
        return RenderFunction.apply(
            self.width, self.height, centers, radii, colors, self.edge_softness
        )


def render_circles(width: int, height: int,
                   centers: torch.Tensor,
                   radii: torch.Tensor,
                   colors: torch.Tensor,
                   edge_softness: float = 0.5) -> torch.Tensor:
    """Convenience function for rendering circles."""
    return RenderFunction.apply(width, height, centers, radii, colors, edge_softness)


# =============================================================================
# Testing and Verification
# =============================================================================

def numerical_gradient_check(width, height, centers, radii, colors, eps=1e-3):
    """
    Verify analytical gradients against numerical gradients.
    """
    results = {}

    # Test center gradients
    for i in range(centers.shape[0]):
        for j in range(2):  # x and y
            centers_plus = centers.clone()
            centers_plus[i, j] += eps
            img_plus = render_circles(width, height, centers_plus, radii.clone(), colors.clone())

            centers_minus = centers.clone()
            centers_minus[i, j] -= eps
            img_minus = render_circles(width, height, centers_minus, radii.clone(), colors.clone())

            numerical_grad = (img_plus.sum() - img_minus.sum()) / (2 * eps)
            results[f'center_{i}_{["x","y"][j]}_numerical'] = numerical_grad.item()

    # Test radius gradients
    for i in range(radii.shape[0]):
        radii_plus = radii.clone()
        radii_plus[i] += eps
        img_plus = render_circles(width, height, centers.clone(), radii_plus, colors.clone())

        radii_minus = radii.clone()
        radii_minus[i] -= eps
        img_minus = render_circles(width, height, centers.clone(), radii_minus, colors.clone())

        numerical_grad = (img_plus.sum() - img_minus.sum()) / (2 * eps)
        results[f'radius_{i}_numerical'] = numerical_grad.item()

    # Get analytical gradients
    centers_grad = centers.clone().requires_grad_(True)
    radii_grad = radii.clone().requires_grad_(True)
    colors_grad = colors.clone().requires_grad_(True)

    img = render_circles(width, height, centers_grad, radii_grad, colors_grad)
    loss = img.sum()
    loss.backward()

    for i in range(centers.shape[0]):
        results[f'center_{i}_x_analytical'] = centers_grad.grad[i, 0].item()
        results[f'center_{i}_y_analytical'] = centers_grad.grad[i, 1].item()

    for i in range(radii.shape[0]):
        results[f'radius_{i}_analytical'] = radii_grad.grad[i].item()

    return results


if __name__ == '__main__':
    # Quick test
    print("Testing RenderFunction...")

    width, height = 64, 64
    centers = torch.tensor([[32.0, 32.0]], requires_grad=True)
    radii = torch.tensor([20.0], requires_grad=True)
    colors = torch.tensor([[1.0, 0.0, 0.0, 1.0]], requires_grad=True)

    img = render_circles(width, height, centers, radii, colors)
    print(f"Output shape: {img.shape}")
    print(f"Output sum: {img.sum().item():.2f}")

    # Test backward
    loss = img.sum()
    loss.backward()

    print(f"d_loss/d_center: {centers.grad}")
    print(f"d_loss/d_radius: {radii.grad}")
    print(f"d_loss/d_color: {colors.grad}")

    # Numerical check
    print("\nNumerical gradient check:")
    results = numerical_gradient_check(width, height,
                                       torch.tensor([[32.0, 32.0]]),
                                       torch.tensor([20.0]),
                                       torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")
