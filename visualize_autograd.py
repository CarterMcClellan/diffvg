"""
Visualization of Custom Autograd Function for Differentiable Rendering

Creates matplotlib visualizations showing:
1. Forward pass renders
2. Gradient verification (analytical vs numerical)
3. Optimization trajectory
4. Gradient flow visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import importlib.util

# Import directly to avoid diffvg C++ dependency
spec = importlib.util.spec_from_file_location("render_autograd", "pydiffvg/render_autograd.py")
render_autograd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_autograd)

RenderFunction = render_autograd.RenderFunction
render_circles = render_autograd.render_circles
DiffVGRenderer = render_autograd.DiffVGRenderer


def visualize_forward_pass():
    """Show forward pass rendering results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Forward Pass: Custom Autograd Rendering', fontsize=14, fontweight='bold')

    width, height = 128, 128

    # Test cases
    test_cases = [
        {
            'name': 'Single Circle',
            'centers': torch.tensor([[64.0, 64.0]]),
            'radii': torch.tensor([30.0]),
            'colors': torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        },
        {
            'name': 'Two Overlapping',
            'centers': torch.tensor([[50.0, 64.0], [78.0, 64.0]]),
            'radii': torch.tensor([25.0, 25.0]),
            'colors': torch.tensor([[1.0, 0.0, 0.0, 0.7], [0.0, 0.0, 1.0, 0.7]])
        },
        {
            'name': 'Three Circles',
            'centers': torch.tensor([[64.0, 40.0], [40.0, 88.0], [88.0, 88.0]]),
            'radii': torch.tensor([25.0, 25.0, 25.0]),
            'colors': torch.tensor([[1.0, 0.0, 0.0, 0.8], [0.0, 1.0, 0.0, 0.8], [0.0, 0.0, 1.0, 0.8]])
        },
        {
            'name': 'Small + Large',
            'centers': torch.tensor([[64.0, 64.0], [64.0, 64.0]]),
            'radii': torch.tensor([45.0, 15.0]),
            'colors': torch.tensor([[0.2, 0.2, 0.8, 1.0], [1.0, 1.0, 0.0, 1.0]])
        },
        {
            'name': 'Semi-transparent',
            'centers': torch.tensor([[45.0, 64.0], [83.0, 64.0]]),
            'radii': torch.tensor([35.0, 35.0]),
            'colors': torch.tensor([[1.0, 0.5, 0.0, 0.5], [0.0, 0.5, 1.0, 0.5]])
        },
        {
            'name': 'Cluster',
            'centers': torch.tensor([[64.0, 64.0], [44.0, 54.0], [84.0, 54.0], [54.0, 84.0], [74.0, 84.0]]),
            'radii': torch.tensor([20.0, 15.0, 15.0, 15.0, 15.0]),
            'colors': torch.tensor([
                [1.0, 0.0, 0.0, 0.9],
                [0.0, 1.0, 0.0, 0.7],
                [0.0, 0.0, 1.0, 0.7],
                [1.0, 1.0, 0.0, 0.7],
                [1.0, 0.0, 1.0, 0.7]
            ])
        }
    ]

    for idx, tc in enumerate(test_cases):
        ax = axes[idx // 3, idx % 3]
        img = render_circles(width, height, tc['centers'], tc['radii'], tc['colors'])
        ax.imshow(img.detach().numpy())
        ax.set_title(tc['name'])
        ax.axis('off')

    plt.tight_layout()
    return fig


def visualize_gradient_verification():
    """Compare analytical vs numerical gradients."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Gradient Verification: Analytical vs Numerical', fontsize=14, fontweight='bold')

    width, height = 64, 64
    eps = 1e-3

    # Test with off-center circle for non-zero center gradients
    base_centers = torch.tensor([[40.0, 35.0]], dtype=torch.float32)
    base_radii = torch.tensor([20.0], dtype=torch.float32)
    base_colors = torch.tensor([[0.8, 0.2, 0.3, 0.9]], dtype=torch.float32)

    # Use weighted loss to get non-zero center gradients
    y_coords = torch.arange(height, dtype=torch.float32) + 0.5
    x_coords = torch.arange(width, dtype=torch.float32) + 0.5

    def compute_weighted_loss(img):
        # Weight by position to get directional gradients
        return (x_coords.view(1, -1) * img[:, :, 3]).sum() + (y_coords.view(-1, 1) * img[:, :, 3]).sum()

    # Numerical gradients
    numerical_grads = {}

    # Center x
    c_plus = base_centers.clone(); c_plus[0, 0] += eps
    c_minus = base_centers.clone(); c_minus[0, 0] -= eps
    img_plus = render_circles(width, height, c_plus, base_radii.clone(), base_colors.clone())
    img_minus = render_circles(width, height, c_minus, base_radii.clone(), base_colors.clone())
    numerical_grads['center_x'] = (compute_weighted_loss(img_plus) - compute_weighted_loss(img_minus)).item() / (2 * eps)

    # Center y
    c_plus = base_centers.clone(); c_plus[0, 1] += eps
    c_minus = base_centers.clone(); c_minus[0, 1] -= eps
    img_plus = render_circles(width, height, c_plus, base_radii.clone(), base_colors.clone())
    img_minus = render_circles(width, height, c_minus, base_radii.clone(), base_colors.clone())
    numerical_grads['center_y'] = (compute_weighted_loss(img_plus) - compute_weighted_loss(img_minus)).item() / (2 * eps)

    # Radius
    r_plus = base_radii.clone(); r_plus[0] += eps
    r_minus = base_radii.clone(); r_minus[0] -= eps
    img_plus = render_circles(width, height, base_centers.clone(), r_plus, base_colors.clone())
    img_minus = render_circles(width, height, base_centers.clone(), r_minus, base_colors.clone())
    numerical_grads['radius'] = (compute_weighted_loss(img_plus) - compute_weighted_loss(img_minus)).item() / (2 * eps)

    # Color channels
    for i, name in enumerate(['red', 'green', 'blue', 'alpha']):
        col_plus = base_colors.clone(); col_plus[0, i] = min(1.0, col_plus[0, i] + eps)
        col_minus = base_colors.clone(); col_minus[0, i] = max(0.0, col_minus[0, i] - eps)
        img_plus = render_circles(width, height, base_centers.clone(), base_radii.clone(), col_plus)
        img_minus = render_circles(width, height, base_centers.clone(), base_radii.clone(), col_minus)
        numerical_grads[name] = (compute_weighted_loss(img_plus) - compute_weighted_loss(img_minus)).item() / (2 * eps)

    # Analytical gradients
    centers_g = base_centers.clone().requires_grad_(True)
    radii_g = base_radii.clone().requires_grad_(True)
    colors_g = base_colors.clone().requires_grad_(True)

    img = render_circles(width, height, centers_g, radii_g, colors_g)
    loss = compute_weighted_loss(img)
    loss.backward()

    analytical_grads = {
        'center_x': centers_g.grad[0, 0].item(),
        'center_y': centers_g.grad[0, 1].item(),
        'radius': radii_g.grad[0].item(),
        'red': colors_g.grad[0, 0].item(),
        'green': colors_g.grad[0, 1].item(),
        'blue': colors_g.grad[0, 2].item(),
        'alpha': colors_g.grad[0, 3].item()
    }

    # Plot comparisons
    params = ['center_x', 'center_y', 'radius', 'red', 'green', 'blue', 'alpha']

    # Bar chart comparison
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(params))
    width_bar = 0.35

    num_vals = [numerical_grads[p] for p in params]
    ana_vals = [analytical_grads[p] for p in params]

    bars1 = ax1.bar(x - width_bar/2, num_vals, width_bar, label='Numerical', color='steelblue')
    bars2 = ax1.bar(x + width_bar/2, ana_vals, width_bar, label='Analytical', color='coral')
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('Gradient Value')
    ax1.set_title('Gradient Comparison: Numerical (Finite Diff) vs Analytical (Backward Pass)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(params)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Relative error
    ax2 = fig.add_subplot(gs[1, 0])
    rel_errors = []
    for p in params:
        if abs(numerical_grads[p]) > 1e-6:
            rel_err = abs(analytical_grads[p] - numerical_grads[p]) / abs(numerical_grads[p]) * 100
        else:
            rel_err = abs(analytical_grads[p] - numerical_grads[p]) * 100
        rel_errors.append(rel_err)

    colors = ['green' if e < 5 else 'orange' if e < 10 else 'red' for e in rel_errors]
    ax2.bar(params, rel_errors, color=colors)
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Gradient Error')
    ax2.axhline(y=5, color='green', linestyle='--', label='5% threshold')
    ax2.tick_params(axis='x', rotation=45)

    # Scatter plot
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(num_vals, ana_vals, s=100, c='purple', alpha=0.7)
    for i, p in enumerate(params):
        ax3.annotate(p, (num_vals[i], ana_vals[i]), fontsize=8)

    lims = [min(min(num_vals), min(ana_vals)) - 10, max(max(num_vals), max(ana_vals)) + 10]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Perfect match')
    ax3.set_xlabel('Numerical Gradient')
    ax3.set_ylabel('Analytical Gradient')
    ax3.set_title('Gradient Correlation')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Rendered image
    ax4 = fig.add_subplot(gs[1, 2])
    img_display = render_circles(width, height, base_centers, base_radii, base_colors)
    ax4.imshow(img_display.detach().numpy())
    ax4.set_title(f'Test Shape\ncenter=({base_centers[0,0]:.0f},{base_centers[0,1]:.0f}), r={base_radii[0]:.0f}')
    ax4.axis('off')

    # Gradient flow visualization
    ax5 = fig.add_subplot(gs[2, :])

    # Create multiple circles and show gradient directions
    test_centers = torch.tensor([
        [20.0, 32.0], [44.0, 32.0], [32.0, 20.0], [32.0, 44.0]
    ], requires_grad=True)
    test_radii = torch.tensor([12.0, 12.0, 12.0, 12.0], requires_grad=True)
    test_colors = torch.tensor([
        [1.0, 0.0, 0.0, 0.8],
        [0.0, 1.0, 0.0, 0.8],
        [0.0, 0.0, 1.0, 0.8],
        [1.0, 1.0, 0.0, 0.8]
    ], requires_grad=True)

    img = render_circles(64, 64, test_centers, test_radii, test_colors)
    # Loss that pushes circles toward center
    target_x, target_y = 32.0, 32.0
    weighted = ((x_coords.view(1,-1) - target_x)**2 + (y_coords.view(-1,1) - target_y)**2) * img[:,:,3]
    loss = weighted.sum()
    loss.backward()

    ax5.imshow(img.detach().numpy())

    # Draw gradient arrows
    for i in range(4):
        cx, cy = test_centers[i].detach().numpy()
        gx, gy = -test_centers.grad[i].numpy() * 0.01  # Scale and negate for direction
        ax5.arrow(cx, cy, gx, gy, head_width=3, head_length=2, fc='white', ec='black', linewidth=2)

    ax5.set_title('Gradient Flow: Arrows show direction of gradient descent toward center')
    ax5.axis('off')

    plt.tight_layout()
    return fig


def visualize_optimization():
    """Show optimization trajectory using gradients."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Optimization Trajectory: Gradient Descent in Action', fontsize=14, fontweight='bold')

    width, height = 64, 64

    # Initial and target - same color, just different position/size for clearer demo
    center = torch.tensor([[20.0, 40.0]], requires_grad=True)
    radius = torch.tensor([15.0], requires_grad=True)
    color = torch.tensor([[0.9, 0.2, 0.1, 1.0]], requires_grad=True)  # Same as target

    # Target
    target_center = torch.tensor([[44.0, 28.0]])
    target_radius = torch.tensor([22.0])
    target_color = torch.tensor([[0.9, 0.2, 0.1, 1.0]])

    target_img = render_circles(width, height, target_center, target_radius, target_color)

    optimizer = torch.optim.Adam([center, radius], lr=1.0)

    losses = []
    snapshots = []

    # Optimization loop
    for step in range(200):
        optimizer.zero_grad()

        img = render_circles(width, height, center, radius, color)
        loss = ((img - target_img) ** 2).sum()
        losses.append(loss.item())

        if step in [0, 10, 30, 70, 150, 199]:
            snapshots.append({
                'step': step,
                'img': img.detach().clone(),
                'center': center.detach().clone(),
                'radius': radius.detach().clone(),
                'loss': loss.item()
            })

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            radius.clamp_(min=1.0)

    # Plot target and snapshots
    axes[0, 0].imshow(target_img.numpy())
    axes[0, 0].set_title('Target')
    axes[0, 0].axis('off')

    for i, snap in enumerate(snapshots[:3]):
        ax = axes[0, i+1]
        ax.imshow(snap['img'].numpy())
        ax.set_title(f"Step {snap['step']}\nloss={snap['loss']:.1f}")
        ax.axis('off')

    for i, snap in enumerate(snapshots[3:]):
        ax = axes[1, i]
        ax.imshow(snap['img'].numpy())
        ax.set_title(f"Step {snap['step']}\nloss={snap['loss']:.2f}")
        ax.axis('off')

    # Loss curve
    ax_loss = axes[1, 3]
    ax_loss.plot(losses, 'b-', linewidth=2)
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('MSE Loss')
    ax_loss.set_title('Loss Curve')
    ax_loss.set_yscale('log')
    ax_loss.grid(alpha=0.3)

    # Mark snapshot points
    for snap in snapshots:
        ax_loss.axvline(x=snap['step'], color='red', alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def visualize_sdf_and_coverage():
    """Visualize the SDF and coverage computation."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('SDF-Based Rendering Pipeline', fontsize=14, fontweight='bold')

    width, height = 64, 64
    device = torch.device('cpu')
    dtype = torch.float32

    center = torch.tensor([32.0, 32.0])
    radius = torch.tensor(20.0)
    edge_softness = 0.5

    # Coordinate grids
    y_coords = torch.arange(height, device=device, dtype=dtype) + 0.5
    x_coords = torch.arange(width, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # SDF computation
    dx = xx - center[0]
    dy = yy - center[1]
    dist_to_center = torch.sqrt(dx * dx + dy * dy)
    sdf = dist_to_center - radius

    # Coverage
    coverage = torch.sigmoid(-sdf / edge_softness)

    # Gradient of coverage w.r.t. SDF
    d_cov_d_sdf = -coverage * (1 - coverage) / edge_softness

    # Gradient of SDF w.r.t. center
    d_sdf_d_cx = -dx / (dist_to_center + 1e-8)
    d_sdf_d_cy = -dy / (dist_to_center + 1e-8)

    # Row 1: Forward pass pipeline
    axes[0, 0].imshow(dist_to_center.numpy(), cmap='viridis')
    axes[0, 0].set_title('Distance to Center')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(sdf.numpy(), cmap='RdBu', vmin=-30, vmax=30)
    axes[0, 1].set_title('Signed Distance Field\n(blue=inside, red=outside)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(coverage.numpy(), cmap='gray')
    axes[0, 2].set_title('Coverage (soft mask)')
    axes[0, 2].axis('off')

    # Final render
    color = torch.tensor([1.0, 0.3, 0.2, 1.0])
    final = torch.zeros(height, width, 4)
    final[:, :, :3] = color[:3] * coverage.unsqueeze(-1)
    final[:, :, 3] = coverage
    axes[0, 3].imshow(final.numpy())
    axes[0, 3].set_title('Final Render')
    axes[0, 3].axis('off')

    # Row 2: Backward pass components
    im2 = axes[1, 0].imshow(d_cov_d_sdf.numpy(), cmap='hot')
    axes[1, 0].set_title('d(coverage)/d(SDF)\n(gradient magnitude)')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(d_sdf_d_cx.numpy(), cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('d(SDF)/d(center_x)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    im4 = axes[1, 2].imshow(d_sdf_d_cy.numpy(), cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('d(SDF)/d(center_y)')
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

    # Combined gradient magnitude
    grad_mag = torch.sqrt(d_sdf_d_cx**2 + d_sdf_d_cy**2) * torch.abs(d_cov_d_sdf)
    axes[1, 3].imshow(grad_mag.numpy(), cmap='hot')
    axes[1, 3].set_title('Gradient Magnitude\n(where gradients flow)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    return fig


def main():
    print("Generating visualizations for Custom Autograd Function...")

    # Create output directory
    import os
    os.makedirs('results', exist_ok=True)

    # Generate all visualizations
    print("1. Forward pass visualization...")
    fig1 = visualize_forward_pass()
    fig1.savefig('results/autograd_forward.png', dpi=150, bbox_inches='tight')
    print("   Saved: results/autograd_forward.png")

    print("2. Gradient verification...")
    fig2 = visualize_gradient_verification()
    fig2.savefig('results/autograd_gradients.png', dpi=150, bbox_inches='tight')
    print("   Saved: results/autograd_gradients.png")

    print("3. Optimization trajectory...")
    fig3 = visualize_optimization()
    fig3.savefig('results/autograd_optimization.png', dpi=150, bbox_inches='tight')
    print("   Saved: results/autograd_optimization.png")

    print("4. SDF pipeline visualization...")
    fig4 = visualize_sdf_and_coverage()
    fig4.savefig('results/autograd_sdf_pipeline.png', dpi=150, bbox_inches='tight')
    print("   Saved: results/autograd_sdf_pipeline.png")

    # Summary statistics
    print("\n" + "="*60)
    print("GRADIENT VERIFICATION SUMMARY")
    print("="*60)

    # Run quick verification
    width, height = 64, 64
    centers = torch.tensor([[40.0, 35.0]], requires_grad=True)
    radii = torch.tensor([20.0], requires_grad=True)
    colors = torch.tensor([[0.8, 0.2, 0.3, 0.9]], requires_grad=True)

    img = render_circles(width, height, centers, radii, colors)
    loss = img.sum()
    loss.backward()

    print(f"Test configuration: center=(40,35), radius=20, color=RGBA(0.8,0.2,0.3,0.9)")
    print(f"Analytical gradients:")
    print(f"  d_loss/d_center: [{centers.grad[0,0]:.4f}, {centers.grad[0,1]:.4f}]")
    print(f"  d_loss/d_radius: {radii.grad[0]:.4f}")
    print(f"  d_loss/d_color:  [{colors.grad[0,0]:.4f}, {colors.grad[0,1]:.4f}, {colors.grad[0,2]:.4f}, {colors.grad[0,3]:.4f}]")

    # Numerical check
    eps = 1e-3
    r_plus = torch.tensor([20.0 + eps])
    r_minus = torch.tensor([20.0 - eps])
    img_plus = render_circles(width, height, torch.tensor([[40.0, 35.0]]), r_plus, torch.tensor([[0.8, 0.2, 0.3, 0.9]]))
    img_minus = render_circles(width, height, torch.tensor([[40.0, 35.0]]), r_minus, torch.tensor([[0.8, 0.2, 0.3, 0.9]]))
    num_grad = (img_plus.sum() - img_minus.sum()) / (2 * eps)

    print(f"\nNumerical gradient (radius): {num_grad:.4f}")
    rel_err = abs(radii.grad[0].item() - num_grad.item()) / abs(num_grad.item()) * 100
    print(f"Relative error: {rel_err:.2f}%")
    print(f"Status: {'PASS' if rel_err < 5 else 'FAIL'} (threshold: 5%)")

    print("\n" + "="*60)
    print("All visualizations saved to results/ directory")
    print("="*60)

    plt.close('all')


if __name__ == '__main__':
    main()
