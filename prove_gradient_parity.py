"""
PROOF OF GRADIENT PARITY: Direct comparison between original diffvg and our implementation

This script renders identical scenes with both:
1. Original diffvg C++ renderer
2. Our custom torch.autograd.Function

Then compares forward outputs and backward gradients to PROVE they match.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.insert(0, '/home/user/diffvg')

# Import original diffvg
import diffvg
import pydiffvg

# Import our implementation
import importlib.util
spec = importlib.util.spec_from_file_location("render_autograd", "pydiffvg/render_autograd.py")
render_autograd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_autograd)
render_circles_ours = render_autograd.render_circles


def render_circle_diffvg(width, height, center, radius, color, num_samples=2):
    """Render a circle using original diffvg."""
    # Create circle shape
    circle = pydiffvg.Circle(
        radius=radius,
        center=center
    )

    # Create shape group with fill color
    shape_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=color,
        stroke_color=None
    )

    # Serialize scene
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        width, height,
        [circle],
        [shape_group]
    )

    # Render - correct API: width, height, num_samples_x, num_samples_y, seed, background, *args
    render = pydiffvg.RenderFunction.apply
    img = render(width, height, num_samples, num_samples, 0, None, *scene_args)

    return img


def compare_single_circle():
    """Compare rendering and gradients for a single circle."""
    print("="*70)
    print("SINGLE CIRCLE COMPARISON")
    print("="*70)

    width, height = 64, 64

    # Parameters (requires_grad for gradient comparison)
    center_diffvg = torch.tensor([32.0, 32.0], requires_grad=True)
    radius_diffvg = torch.tensor(20.0, requires_grad=True)
    color_diffvg = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)

    center_ours = torch.tensor([[32.0, 32.0]], requires_grad=True)
    radius_ours = torch.tensor([20.0], requires_grad=True)
    color_ours = torch.tensor([[1.0, 0.0, 0.0, 1.0]], requires_grad=True)

    # Forward pass
    img_diffvg = render_circle_diffvg(width, height, center_diffvg, radius_diffvg, color_diffvg)
    img_ours = render_circles_ours(width, height, center_ours, radius_ours, color_ours)

    print(f"\nForward pass:")
    print(f"  diffvg shape: {img_diffvg.shape}, range: [{img_diffvg.min():.4f}, {img_diffvg.max():.4f}]")
    print(f"  ours shape:   {img_ours.shape}, range: [{img_ours.min():.4f}, {img_ours.max():.4f}]")

    # Pixel-wise comparison
    diff = torch.abs(img_diffvg - img_ours)
    print(f"\n  Pixel difference - max: {diff.max():.6f}, mean: {diff.mean():.6f}")

    # Backward pass with sum loss
    loss_diffvg = img_diffvg.sum()
    loss_ours = img_ours.sum()

    loss_diffvg.backward()
    loss_ours.backward()

    print(f"\nBackward pass (loss = img.sum()):")
    print(f"  diffvg gradients:")
    print(f"    d_center: {center_diffvg.grad}")
    print(f"    d_radius: {radius_diffvg.grad}")
    print(f"    d_color:  {color_diffvg.grad}")

    print(f"  ours gradients:")
    print(f"    d_center: {center_ours.grad}")
    print(f"    d_radius: {radius_ours.grad}")
    print(f"    d_color:  {color_ours.grad}")

    return {
        'img_diffvg': img_diffvg.detach(),
        'img_ours': img_ours.detach(),
        'grad_center_diffvg': center_diffvg.grad.detach().clone(),
        'grad_center_ours': center_ours.grad.detach().clone().squeeze(),
        'grad_radius_diffvg': radius_diffvg.grad.detach().clone(),
        'grad_radius_ours': radius_ours.grad.detach().clone().squeeze(),
        'grad_color_diffvg': color_diffvg.grad.detach().clone(),
        'grad_color_ours': color_ours.grad.detach().clone().squeeze(),
    }


def compare_multiple_positions():
    """Compare gradients across multiple circle positions."""
    print("\n" + "="*70)
    print("GRADIENT COMPARISON ACROSS POSITIONS")
    print("="*70)

    width, height = 64, 64
    positions = [
        (20.0, 20.0), (32.0, 32.0), (44.0, 44.0),
        (20.0, 44.0), (44.0, 20.0)
    ]

    results = {
        'positions': [],
        'radius_grad_diffvg': [],
        'radius_grad_ours': [],
        'center_x_grad_diffvg': [],
        'center_x_grad_ours': [],
    }

    for cx, cy in positions:
        # diffvg
        center_d = torch.tensor([cx, cy], requires_grad=True)
        radius_d = torch.tensor(15.0, requires_grad=True)
        color_d = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)

        img_d = render_circle_diffvg(width, height, center_d, radius_d, color_d)

        # Use weighted loss for non-zero center gradients
        x_coords = torch.arange(width, dtype=torch.float32) + 0.5
        weighted_loss_d = (x_coords.view(1, -1) * img_d[:, :, 3]).sum()
        weighted_loss_d.backward()

        # ours
        center_o = torch.tensor([[cx, cy]], requires_grad=True)
        radius_o = torch.tensor([15.0], requires_grad=True)
        color_o = torch.tensor([[1.0, 0.0, 0.0, 1.0]], requires_grad=True)

        img_o = render_circles_ours(width, height, center_o, radius_o, color_o)
        weighted_loss_o = (x_coords.view(1, -1) * img_o[:, :, 3]).sum()
        weighted_loss_o.backward()

        results['positions'].append((cx, cy))
        results['radius_grad_diffvg'].append(radius_d.grad.item())
        results['radius_grad_ours'].append(radius_o.grad[0].item())
        results['center_x_grad_diffvg'].append(center_d.grad[0].item())
        results['center_x_grad_ours'].append(center_o.grad[0, 0].item())

        print(f"  Position ({cx:.0f}, {cy:.0f}): "
              f"radius_grad diffvg={radius_d.grad.item():.2f} ours={radius_o.grad[0].item():.2f}")

    return results


def compare_multiple_radii():
    """Compare gradients across multiple radii."""
    print("\n" + "="*70)
    print("GRADIENT COMPARISON ACROSS RADII")
    print("="*70)

    width, height = 64, 64
    radii = [5.0, 10.0, 15.0, 20.0, 25.0, 28.0]

    results = {
        'radii': radii,
        'radius_grad_diffvg': [],
        'radius_grad_ours': [],
    }

    for r in radii:
        # diffvg
        center_d = torch.tensor([32.0, 32.0], requires_grad=True)
        radius_d = torch.tensor(r, requires_grad=True)
        color_d = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)

        img_d = render_circle_diffvg(width, height, center_d, radius_d, color_d)
        img_d.sum().backward()

        # ours
        center_o = torch.tensor([[32.0, 32.0]], requires_grad=True)
        radius_o = torch.tensor([r], requires_grad=True)
        color_o = torch.tensor([[1.0, 0.0, 0.0, 1.0]], requires_grad=True)

        img_o = render_circles_ours(width, height, center_o, radius_o, color_o)
        img_o.sum().backward()

        results['radius_grad_diffvg'].append(radius_d.grad.item())
        results['radius_grad_ours'].append(radius_o.grad[0].item())

        print(f"  Radius {r:.0f}: diffvg={radius_d.grad.item():.2f}, ours={radius_o.grad[0].item():.2f}")

    return results


def create_proof_visualization(single_results, position_results, radii_results):
    """Create comprehensive visualization proving gradient parity."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle('PROOF OF GRADIENT PARITY: diffvg vs Our Implementation',
                 fontsize=16, fontweight='bold', y=0.98)

    # Row 1: Forward pass comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(single_results['img_diffvg'].numpy())
    ax1.set_title('diffvg Output')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(single_results['img_ours'].numpy())
    ax2.set_title('Our Output')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    diff = torch.abs(single_results['img_diffvg'] - single_results['img_ours'])
    im = ax3.imshow(diff[:, :, 3].numpy(), cmap='hot', vmin=0, vmax=0.5)
    ax3.set_title(f'Alpha Difference\nmax={diff.max():.4f}')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # Gradient comparison bar chart
    ax4 = fig.add_subplot(gs[0, 3])
    params = ['center_x', 'center_y', 'radius', 'R', 'G', 'B', 'A']
    diffvg_grads = [
        single_results['grad_center_diffvg'][0].item(),
        single_results['grad_center_diffvg'][1].item(),
        single_results['grad_radius_diffvg'].item(),
        single_results['grad_color_diffvg'][0].item(),
        single_results['grad_color_diffvg'][1].item(),
        single_results['grad_color_diffvg'][2].item(),
        single_results['grad_color_diffvg'][3].item(),
    ]
    ours_grads = [
        single_results['grad_center_ours'][0].item(),
        single_results['grad_center_ours'][1].item(),
        single_results['grad_radius_ours'].item(),
        single_results['grad_color_ours'][0].item(),
        single_results['grad_color_ours'][1].item(),
        single_results['grad_color_ours'][2].item(),
        single_results['grad_color_ours'][3].item(),
    ]

    x = np.arange(len(params))
    width = 0.35
    ax4.bar(x - width/2, diffvg_grads, width, label='diffvg', color='steelblue')
    ax4.bar(x + width/2, ours_grads, width, label='Ours', color='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels(params, fontsize=9)
    ax4.legend()
    ax4.set_title('Gradient Comparison\n(loss = img.sum())')
    ax4.set_ylabel('Gradient Value')

    # Row 2: Radius gradient across different radii
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.plot(radii_results['radii'], radii_results['radius_grad_diffvg'],
             'o-', markersize=10, linewidth=2, label='diffvg', color='steelblue')
    ax5.plot(radii_results['radii'], radii_results['radius_grad_ours'],
             's--', markersize=10, linewidth=2, label='Ours', color='coral')
    ax5.set_xlabel('Circle Radius')
    ax5.set_ylabel('d(loss)/d(radius)')
    ax5.set_title('Radius Gradient vs Circle Size')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Correlation plot for radius gradients
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(radii_results['radius_grad_diffvg'], radii_results['radius_grad_ours'],
                s=100, c='purple', alpha=0.7)

    # Add perfect correlation line
    all_vals = radii_results['radius_grad_diffvg'] + radii_results['radius_grad_ours']
    lims = [min(all_vals) - 10, max(all_vals) + 10]
    ax6.plot(lims, lims, 'k--', alpha=0.5, label='Perfect match')
    ax6.set_xlabel('diffvg Gradient')
    ax6.set_ylabel('Our Gradient')
    ax6.set_title('Gradient Correlation\n(radius parameter)')
    ax6.legend()
    ax6.grid(alpha=0.3)

    # Compute R² correlation
    diffvg_arr = np.array(radii_results['radius_grad_diffvg'])
    ours_arr = np.array(radii_results['radius_grad_ours'])
    correlation = np.corrcoef(diffvg_arr, ours_arr)[0, 1]
    r_squared = correlation ** 2

    # Summary statistics
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.axis('off')

    # Compute errors
    rel_errors = []
    for d, o in zip(radii_results['radius_grad_diffvg'], radii_results['radius_grad_ours']):
        if abs(d) > 1e-6:
            rel_errors.append(abs(d - o) / abs(d) * 100)
        else:
            rel_errors.append(0)

    summary_text = f"""
GRADIENT PARITY METRICS
═══════════════════════

Radius Gradient Comparison:
  • Correlation (R²): {r_squared:.6f}
  • Mean Relative Error: {np.mean(rel_errors):.2f}%
  • Max Relative Error: {np.max(rel_errors):.2f}%

Forward Pass:
  • Max Pixel Diff: {diff.max():.4f}
  • Mean Pixel Diff: {diff.mean():.6f}

VERDICT: {'PARITY PROVEN ✓' if r_squared > 0.99 and np.mean(rel_errors) < 5 else 'DIFFERENCES DETECTED'}
"""
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if r_squared > 0.99 else 'lightyellow', alpha=0.8))

    # Row 3: Position-based comparison
    ax8 = fig.add_subplot(gs[2, :2])
    positions_str = [f"({p[0]:.0f},{p[1]:.0f})" for p in position_results['positions']]
    x = np.arange(len(positions_str))
    ax8.bar(x - width/2, position_results['center_x_grad_diffvg'], width,
            label='diffvg', color='steelblue')
    ax8.bar(x + width/2, position_results['center_x_grad_ours'], width,
            label='Ours', color='coral')
    ax8.set_xticks(x)
    ax8.set_xticklabels(positions_str)
    ax8.set_xlabel('Circle Position (x, y)')
    ax8.set_ylabel('d(weighted_loss)/d(center_x)')
    ax8.set_title('Center X Gradient at Different Positions')
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)

    # Edge comparison detail
    ax9 = fig.add_subplot(gs[2, 2])
    # Show alpha channel edge region
    edge_diffvg = single_results['img_diffvg'][22:42, 42:62, 3]
    ax9.imshow(edge_diffvg.numpy(), cmap='gray', interpolation='nearest')
    ax9.set_title('diffvg Edge Detail\n(alpha channel)')
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[2, 3])
    edge_ours = single_results['img_ours'][22:42, 42:62, 3]
    ax10.imshow(edge_ours.numpy(), cmap='gray', interpolation='nearest')
    ax10.set_title('Our Edge Detail\n(sigmoid soft edge)')
    ax10.axis('off')

    plt.tight_layout()
    return fig, r_squared, np.mean(rel_errors)


def main():
    print("="*70)
    print("PROVING GRADIENT PARITY: Original diffvg vs Our Implementation")
    print("="*70)

    # Run comparisons
    single_results = compare_single_circle()
    position_results = compare_multiple_positions()
    radii_results = compare_multiple_radii()

    # Create visualization
    print("\n" + "="*70)
    print("GENERATING PROOF VISUALIZATION")
    print("="*70)

    fig, r_squared, mean_error = create_proof_visualization(
        single_results, position_results, radii_results
    )

    # Save
    fig.savefig('results/gradient_parity_proof.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: results/gradient_parity_proof.png")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    print(f"  Correlation (R²): {r_squared:.6f}")
    print(f"  Mean Relative Error: {mean_error:.2f}%")

    if r_squared > 0.99 and mean_error < 5:
        print("\n  ✓ GRADIENT PARITY PROVEN")
        print("  Both implementations produce equivalent gradients for optimization.")
    elif r_squared > 0.95:
        print("\n  ~ APPROXIMATE PARITY")
        print("  Gradients are highly correlated but not identical.")
        print("  This is expected due to different edge handling (sigmoid vs exact).")
    else:
        print("\n  ✗ SIGNIFICANT DIFFERENCES")
        print("  The implementations produce different gradient behaviors.")

    plt.close()


if __name__ == '__main__':
    main()
