"""
Compare analytical implementation vs diffvg for TRUE parity verification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/user/diffvg')

import diffvg
import pydiffvg

# Import our implementations
import importlib.util

# Sigmoid-based (render_autograd.py)
spec1 = importlib.util.spec_from_file_location("render_autograd", "pydiffvg/render_autograd.py")
render_autograd = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(render_autograd)
render_sigmoid = render_autograd.render_circles

# Analytical (render_analytical.py)
spec2 = importlib.util.spec_from_file_location("render_analytical", "pydiffvg/render_analytical.py")
render_analytical_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(render_analytical_mod)
render_analytical = render_analytical_mod.render_analytical


def render_diffvg(width, height, center, radius, color):
    """Render with original diffvg."""
    circle = pydiffvg.Circle(radius=radius, center=center)
    shape_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=color
    )
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        width, height, [circle], [shape_group]
    )
    render = pydiffvg.RenderFunction.apply
    return render(width, height, 2, 2, 0, None, *scene_args)


def main():
    print("="*70)
    print("COMPARISON: diffvg vs Sigmoid vs Analytical")
    print("="*70)

    width, height = 64, 64

    results = {
        'diffvg': {'forward': [], 'radius_grad': [], 'center_grad': []},
        'sigmoid': {'forward': [], 'radius_grad': [], 'center_grad': []},
        'analytical': {'forward': [], 'radius_grad': [], 'center_grad': []},
    }

    radii_test = [10, 15, 20, 25]

    for r in radii_test:
        print(f"\nRadius = {r}")

        # diffvg
        c_d = torch.tensor([32.0, 32.0], requires_grad=True)
        r_d = torch.tensor(float(r), requires_grad=True)
        col_d = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)
        img_d = render_diffvg(width, height, c_d, r_d, col_d)
        img_d.sum().backward()
        results['diffvg']['forward'].append(img_d[..., 3].sum().item())
        results['diffvg']['radius_grad'].append(r_d.grad.item())
        results['diffvg']['center_grad'].append(c_d.grad[0].item())
        print(f"  diffvg:     pixels={img_d[...,3].sum().item():.0f}, d_radius={r_d.grad.item():.2f}")

        # Sigmoid-based
        c_s = torch.tensor([[32.0, 32.0]], requires_grad=True)
        r_s = torch.tensor([float(r)], requires_grad=True)
        col_s = torch.tensor([[1.0, 0.0, 0.0, 1.0]], requires_grad=True)
        img_s = render_sigmoid(width, height, c_s, r_s, col_s)
        img_s.sum().backward()
        results['sigmoid']['forward'].append(img_s[..., 3].sum().item())
        results['sigmoid']['radius_grad'].append(r_s.grad[0].item())
        results['sigmoid']['center_grad'].append(c_s.grad[0, 0].item())
        print(f"  sigmoid:    pixels={img_s[...,3].sum().item():.0f}, d_radius={r_s.grad[0].item():.2f}")

        # Analytical
        c_a = torch.tensor([[32.0, 32.0]], requires_grad=True)
        r_a = torch.tensor([float(r)], requires_grad=True)
        col_a = torch.tensor([[1.0, 0.0, 0.0, 1.0]], requires_grad=True)
        img_a = render_analytical(width, height, c_a, r_a, col_a)
        img_a.sum().backward()
        results['analytical']['forward'].append(img_a[..., 3].sum().item())
        results['analytical']['radius_grad'].append(r_a.grad[0].item())
        results['analytical']['center_grad'].append(c_a.grad[0, 0].item())
        print(f"  analytical: pixels={img_a[...,3].sum().item():.0f}, d_radius={r_a.grad[0].item():.2f}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparison: diffvg vs Sigmoid vs Analytical', fontsize=14, fontweight='bold')

    # Forward pass pixel counts
    ax = axes[0, 0]
    ax.plot(radii_test, results['diffvg']['forward'], 'bo-', label='diffvg', markersize=10)
    ax.plot(radii_test, results['sigmoid']['forward'], 'rs--', label='Sigmoid', markersize=10)
    ax.plot(radii_test, results['analytical']['forward'], 'g^:', label='Analytical', markersize=10)
    ax.set_xlabel('Radius')
    ax.set_ylabel('Total Alpha')
    ax.set_title('Forward Pass: Total Covered Pixels')
    ax.legend()
    ax.grid(alpha=0.3)

    # Radius gradients
    ax = axes[0, 1]
    ax.plot(radii_test, results['diffvg']['radius_grad'], 'bo-', label='diffvg', markersize=10)
    ax.plot(radii_test, results['sigmoid']['radius_grad'], 'rs--', label='Sigmoid', markersize=10)
    ax.plot(radii_test, results['analytical']['radius_grad'], 'g^:', label='Analytical', markersize=10)
    # Expected: 2*pi*r
    expected = [2 * 3.14159 * r for r in radii_test]
    ax.plot(radii_test, expected, 'k--', alpha=0.5, label='2πr (theory)')
    ax.set_xlabel('Radius')
    ax.set_ylabel('d(loss)/d(radius)')
    ax.set_title('Radius Gradient')
    ax.legend()
    ax.grid(alpha=0.3)

    # Correlation plots
    ax = axes[0, 2]
    ax.scatter(results['diffvg']['radius_grad'], results['sigmoid']['radius_grad'],
               s=100, c='red', label='Sigmoid', alpha=0.7)
    ax.scatter(results['diffvg']['radius_grad'], results['analytical']['radius_grad'],
               s=100, c='green', marker='^', label='Analytical', alpha=0.7)
    lims = [0, max(results['diffvg']['radius_grad']) + 20]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('diffvg gradient')
    ax.set_ylabel('Our gradient')
    ax.set_title('Gradient Correlation')
    ax.legend()
    ax.grid(alpha=0.3)

    # Render comparison for r=20
    r = 20
    c_d = torch.tensor([32.0, 32.0])
    r_d = torch.tensor(float(r))
    col_d = torch.tensor([1.0, 0.0, 0.0, 1.0])
    img_d = render_diffvg(width, height, c_d, r_d, col_d)

    img_s = render_sigmoid(width, height,
                           torch.tensor([[32.0, 32.0]]),
                           torch.tensor([float(r)]),
                           torch.tensor([[1.0, 0.0, 0.0, 1.0]]))

    img_a = render_analytical(width, height,
                              torch.tensor([[32.0, 32.0]]),
                              torch.tensor([float(r)]),
                              torch.tensor([[1.0, 0.0, 0.0, 1.0]]))

    axes[1, 0].imshow(img_d.detach().numpy())
    axes[1, 0].set_title('diffvg (multi-sample)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_s.detach().numpy())
    axes[1, 1].set_title('Sigmoid (soft edge)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(img_a.detach().numpy())
    axes[1, 2].set_title('Analytical (binary)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('results/analytical_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/analytical_comparison.png")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Correlation with diffvg
    sig_corr = np.corrcoef(results['diffvg']['radius_grad'],
                           results['sigmoid']['radius_grad'])[0, 1]
    ana_corr = np.corrcoef(results['diffvg']['radius_grad'],
                           results['analytical']['radius_grad'])[0, 1]

    print(f"Sigmoid correlation with diffvg:    R² = {sig_corr**2:.6f}")
    print(f"Analytical correlation with diffvg: R² = {ana_corr**2:.6f}")

    # Mean relative error
    sig_err = np.mean([abs(s - d) / d * 100
                       for s, d in zip(results['sigmoid']['radius_grad'],
                                       results['diffvg']['radius_grad'])])
    ana_err = np.mean([abs(a - d) / d * 100
                       for a, d in zip(results['analytical']['radius_grad'],
                                       results['diffvg']['radius_grad'])])

    print(f"Sigmoid mean relative error:    {sig_err:.2f}%")
    print(f"Analytical mean relative error: {ana_err:.2f}%")

    print("\nConclusion:")
    if sig_err < ana_err:
        print("  Sigmoid implementation is CLOSER to diffvg gradients")
    else:
        print("  Analytical implementation is CLOSER to diffvg gradients")


if __name__ == '__main__':
    main()
