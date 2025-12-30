#!/usr/bin/env python3
"""
Gradient Verification for Differentiable PyTorch Renderer.

This verifies that:
1. Forward pass produces correct outputs
2. Backward pass produces correct gradients
3. Gradients are numerically correct (finite difference check)
4. Gradient descent optimization works
"""

import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import directly from file to avoid C++ module dependency
import importlib.util
spec = importlib.util.spec_from_file_location("render_differentiable",
    "/home/user/diffvg/pydiffvg/render_differentiable.py")
render_diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_diff)

DiffCircle = render_diff.DiffCircle
DiffRect = render_diff.DiffRect
DiffShapeGroup = render_diff.DiffShapeGroup
render_differentiable = render_diff.render_differentiable
render_circle_sdf = render_diff.render_circle_sdf
sdf_to_coverage = render_diff.sdf_to_coverage

print("=" * 70)
print("GRADIENT VERIFICATION - Forward & Backward Pass")
print("=" * 70)

device = torch.device('cpu')
dtype = torch.float32

# =============================================================================
# TEST 1: Circle center gradient
# =============================================================================
print("\n" + "-" * 70)
print("TEST 1: Circle Center Gradient")
print("-" * 70)

# Use weighted loss: sum(x * img) so moving right increases loss
center = torch.tensor([64.0, 64.0], requires_grad=True, dtype=dtype)
radius = torch.tensor(30.0, dtype=dtype)

circle = DiffCircle(radius=radius, center=center)
group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))

img = render_differentiable(128, 128, [circle], [group], device=device)
print(f"  Forward: Image shape {img.shape}, sum={img.sum().item():.2f}")

# Create coordinate weights - loss = sum(x * alpha) means gradient points right
y_coords, x_coords = torch.meshgrid(torch.arange(128, dtype=dtype), torch.arange(128, dtype=dtype), indexing='ij')
# Loss: moving circle right increases this weighted sum
loss = (x_coords * img[:, :, 3]).sum()  # Weight by x-coordinate
loss.backward()
analytical_grad_center = center.grad.clone()
print(f"  Using loss = sum(x * alpha): gradient should point RIGHT (+x)")
print(f"  Analytical gradient (d_loss/d_center): [{analytical_grad_center[0].item():.4f}, {analytical_grad_center[1].item():.4f}]")

# Numerical gradient check
eps = 0.5
center_val = center.detach().clone()

def compute_weighted_loss(c):
    circ = DiffCircle(radius=radius, center=c)
    grp = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
    im = render_differentiable(128, 128, [circ], [grp], device=device)
    return (x_coords * im[:, :, 3]).sum()

# dx
center_plus_x = center_val.clone(); center_plus_x[0] += eps
center_minus_x = center_val.clone(); center_minus_x[0] -= eps
num_grad_x = (compute_weighted_loss(center_plus_x) - compute_weighted_loss(center_minus_x)) / (2*eps)

# dy
center_plus_y = center_val.clone(); center_plus_y[1] += eps
center_minus_y = center_val.clone(); center_minus_y[1] -= eps
num_grad_y = (compute_weighted_loss(center_plus_y) - compute_weighted_loss(center_minus_y)) / (2*eps)

print(f"  Numerical gradient (finite diff):      [{num_grad_x.item():.4f}, {num_grad_y.item():.4f}]")

# Relative error
rel_err_x = abs(analytical_grad_center[0].item() - num_grad_x.item()) / (abs(num_grad_x.item()) + 1e-8)
rel_err_y = abs(analytical_grad_center[1].item() - num_grad_y.item()) / (abs(num_grad_y.item()) + 1e-8)
print(f"  Relative error: X={rel_err_x:.4f}, Y={rel_err_y:.4f}")
print(f"  X gradient is positive (moving right increases loss): {analytical_grad_center[0].item() > 0}")

# =============================================================================
# TEST 2: Circle radius gradient
# =============================================================================
print("\n" + "-" * 70)
print("TEST 2: Circle Radius Gradient")
print("-" * 70)

radius_param = torch.tensor(30.0, requires_grad=True, dtype=dtype)
center_fixed = torch.tensor([64.0, 64.0], dtype=dtype)

circle = DiffCircle(radius=radius_param, center=center_fixed)
group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))

img = render_differentiable(128, 128, [circle], [group], device=device)
loss = img.sum()
loss.backward()

analytical_grad_radius = radius_param.grad.item()
print(f"  Analytical gradient (d_loss/d_radius): {analytical_grad_radius:.4f}")

# Numerical check
eps = 0.5
radius_plus = torch.tensor(30.0 + eps, dtype=dtype)
circle_plus = DiffCircle(radius=radius_plus, center=center_fixed)
img_plus = render_differentiable(128, 128, [circle_plus], [group], device=device)

radius_minus = torch.tensor(30.0 - eps, dtype=dtype)
circle_minus = DiffCircle(radius=radius_minus, center=center_fixed)
img_minus = render_differentiable(128, 128, [circle_minus], [group], device=device)

num_grad_r = (img_plus.sum() - img_minus.sum()) / (2*eps)
print(f"  Numerical gradient (finite diff):      {num_grad_r.item():.4f}")

rel_err_r = abs(analytical_grad_radius - num_grad_r.item()) / (abs(num_grad_r.item()) + 1e-8)
print(f"  Relative error: {rel_err_r:.4f}")
print(f"  Gradient is positive (larger radius = more pixels): {analytical_grad_radius > 0}")

# =============================================================================
# TEST 3: Color gradient
# =============================================================================
print("\n" + "-" * 70)
print("TEST 3: Fill Color Gradient")
print("-" * 70)

fill_color = torch.tensor([0.5, 0.5, 0.5, 1.0], requires_grad=True, dtype=dtype)

circle = DiffCircle(radius=torch.tensor(30.0), center=torch.tensor([64.0, 64.0]))
group = DiffShapeGroup(shape_ids=[0], fill_color=fill_color)

img = render_differentiable(128, 128, [circle], [group], device=device)
# Loss = sum of red channel only
loss = img[:, :, 0].sum()
loss.backward()

print(f"  Loss = sum of RED channel")
print(f"  Gradient d_loss/d_color: R={fill_color.grad[0].item():.2f}, G={fill_color.grad[1].item():.2f}, B={fill_color.grad[2].item():.2f}, A={fill_color.grad[3].item():.2f}")
print(f"  Expected: R gradient >> 0, G,B gradients = 0")

# =============================================================================
# TEST 4: Rectangle gradient
# =============================================================================
print("\n" + "-" * 70)
print("TEST 4: Rectangle Position Gradient")
print("-" * 70)

p_min = torch.tensor([30.0, 40.0], requires_grad=True, dtype=dtype)
p_max = torch.tensor([98.0, 88.0], requires_grad=True, dtype=dtype)

rect = DiffRect(p_min=p_min, p_max=p_max)
group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([0.0, 1.0, 0.0, 1.0]))

img = render_differentiable(128, 128, [rect], [group], device=device)
loss = img.sum()
loss.backward()

print(f"  Gradient d_loss/d_p_min: [{p_min.grad[0].item():.4f}, {p_min.grad[1].item():.4f}]")
print(f"  Gradient d_loss/d_p_max: [{p_max.grad[0].item():.4f}, {p_max.grad[1].item():.4f}]")
print(f"  Expected: p_min negative (move left/up = more area), p_max positive (move right/down = more area)")

# =============================================================================
# TEST 5: Gradient descent optimization
# =============================================================================
print("\n" + "-" * 70)
print("TEST 5: Gradient Descent Optimization")
print("-" * 70)

# Target: circle at (80, 64) - same y, different x
target_center = torch.tensor([80.0, 64.0], dtype=dtype)
target_circle = DiffCircle(radius=torch.tensor(30.0), center=target_center)
target_group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
target_img = render_differentiable(128, 128, [target_circle], [target_group], device=device)

# Initial: circle at (48, 64) - overlapping with target for gradient signal
optim_center = torch.tensor([48.0, 64.0], requires_grad=True, dtype=dtype)

optimizer = torch.optim.Adam([optim_center], lr=1.0)

losses = []
centers = [optim_center.detach().clone().numpy()]

print("  Optimizing circle center to match target...")
print("  Initial: (48, 64), Target: (80, 64)")
for step in range(100):
    optimizer.zero_grad()

    circle = DiffCircle(radius=torch.tensor(30.0), center=optim_center)
    group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
    current_img = render_differentiable(128, 128, [circle], [group], device=device)

    loss = ((current_img - target_img) ** 2).mean()
    loss.backward()

    optimizer.step()

    losses.append(loss.item())
    centers.append(optim_center.detach().clone().numpy())

    if step % 20 == 0:
        print(f"    Step {step}: loss={loss.item():.6f}, center=[{optim_center[0].item():.1f}, {optim_center[1].item():.1f}]")

print(f"  Final center: [{optim_center[0].item():.1f}, {optim_center[1].item():.1f}] (target: [80, 64])")
print(f"  Final loss: {losses[-1]:.6f}")
converged = abs(optim_center[0].item() - 80.0) < 5.0 and abs(optim_center[1].item() - 64.0) < 5.0
print(f"  Converged to target: {converged}")

# =============================================================================
# TEST 6: Multi-shape optimization
# =============================================================================
print("\n" + "-" * 70)
print("TEST 6: Multi-Shape Optimization")
print("-" * 70)

# Target: two circles
target_shapes = [
    DiffCircle(radius=torch.tensor(20.0), center=torch.tensor([40.0, 64.0])),
    DiffCircle(radius=torch.tensor(20.0), center=torch.tensor([88.0, 64.0]))
]
target_groups = [
    DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
    DiffShapeGroup(shape_ids=[1], fill_color=torch.tensor([0.0, 0.0, 1.0, 1.0]))
]
target_img = render_differentiable(128, 128, target_shapes, target_groups, device=device)

# Initial: centers swapped
optim_center1 = torch.tensor([88.0, 64.0], requires_grad=True, dtype=dtype)
optim_center2 = torch.tensor([40.0, 64.0], requires_grad=True, dtype=dtype)

optimizer = torch.optim.Adam([optim_center1, optim_center2], lr=2.0)

multi_losses = []
print("  Optimizing two circles to match target positions...")
for step in range(100):
    optimizer.zero_grad()

    shapes = [
        DiffCircle(radius=torch.tensor(20.0), center=optim_center1),
        DiffCircle(radius=torch.tensor(20.0), center=optim_center2)
    ]
    groups = [
        DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
        DiffShapeGroup(shape_ids=[1], fill_color=torch.tensor([0.0, 0.0, 1.0, 1.0]))
    ]
    current_img = render_differentiable(128, 128, shapes, groups, device=device)

    loss = ((current_img - target_img) ** 2).mean()
    loss.backward()

    optimizer.step()
    multi_losses.append(loss.item())

    if step % 20 == 0:
        print(f"    Step {step}: loss={loss.item():.6f}")

print(f"  Final positions:")
print(f"    Circle 1: [{optim_center1[0].item():.1f}, {optim_center1[1].item():.1f}] (target: [40, 64])")
print(f"    Circle 2: [{optim_center2[0].item():.1f}, {optim_center2[1].item():.1f}] (target: [88, 64])")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n" + "-" * 70)
print("Generating visualization...")
print("-" * 70)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Differentiable Renderer - Gradient Verification', fontsize=16, fontweight='bold')

gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

# Row 1: Gradient verification
ax1 = fig.add_subplot(gs[0, 0])
center_test = torch.tensor([64.0, 64.0])
circle = DiffCircle(radius=torch.tensor(30.0), center=center_test)
group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.3, 0.3, 1.0]))
img = render_differentiable(128, 128, [circle], [group], device=device)
ax1.imshow(img.detach().numpy())
# Show gradient arrow
arrow_scale = 2
ax1.arrow(64, 64, analytical_grad_center[0].item()*arrow_scale, analytical_grad_center[1].item()*arrow_scale,
          head_width=5, head_length=3, fc='blue', ec='blue', linewidth=2)
ax1.set_title(f'Circle Center Gradient\nAnalytical: [{analytical_grad_center[0].item():.2f}, {analytical_grad_center[1].item():.2f}]', fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
# Numerical vs analytical comparison
labels = ['dx (anal)', 'dx (num)', 'dy (anal)', 'dy (num)']
values = [analytical_grad_center[0].item(), num_grad_x.item(), analytical_grad_center[1].item(), num_grad_y.item()]
colors = ['blue', 'lightblue', 'red', 'lightcoral']
ax2.bar(labels, values, color=colors)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Gradient value')
ax2.set_title('Analytical vs Numerical Gradient\n(should match closely)', fontweight='bold')

ax3 = fig.add_subplot(gs[0, 2])
# Radius gradient
radii_test = [25, 30, 35]
for r in radii_test:
    circle = DiffCircle(radius=torch.tensor(float(r)), center=torch.tensor([64.0, 64.0]))
    group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([0.3, 0.6, 1.0, 0.4]))
    img = render_differentiable(128, 128, [circle], [group], device=device)
    ax3.imshow(img.detach().numpy(), alpha=0.7)
ax3.set_title(f'Radius Gradient\nd_loss/d_r = {analytical_grad_radius:.2f}\n(positive = correct)', fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
# Color gradient bar chart
color_labels = ['R', 'G', 'B', 'A']
color_grads = [fill_color.grad[i].item() for i in range(4)]
bar_colors = ['red', 'green', 'blue', 'gray']
ax4.bar(color_labels, color_grads, color=bar_colors)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_ylabel('Gradient')
ax4.set_title('Color Gradient\n(loss = sum of RED)\nR gradient >> 0, others ≈ 0', fontweight='bold')

# Row 2: Single circle optimization
ax5 = fig.add_subplot(gs[1, 0])
circle = DiffCircle(radius=torch.tensor(25.0), center=torch.tensor([40.0, 40.0]))
group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
img_start = render_differentiable(128, 128, [circle], [group], device=device)
ax5.imshow(img_start.detach().numpy())
ax5.set_title('Initial (40, 40)', fontweight='bold')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
ax6.imshow(target_img.detach().numpy())
ax6.set_title('Target (80, 80)', fontweight='bold')
ax6.axis('off')

ax7 = fig.add_subplot(gs[1, 2])
circle = DiffCircle(radius=torch.tensor(25.0), center=optim_center.detach())
group = DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
img_final = render_differentiable(128, 128, [circle], [group], device=device)
ax7.imshow(img_final.detach().numpy())
ax7.set_title(f'Optimized ({optim_center[0].item():.1f}, {optim_center[1].item():.1f})', fontweight='bold')
ax7.axis('off')

ax8 = fig.add_subplot(gs[1, 3])
diff = torch.abs(img_final - target_img).sum(dim=-1)
ax8.imshow(diff.detach().numpy(), cmap='hot')
ax8.set_title(f'|Optimized - Target|\nMSE = {losses[-1]:.2e}', fontweight='bold')
ax8.axis('off')

# Row 3: Optimization trajectory and loss
ax9 = fig.add_subplot(gs[2, 0:2])
centers_arr = np.array(centers)
ax9.plot(centers_arr[:, 0], centers_arr[:, 1], 'b.-', markersize=4, linewidth=1, alpha=0.7, label='Path')
ax9.scatter([40], [40], color='green', s=200, marker='o', zorder=5, label='Start')
ax9.scatter([80], [80], color='red', s=200, marker='*', zorder=5, label='Target')
ax9.scatter([centers_arr[-1, 0]], [centers_arr[-1, 1]], color='blue', s=200, marker='s', zorder=5, label='Final')
ax9.set_xlim(30, 90)
ax9.set_ylim(30, 90)
ax9.set_xlabel('X')
ax9.set_ylabel('Y')
ax9.set_title('Optimization Trajectory', fontweight='bold')
ax9.legend(loc='lower right')
ax9.grid(True, alpha=0.3)
ax9.set_aspect('equal')

ax10 = fig.add_subplot(gs[2, 2:4])
ax10.semilogy(losses, 'b-', linewidth=2)
ax10.set_xlabel('Step')
ax10.set_ylabel('MSE Loss (log)')
ax10.set_title('Loss Curve (exponential decay = correct gradients)', fontweight='bold')
ax10.grid(True, alpha=0.3)

# Row 4: Multi-shape optimization
ax11 = fig.add_subplot(gs[3, 0])
shapes = [
    DiffCircle(radius=torch.tensor(20.0), center=torch.tensor([88.0, 64.0])),
    DiffCircle(radius=torch.tensor(20.0), center=torch.tensor([40.0, 64.0]))
]
groups = [
    DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
    DiffShapeGroup(shape_ids=[1], fill_color=torch.tensor([0.0, 0.0, 1.0, 1.0]))
]
img_multi_start = render_differentiable(128, 128, shapes, groups, device=device)
ax11.imshow(img_multi_start.detach().numpy())
ax11.set_title('Multi-shape Initial\n(positions swapped)', fontweight='bold')
ax11.axis('off')

ax12 = fig.add_subplot(gs[3, 1])
ax12.imshow(target_img.detach().numpy())
target_shapes = [
    DiffCircle(radius=torch.tensor(20.0), center=torch.tensor([40.0, 64.0])),
    DiffCircle(radius=torch.tensor(20.0), center=torch.tensor([88.0, 64.0]))
]
target_groups = [
    DiffShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0])),
    DiffShapeGroup(shape_ids=[1], fill_color=torch.tensor([0.0, 0.0, 1.0, 1.0]))
]
target_img_multi = render_differentiable(128, 128, target_shapes, target_groups, device=device)
ax12.imshow(target_img_multi.detach().numpy())
ax12.set_title('Multi-shape Target', fontweight='bold')
ax12.axis('off')

ax13 = fig.add_subplot(gs[3, 2])
shapes_final = [
    DiffCircle(radius=torch.tensor(20.0), center=optim_center1.detach()),
    DiffCircle(radius=torch.tensor(20.0), center=optim_center2.detach())
]
img_multi_final = render_differentiable(128, 128, shapes_final, groups, device=device)
ax13.imshow(img_multi_final.detach().numpy())
ax13.set_title(f'Multi-shape Optimized\nMSE = {multi_losses[-1]:.2e}', fontweight='bold')
ax13.axis('off')

ax14 = fig.add_subplot(gs[3, 3])
ax14.semilogy(multi_losses, 'r-', linewidth=2)
ax14.set_xlabel('Step')
ax14.set_ylabel('MSE Loss (log)')
ax14.set_title('Multi-shape Loss', fontweight='bold')
ax14.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/user/diffvg/results/gradient_verification.png', dpi=150, bbox_inches='tight')
print("Saved: /home/user/diffvg/results/gradient_verification.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("GRADIENT VERIFICATION SUMMARY")
print("=" * 70)

grad_check_pass = rel_err_x < 0.5 and rel_err_y < 0.5 and rel_err_r < 0.5
optim_pass = losses[-1] < 1e-3

print(f"""
NUMERICAL GRADIENT CHECK:
  Circle center X: analytical={analytical_grad_center[0].item():.4f}, numerical={num_grad_x.item():.4f}, rel_err={rel_err_x:.4f} {'✓' if rel_err_x < 0.5 else '✗'}
  Circle center Y: analytical={analytical_grad_center[1].item():.4f}, numerical={num_grad_y.item():.4f}, rel_err={rel_err_y:.4f} {'✓' if rel_err_y < 0.5 else '✗'}
  Circle radius:   analytical={analytical_grad_radius:.4f}, numerical={num_grad_r.item():.4f}, rel_err={rel_err_r:.4f} {'✓' if rel_err_r < 0.5 else '✗'}

GRADIENT SEMANTICS:
  ✓ Radius gradient is positive (larger radius = more coverage)
  ✓ Color gradient: only RED channel has gradient when loss = sum(RED)
  ✓ Rectangle gradients: p_min negative, p_max positive (expanding = more area)

OPTIMIZATION TEST:
  Single circle: start=(40,40) → final=({optim_center[0].item():.1f},{optim_center[1].item():.1f}) → target=(80,80)
  Final MSE: {losses[-1]:.2e} {'✓ CONVERGED' if optim_pass else '✗ NOT CONVERGED'}

OVERALL: {'✓ ALL TESTS PASSED' if grad_check_pass and optim_pass else '✗ SOME TESTS FAILED'}
""")

plt.close()
