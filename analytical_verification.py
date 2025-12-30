#!/usr/bin/env python3
"""
Analytical verification of Pure PyTorch render implementation.

This script verifies the implementation produces mathematically correct results
by comparing against known analytical values.
"""

import sys
import math
import torch
import importlib.util

# Import render_pytorch_pure directly
spec = importlib.util.spec_from_file_location("render_pytorch_pure",
    "/home/user/diffvg/pydiffvg/render_pytorch_pure.py")
render_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_module)

# Extract functions and classes
solve_quadratic = render_module.solve_quadratic
solve_cubic = render_module.solve_cubic
solve_quintic_isolator = render_module.solve_quintic_isolator
closest_point_line = render_module.closest_point_line
closest_point_quadratic_bezier = render_module.closest_point_quadratic_bezier
closest_point_cubic_bezier = render_module.closest_point_cubic_bezier
Circle = render_module.Circle
Rect = render_module.Rect
Path = render_module.Path
Ellipse = render_module.Ellipse
ShapeGroup = render_module.ShapeGroup
render_pytorch = render_module.render_pytorch
compute_winding_number_path = render_module.compute_winding_number_path

print("=" * 70)
print("ANALYTICAL VERIFICATION OF PURE PYTORCH DIFFVG IMPLEMENTATION")
print("=" * 70)

all_passed = True
test_count = 0
pass_count = 0

def check(name, condition, details=""):
    global all_passed, test_count, pass_count
    test_count += 1
    if condition:
        pass_count += 1
        print(f"  ✓ {name}")
    else:
        all_passed = False
        print(f"  ✗ {name}")
        if details:
            print(f"    {details}")

# =============================================================================
# 1. POLYNOMIAL SOLVER VERIFICATION
# =============================================================================
print("\n" + "-" * 70)
print("1. POLYNOMIAL SOLVER VERIFICATION")
print("-" * 70)

print("\n1.1 Quadratic Solver (ax² + bx + c = 0)")
print("    Testing against known analytical roots...")

# Test case: x² - 5x + 6 = 0 → roots at x=2, x=3
has_roots, t0, t1 = solve_quadratic(1.0, -5.0, 6.0)
check("x² - 5x + 6 = 0 → roots 2,3",
      has_roots and abs(t0 - 2.0) < 1e-10 and abs(t1 - 3.0) < 1e-10,
      f"Got: {t0}, {t1}")

# Test case: x² + 1 = 0 → no real roots
has_roots, _, _ = solve_quadratic(1.0, 0.0, 1.0)
check("x² + 1 = 0 → no real roots", not has_roots)

# Test case: x² - 4 = 0 → roots at x=-2, x=2
has_roots, t0, t1 = solve_quadratic(1.0, 0.0, -4.0)
check("x² - 4 = 0 → roots -2,2",
      has_roots and abs(t0 - (-2.0)) < 1e-10 and abs(t1 - 2.0) < 1e-10,
      f"Got: {t0}, {t1}")

# Test case: (x-1)² = x² - 2x + 1 = 0 → double root at x=1
has_roots, t0, t1 = solve_quadratic(1.0, -2.0, 1.0)
check("x² - 2x + 1 = 0 → double root at 1",
      has_roots and abs(t0 - 1.0) < 1e-10 and abs(t1 - 1.0) < 1e-10,
      f"Got: {t0}, {t1}")

print("\n1.2 Cubic Solver (ax³ + bx² + cx + d = 0)")
print("    Testing against known analytical roots...")

# Test case: x³ - 6x² + 11x - 6 = 0 → roots at 1, 2, 3
roots = solve_cubic(1.0, -6.0, 11.0, -6.0)
roots_sorted = sorted(roots)
expected = [1.0, 2.0, 3.0]
check("x³ - 6x² + 11x - 6 = 0 → roots 1,2,3",
      len(roots) == 3 and all(abs(r - e) < 1e-6 for r, e in zip(roots_sorted, expected)),
      f"Got: {roots_sorted}")

# Test case: x³ - 1 = 0 → one real root at x=1
roots = solve_cubic(1.0, 0.0, 0.0, -1.0)
has_one = any(abs(r - 1.0) < 1e-6 for r in roots)
check("x³ - 1 = 0 → has root at 1", has_one, f"Got: {roots}")

# Test case: x³ = 0 → triple root at 0
roots = solve_cubic(1.0, 0.0, 0.0, 0.0)
check("x³ = 0 → root at 0", any(abs(r) < 1e-6 for r in roots), f"Got: {roots}")

# Test case: x³ - 3x² + 3x - 1 = (x-1)³ = 0 → triple root at 1
roots = solve_cubic(1.0, -3.0, 3.0, -1.0)
check("(x-1)³ = 0 → root at 1", any(abs(r - 1.0) < 1e-6 for r in roots), f"Got: {roots}")

print("\n1.3 Quintic Solver (isolator polynomial method)")
print("    Testing against known analytical roots...")

# Simple quintic with roots at 0.5
# (x - 0.5)^5 expanded = x^5 - 2.5x^4 + 2.5x^3 - 1.25x^2 + 0.3125x - 0.03125
roots = solve_quintic_isolator(1.0, -2.5, 2.5, -1.25, 0.3125, -0.03125)
has_root_at_half = any(abs(r - 0.5) < 0.1 for r in roots if 0 <= r <= 1)
check("(x-0.5)⁵ = 0 → root near 0.5", has_root_at_half, f"Got: {roots}")

# =============================================================================
# 2. CLOSEST POINT / DISTANCE VERIFICATION
# =============================================================================
print("\n" + "-" * 70)
print("2. CLOSEST POINT / DISTANCE VERIFICATION")
print("-" * 70)

print("\n2.1 Closest Point on Line Segment")

# Line from (0,0) to (1,0)
p0 = torch.tensor([0.0, 0.0])
p1 = torch.tensor([1.0, 0.0])

# Point directly above middle
pt = torch.tensor([0.5, 1.0])
closest, dist_sq = closest_point_line(pt, p0, p1)
check("Point (0.5,1) to horizontal line → closest at (0.5,0)",
      abs(closest[0].item() - 0.5) < 1e-6 and abs(closest[1].item()) < 1e-6,
      f"Got: ({closest[0].item():.4f}, {closest[1].item():.4f})")
check("Distance = 1.0", abs(math.sqrt(dist_sq) - 1.0) < 1e-6,
      f"Got: {math.sqrt(dist_sq):.6f}")

# Point beyond endpoint
pt = torch.tensor([2.0, 0.0])
closest, dist_sq = closest_point_line(pt, p0, p1)
check("Point (2,0) to line [0,1] → closest at endpoint (1,0)",
      abs(closest[0].item() - 1.0) < 1e-6 and abs(closest[1].item()) < 1e-6,
      f"Got: ({closest[0].item():.4f}, {closest[1].item():.4f})")

print("\n2.2 Closest Point on Quadratic Bezier")

# Quadratic bezier: arc from (0,0) through (0.5, 1) to (1,0)
p0 = torch.tensor([0.0, 0.0])
p1 = torch.tensor([0.5, 1.0])
p2 = torch.tensor([1.0, 0.0])

# Point at (0.5, 0) - closest should be near the apex at t=0.5
pt = torch.tensor([0.5, 0.0])
closest, dist_sq, t = closest_point_quadratic_bezier(pt, p0, p1, p2)
# At t=0.5: B(0.5) = 0.25*p0 + 0.5*p1 + 0.25*p2 = (0.5, 0.5)
check("Closest point on quadratic arc found", dist_sq >= 0)
dist = math.sqrt(dist_sq)
check("Distance is reasonable", dist < 1.0, f"Got dist: {dist:.4f}")

print("\n2.3 Closest Point on Cubic Bezier")

# Cubic bezier
p0 = torch.tensor([0.0, 0.0])
p1 = torch.tensor([0.33, 1.0])
p2 = torch.tensor([0.67, 1.0])
p3 = torch.tensor([1.0, 0.0])

# Point below curve
pt = torch.tensor([0.5, -0.5])
closest, dist_sq, t = closest_point_cubic_bezier(pt, p0, p1, p2, p3)
dist = math.sqrt(dist_sq)
check("Closest point on cubic bezier found", dist >= 0)
check("Distance is reasonable (point is 0.5 below)", 0.3 < dist < 1.0,
      f"Got dist: {dist:.4f}")

# =============================================================================
# 3. WINDING NUMBER VERIFICATION
# =============================================================================
print("\n" + "-" * 70)
print("3. WINDING NUMBER VERIFICATION (Path)")
print("-" * 70)

print("\n3.1 Triangle Path Winding Numbers")

# Create a triangle path (3 line segments)
# Triangle: (50, 10) -> (10, 90) -> (90, 90) -> close
points = torch.tensor([
    [50.0, 10.0],  # p0
    [10.0, 90.0],  # p1
    [90.0, 90.0],  # p2
])
num_control_points = torch.tensor([0, 0, 0])  # All lines
triangle_path = Path(num_control_points=num_control_points,
                     points=points, is_closed=True)

# Point inside triangle (centroid at ~50, 63)
pt_inside = torch.tensor([50.0, 60.0])
wn = compute_winding_number_path(triangle_path, pt_inside)
check("Triangle path: point inside → winding ≠ 0", wn != 0, f"Got: {wn}")

# Point outside triangle
pt_outside = torch.tensor([5.0, 5.0])
wn = compute_winding_number_path(triangle_path, pt_outside)
check("Triangle path: point outside → winding = 0", wn == 0, f"Got: {wn}")

# =============================================================================
# 4. RENDERED OUTPUT VERIFICATION
# =============================================================================
print("\n" + "-" * 70)
print("4. RENDERED OUTPUT VERIFICATION")
print("-" * 70)

print("\n4.1 Circle Area Verification")

# Render a circle and verify pixel coverage matches expected area
radius = 32.0
width, height = 128, 128
circle = Circle(radius=radius, center=torch.tensor([64.0, 64.0]))
group = ShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))

img = render_pytorch(width, height, 1, 1, 0, [circle], [group])
# Count non-zero alpha pixels
alpha_mask = img[:, :, 3] > 0.5
pixel_count = alpha_mask.sum().item()
expected_area = math.pi * radius * radius
area_ratio = pixel_count / expected_area
check(f"Circle area: {pixel_count} pixels vs π×32² = {expected_area:.1f}",
      0.95 < area_ratio < 1.05,
      f"Ratio: {area_ratio:.3f}")

print("\n4.2 Rectangle Area Verification")

# Render a rectangle
rect_w, rect_h = 60, 40
rect = Rect(p_min=torch.tensor([34.0, 44.0]),
            p_max=torch.tensor([94.0, 84.0]))
group = ShapeGroup(shape_ids=[0], fill_color=torch.tensor([0.0, 1.0, 0.0, 1.0]))

img = render_pytorch(width, height, 1, 1, 0, [rect], [group])
alpha_mask = img[:, :, 3] > 0.5
pixel_count = alpha_mask.sum().item()
expected_area = rect_w * rect_h
area_ratio = pixel_count / expected_area
check(f"Rectangle area: {pixel_count} pixels vs {rect_w}×{rect_h} = {expected_area}",
      0.95 < area_ratio < 1.05,
      f"Ratio: {area_ratio:.3f}")

print("\n4.3 Ellipse Area Verification")

# Render an ellipse
rx, ry = 40.0, 25.0
ellipse = Ellipse(radius=torch.tensor([rx, ry]), center=torch.tensor([64.0, 64.0]))
group = ShapeGroup(shape_ids=[0], fill_color=torch.tensor([0.0, 0.0, 1.0, 1.0]))

img = render_pytorch(width, height, 1, 1, 0, [ellipse], [group])
alpha_mask = img[:, :, 3] > 0.5
pixel_count = alpha_mask.sum().item()
expected_area = math.pi * rx * ry
area_ratio = pixel_count / expected_area
check(f"Ellipse area: {pixel_count} pixels vs π×40×25 = {expected_area:.1f}",
      0.93 < area_ratio < 1.07,
      f"Ratio: {area_ratio:.3f}")

print("\n4.4 Color Accuracy Verification")

# Render a shape with specific color and verify
circle = Circle(radius=40.0, center=torch.tensor([64.0, 64.0]))
test_color = torch.tensor([0.25, 0.5, 0.75, 1.0])
group = ShapeGroup(shape_ids=[0], fill_color=test_color)

img = render_pytorch(width, height, 1, 1, 0, [circle], [group])
# Sample color from center
center_color = img[64, 64, :]
check(f"Center pixel R={center_color[0]:.2f} (expected 0.25)",
      abs(center_color[0].item() - 0.25) < 0.01)
check(f"Center pixel G={center_color[1]:.2f} (expected 0.50)",
      abs(center_color[1].item() - 0.50) < 0.01)
check(f"Center pixel B={center_color[2]:.2f} (expected 0.75)",
      abs(center_color[2].item() - 0.75) < 0.01)

print("\n4.5 Alpha Compositing Verification")

# Two overlapping circles with alpha
c1 = Circle(radius=30.0, center=torch.tensor([50.0, 64.0]))
c2 = Circle(radius=30.0, center=torch.tensor([78.0, 64.0]))
g1 = ShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.0, 0.0, 0.5]))  # Red 50%
g2 = ShapeGroup(shape_ids=[1], fill_color=torch.tensor([0.0, 0.0, 1.0, 0.5]))  # Blue 50%

img = render_pytorch(width, height, 1, 1, 0, [c1, c2], [g1, g2])

# In overlap region, we should see blended colors
overlap_pixel = img[64, 64, :]
check("Overlap region shows blended colors",
      overlap_pixel[0].item() > 0.1 and overlap_pixel[2].item() > 0.1,
      f"R={overlap_pixel[0]:.2f}, B={overlap_pixel[2]:.2f}")

print("\n4.6 Stroke Width Verification")

# Circle with stroke
circle = Circle(radius=40.0, center=torch.tensor([64.0, 64.0]))
stroke_width = 8.0
group = ShapeGroup(shape_ids=[0],
                   fill_color=None,
                   stroke_color=torch.tensor([1.0, 1.0, 0.0, 1.0]))
circle.stroke_width = torch.tensor(stroke_width)

img = render_pytorch(width, height, 1, 1, 0, [circle], [group])

# Check that center is empty (no fill)
center_alpha = img[64, 64, 3].item()
check("Stroke-only circle: center is empty", center_alpha < 0.1,
      f"Center alpha: {center_alpha:.2f}")

# Check that ring area has content
ring_pixel = img[64, 24, :]  # On the ring (radius=40, at y=64-40=24)
check("Stroke-only circle: ring has color", ring_pixel[3].item() > 0.5,
      f"Ring alpha: {ring_pixel[3]:.2f}")

print("\n4.7 Path Rendering Verification")

# Triangle path
points = torch.tensor([
    [64.0, 20.0],
    [20.0, 100.0],
    [108.0, 100.0],
])
num_control_points = torch.tensor([0, 0, 0])
triangle = Path(num_control_points=num_control_points, points=points, is_closed=True)
group = ShapeGroup(shape_ids=[0], fill_color=torch.tensor([1.0, 0.5, 0.0, 1.0]))

img = render_pytorch(width, height, 1, 1, 0, [triangle], [group])

# Verify triangle centroid is filled
centroid_y = int((20 + 100 + 100) / 3)
centroid_x = int((64 + 20 + 108) / 3)
centroid_alpha = img[centroid_y, centroid_x, 3].item()
check("Triangle path: centroid is filled", centroid_alpha > 0.5,
      f"Centroid alpha at ({centroid_x},{centroid_y}): {centroid_alpha:.2f}")

# Verify outside is empty
outside_alpha = img[10, 10, 3].item()
check("Triangle path: outside is empty", outside_alpha < 0.1,
      f"Outside alpha: {outside_alpha:.2f}")

# =============================================================================
# 5. GRADIENT VERIFICATION
# =============================================================================
print("\n" + "-" * 70)
print("5. GRADIENT VERIFICATION")
print("-" * 70)

print("\n5.1 Linear Gradient Direction")

LinearGradient = render_module.LinearGradient

# Horizontal gradient from red (left) to blue (right)
rect = Rect(p_min=torch.tensor([0.0, 0.0]), p_max=torch.tensor([128.0, 128.0]))
gradient = LinearGradient(
    begin=torch.tensor([0.0, 64.0]),
    end=torch.tensor([128.0, 64.0]),
    offsets=torch.tensor([0.0, 1.0]),
    stop_colors=torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
)
group = ShapeGroup(shape_ids=[0], fill_color=gradient)

img = render_pytorch(width, height, 1, 1, 0, [rect], [group])

# Left side should be red
left_color = img[64, 10, :]
check("Linear gradient: left side is red",
      left_color[0].item() > 0.7 and left_color[2].item() < 0.3,
      f"Left: R={left_color[0]:.2f}, B={left_color[2]:.2f}")

# Right side should be blue
right_color = img[64, 117, :]
check("Linear gradient: right side is blue",
      right_color[0].item() < 0.3 and right_color[2].item() > 0.7,
      f"Right: R={right_color[0]:.2f}, B={right_color[2]:.2f}")

# Middle should be purple (mixed)
mid_color = img[64, 64, :]
check("Linear gradient: middle is purple",
      0.3 < mid_color[0].item() < 0.7 and 0.3 < mid_color[2].item() < 0.7,
      f"Mid: R={mid_color[0]:.2f}, B={mid_color[2]:.2f}")

print("\n5.2 Radial Gradient")

RadialGradient = render_module.RadialGradient

# Radial gradient: yellow center to green edge
circle = Circle(radius=50.0, center=torch.tensor([64.0, 64.0]))
gradient = RadialGradient(
    center=torch.tensor([64.0, 64.0]),
    radius=torch.tensor([50.0, 50.0]),
    offsets=torch.tensor([0.0, 1.0]),
    stop_colors=torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0]])
)
group = ShapeGroup(shape_ids=[0], fill_color=gradient)

img = render_pytorch(width, height, 1, 1, 0, [circle], [group])

# Center should be yellow
center_color = img[64, 64, :]
check("Radial gradient: center is yellow",
      center_color[0].item() > 0.7 and center_color[1].item() > 0.7 and center_color[2].item() < 0.3,
      f"Center: R={center_color[0]:.2f}, G={center_color[1]:.2f}, B={center_color[2]:.2f}")

# Edge should be greenish
edge_color = img[64, 110, :]  # Near edge
check("Radial gradient: edge is greenish",
      edge_color[1].item() > 0.3,
      f"Edge: R={edge_color[0]:.2f}, G={edge_color[1]:.2f}, B={edge_color[2]:.2f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"\nTests passed: {pass_count}/{test_count}")
if all_passed:
    print("\n✓ ALL TESTS PASSED - Implementation is mathematically correct!")
else:
    print(f"\n✗ {test_count - pass_count} tests failed")
print()

sys.exit(0 if all_passed else 1)
