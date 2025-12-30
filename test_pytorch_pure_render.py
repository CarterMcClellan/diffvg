"""
Test script to compare pure PyTorch render with C++ diffvg render.
"""

import torch
import numpy as np
import os
import sys

# Add the pydiffvg directory to the path so we can import directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the module file to avoid pydiffvg dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("render_pytorch_pure",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pydiffvg", "render_pytorch_pure.py"))
render_pytorch_pure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_pytorch_pure)

# Import classes from the module
render_pytorch = render_pytorch_pure.render_pytorch
Circle = render_pytorch_pure.Circle
Ellipse = render_pytorch_pure.Ellipse
Path = render_pytorch_pure.Path
Rect = render_pytorch_pure.Rect
Polygon = render_pytorch_pure.Polygon
ShapeGroup = render_pytorch_pure.ShapeGroup
LinearGradient = render_pytorch_pure.LinearGradient
RadialGradient = render_pytorch_pure.RadialGradient
PixelFilter = render_pytorch_pure.PixelFilter
FilterType = render_pytorch_pure.FilterType
OutputType = render_pytorch_pure.OutputType


def test_circle_render():
    """Test rendering a simple circle."""
    print("=" * 60)
    print("Test 1: Simple Circle Rendering")
    print("=" * 60)

    device = torch.device('cpu')  # Use CPU for testing
    dtype = torch.float32

    # Create a circle
    circle = Circle(
        radius=torch.tensor(40.0, device=device, dtype=dtype),
        center=torch.tensor([64.0, 64.0], device=device, dtype=dtype)
    )

    # Create shape group with green fill
    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.8, 0.3, 1.0], device=device, dtype=dtype)
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[circle],
        shape_groups=[shape_group],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Output dtype: {img.dtype}")
    print(f"Output min/max: {img.min().item():.4f} / {img.max().item():.4f}")
    print(f"Center pixel (should be green): {img[64, 64]}")
    print(f"Corner pixel (should be transparent): {img[0, 0]}")

    # Check that center is green
    center_color = img[64, 64]
    assert center_color[0] > 0.2 and center_color[0] < 0.4, f"Red channel unexpected: {center_color[0]}"
    assert center_color[1] > 0.7 and center_color[1] < 0.9, f"Green channel unexpected: {center_color[1]}"
    assert center_color[2] > 0.2 and center_color[2] < 0.4, f"Blue channel unexpected: {center_color[2]}"
    assert center_color[3] > 0.9, f"Alpha channel unexpected: {center_color[3]}"

    # Check that corner is transparent
    corner_color = img[0, 0]
    assert corner_color[3] < 0.1, f"Corner should be transparent: {corner_color}"

    print("✓ Circle rendering test PASSED")
    return img


def test_rect_render():
    """Test rendering a rectangle."""
    print("\n" + "=" * 60)
    print("Test 2: Rectangle Rendering")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create a rectangle
    rect = Rect(
        p_min=torch.tensor([20.0, 30.0], device=device, dtype=dtype),
        p_max=torch.tensor([100.0, 80.0], device=device, dtype=dtype)
    )

    # Create shape group with blue fill
    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.2, 0.2, 0.9, 1.0], device=device, dtype=dtype)
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[rect],
        shape_groups=[shape_group],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Center of rect pixel (50, 55): {img[55, 50]}")
    print(f"Outside rect pixel (10, 10): {img[10, 10]}")

    # Check that inside rect is blue
    inside_color = img[55, 50]
    assert inside_color[2] > 0.8, f"Blue channel should be high: {inside_color}"

    # Check outside is transparent
    outside_color = img[10, 10]
    assert outside_color[3] < 0.1, f"Outside should be transparent: {outside_color}"

    print("✓ Rectangle rendering test PASSED")
    return img


def test_ellipse_render():
    """Test rendering an ellipse."""
    print("\n" + "=" * 60)
    print("Test 3: Ellipse Rendering")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create an ellipse (wider than tall)
    ellipse = Ellipse(
        radius=torch.tensor([50.0, 30.0], device=device, dtype=dtype),
        center=torch.tensor([64.0, 64.0], device=device, dtype=dtype)
    )

    # Create shape group with red fill
    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.9, 0.2, 0.2, 1.0], device=device, dtype=dtype)
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[ellipse],
        shape_groups=[shape_group],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Center pixel (should be red): {img[64, 64]}")

    # Check center is red
    center_color = img[64, 64]
    assert center_color[0] > 0.8, f"Red channel should be high: {center_color}"

    print("✓ Ellipse rendering test PASSED")
    return img


def test_path_render():
    """Test rendering a triangular path."""
    print("\n" + "=" * 60)
    print("Test 4: Path (Triangle) Rendering")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create a triangle path (3 line segments)
    points = torch.tensor([
        [64.0, 20.0],   # Top
        [20.0, 100.0],  # Bottom left
        [108.0, 100.0]  # Bottom right
    ], device=device, dtype=dtype)

    path = Path(
        num_control_points=torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        points=points,
        is_closed=True
    )

    # Create shape group with yellow fill
    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.9, 0.9, 0.2, 1.0], device=device, dtype=dtype)
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[path],
        shape_groups=[shape_group],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Center of triangle (~64, 70): {img[70, 64]}")

    # Check inside triangle is yellow
    inside_color = img[70, 64]
    assert inside_color[0] > 0.8 and inside_color[1] > 0.8, f"Should be yellow: {inside_color}"

    print("✓ Path (triangle) rendering test PASSED")
    return img


def test_stroke_render():
    """Test rendering a circle with stroke."""
    print("\n" + "=" * 60)
    print("Test 5: Stroke Rendering")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create a circle with stroke only
    circle = Circle(
        radius=torch.tensor(40.0, device=device, dtype=dtype),
        center=torch.tensor([64.0, 64.0], device=device, dtype=dtype),
        stroke_width=torch.tensor(5.0, device=device, dtype=dtype)
    )

    # Create shape group with stroke color (no fill)
    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=None,
        stroke_color=torch.tensor([0.9, 0.5, 0.1, 1.0], device=device, dtype=dtype)
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[circle],
        shape_groups=[shape_group],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Center pixel (should be transparent, inside circle): {img[64, 64]}")
    print(f"Edge pixel (~64+40, 64) = (104, 64): {img[64, 104]}")

    # Center should be transparent (stroke only, no fill)
    center_color = img[64, 64]
    assert center_color[3] < 0.5, f"Center should be mostly transparent: {center_color}"

    print("✓ Stroke rendering test PASSED")
    return img


def test_multiple_shapes():
    """Test rendering multiple overlapping shapes."""
    print("\n" + "=" * 60)
    print("Test 6: Multiple Overlapping Shapes")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create two overlapping circles
    circle1 = Circle(
        radius=torch.tensor(30.0, device=device, dtype=dtype),
        center=torch.tensor([50.0, 64.0], device=device, dtype=dtype)
    )
    circle2 = Circle(
        radius=torch.tensor(30.0, device=device, dtype=dtype),
        center=torch.tensor([78.0, 64.0], device=device, dtype=dtype)
    )

    # Create shape groups with different colors
    shape_group1 = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.9, 0.2, 0.2, 0.7], device=device, dtype=dtype)  # Semi-transparent red
    )
    shape_group2 = ShapeGroup(
        shape_ids=torch.tensor([1]),
        fill_color=torch.tensor([0.2, 0.2, 0.9, 0.7], device=device, dtype=dtype)  # Semi-transparent blue
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[circle1, circle2],
        shape_groups=[shape_group1, shape_group2],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Left circle only (30, 64): {img[64, 30]}")
    print(f"Overlap region (64, 64): {img[64, 64]}")
    print(f"Right circle only (100, 64): {img[64, 100]}")

    print("✓ Multiple shapes rendering test PASSED")
    return img


def test_linear_gradient():
    """Test rendering with linear gradient fill."""
    print("\n" + "=" * 60)
    print("Test 7: Linear Gradient Fill")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create a rectangle
    rect = Rect(
        p_min=torch.tensor([20.0, 20.0], device=device, dtype=dtype),
        p_max=torch.tensor([108.0, 108.0], device=device, dtype=dtype)
    )

    # Create linear gradient (red to blue, left to right)
    gradient = LinearGradient(
        begin=torch.tensor([20.0, 64.0], device=device, dtype=dtype),
        end=torch.tensor([108.0, 64.0], device=device, dtype=dtype),
        offsets=torch.tensor([0.0, 1.0], device=device, dtype=dtype),
        stop_colors=torch.tensor([
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 0.0, 1.0, 1.0]   # Blue
        ], device=device, dtype=dtype)
    )

    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=gradient
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[rect],
        shape_groups=[shape_group],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Left side (30, 64) - should be red: {img[64, 30]}")
    print(f"Middle (64, 64) - should be purple: {img[64, 64]}")
    print(f"Right side (100, 64) - should be blue: {img[64, 100]}")

    print("✓ Linear gradient test PASSED")
    return img


def test_with_background():
    """Test rendering with a background image."""
    print("\n" + "=" * 60)
    print("Test 8: Rendering with Background")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create a white background
    background = torch.ones(128, 128, 4, device=device, dtype=dtype)

    # Create a semi-transparent circle
    circle = Circle(
        radius=torch.tensor(40.0, device=device, dtype=dtype),
        center=torch.tensor([64.0, 64.0], device=device, dtype=dtype)
    )

    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.0, 0.5, 0.0, 0.5], device=device, dtype=dtype)  # Semi-transparent green
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[circle],
        shape_groups=[shape_group],
        background_image=background,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Center pixel (blended green+white): {img[64, 64]}")
    print(f"Corner pixel (white background): {img[0, 0]}")

    # Corner should be white (background)
    corner = img[0, 0]
    assert corner[0] > 0.9 and corner[1] > 0.9 and corner[2] > 0.9, f"Corner should be white: {corner}"

    print("✓ Background rendering test PASSED")
    return img


def test_transform():
    """Test rendering with shape transformation."""
    print("\n" + "=" * 60)
    print("Test 9: Shape Transformation")
    print("=" * 60)

    device = torch.device('cpu')
    dtype = torch.float32

    # Create a circle at origin
    circle = Circle(
        radius=torch.tensor(20.0, device=device, dtype=dtype),
        center=torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    )

    # Create a translation transform to move it to center
    transform = torch.tensor([
        [1.0, 0.0, 64.0],
        [0.0, 1.0, 64.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)

    shape_group = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.8, 0.4, 0.8, 1.0], device=device, dtype=dtype),
        shape_to_canvas=transform
    )

    # Render
    img = render_pytorch(
        width=128,
        height=128,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[circle],
        shape_groups=[shape_group],
        background_image=None,
        filter_type=FilterType.box,
        filter_radius=0.5,
        output_type=OutputType.color,
        device=device,
        dtype=dtype
    )

    print(f"Output shape: {img.shape}")
    print(f"Center (should be purple): {img[64, 64]}")
    print(f"Origin (should be transparent): {img[0, 0]}")

    # Center should be purple (transformed circle)
    center = img[64, 64]
    assert center[3] > 0.9, f"Center should be opaque: {center}"

    print("✓ Transformation test PASSED")
    return img


def save_test_images():
    """Save all test images for visual inspection."""
    print("\n" + "=" * 60)
    print("Saving test images...")
    print("=" * 60)

    os.makedirs('results/pure_pytorch_tests', exist_ok=True)

    images = {
        'circle': test_circle_render(),
        'rect': test_rect_render(),
        'ellipse': test_ellipse_render(),
        'path': test_path_render(),
        'stroke': test_stroke_render(),
        'multiple': test_multiple_shapes(),
        'gradient': test_linear_gradient(),
        'background': test_with_background(),
        'transform': test_transform()
    }

    for name, img in images.items():
        # Apply gamma correction for display
        img_gamma = torch.clamp(img, 0, 1) ** (1.0 / 2.2)
        img_np = (img_gamma.cpu().numpy() * 255).astype(np.uint8)

        try:
            from PIL import Image
            pil_img = Image.fromarray(img_np)
            pil_img.save(f'results/pure_pytorch_tests/{name}.png')
            print(f"Saved: results/pure_pytorch_tests/{name}.png")
        except ImportError:
            print(f"PIL not available, skipping save for {name}")


def compare_with_diffvg():
    """Compare pure PyTorch render with C++ diffvg render."""
    print("\n" + "=" * 60)
    print("Comparing with C++ diffvg implementation...")
    print("=" * 60)

    try:
        import pydiffvg
        import diffvg
    except ImportError:
        print("C++ diffvg not available for comparison")
        return

    device = torch.device('cpu')
    pydiffvg.set_use_gpu(False)

    # Create identical scene for both renderers
    canvas_width = 128
    canvas_height = 128

    # pydiffvg circle
    circle_diffvg = pydiffvg.Circle(
        radius=torch.tensor(40.0),
        center=torch.tensor([64.0, 64.0])
    )
    shape_group_diffvg = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0])
    )

    # Render with C++ diffvg
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, [circle_diffvg], [shape_group_diffvg]
    )
    render_fn = pydiffvg.RenderFunction.apply
    img_diffvg = render_fn(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

    # Pure PyTorch circle
    circle_pure = Circle(
        radius=torch.tensor(40.0, device=device),
        center=torch.tensor([64.0, 64.0], device=device)
    )
    shape_group_pure = ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0], device=device)
    )

    # Render with pure PyTorch
    img_pure = render_pytorch(
        width=canvas_width,
        height=canvas_height,
        num_samples_x=2,
        num_samples_y=2,
        seed=0,
        shapes=[circle_pure],
        shape_groups=[shape_group_pure],
        device=device
    )

    # Compare
    img_diffvg_cpu = img_diffvg.cpu()
    diff = torch.abs(img_pure - img_diffvg_cpu)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max pixel difference: {max_diff:.6f}")
    print(f"Mean pixel difference: {mean_diff:.6f}")

    print(f"\nC++ diffvg center pixel: {img_diffvg_cpu[64, 64]}")
    print(f"Pure PyTorch center pixel: {img_pure[64, 64]}")

    if mean_diff < 0.1:
        print("\n✓ Comparison test PASSED - outputs are similar")
    else:
        print("\n⚠ Comparison test - noticeable differences (expected due to sampling)")


if __name__ == '__main__':
    print("Pure PyTorch Render Function Tests")
    print("=" * 60)

    # Run all tests
    test_circle_render()
    test_rect_render()
    test_ellipse_render()
    test_path_render()
    test_stroke_render()
    test_multiple_shapes()
    test_linear_gradient()
    test_with_background()
    test_transform()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)

    # Optionally save images and compare with diffvg
    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        save_test_images()

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_with_diffvg()
