"""
Pure Python/PyTorch implementation of diffvg's render function.
This implementation replicates the C++/CUDA rendering without any native code dependencies.
"""

import torch
import torch.nn.functional as F
from enum import IntEnum
from typing import List, Optional, Union, Tuple
import math


class OutputType(IntEnum):
    color = 1
    sdf = 2


class FilterType(IntEnum):
    box = 0
    tent = 1
    gaussian = 2


# ============================================================================
# Shape Classes (compatible with pydiffvg)
# ============================================================================

class Circle:
    def __init__(self, radius, center, stroke_width=torch.tensor(1.0)):
        self.radius = radius
        self.center = center
        self.stroke_width = stroke_width


class Ellipse:
    def __init__(self, radius, center, stroke_width=torch.tensor(1.0)):
        self.radius = radius
        self.center = center
        self.stroke_width = stroke_width


class Path:
    def __init__(self, num_control_points, points, is_closed,
                 stroke_width=torch.tensor(1.0), use_distance_approx=False):
        self.num_control_points = num_control_points
        self.points = points
        self.is_closed = is_closed
        self.stroke_width = stroke_width
        self.use_distance_approx = use_distance_approx


class Polygon:
    def __init__(self, points, is_closed, stroke_width=torch.tensor(1.0)):
        self.points = points
        self.is_closed = is_closed
        self.stroke_width = stroke_width


class Rect:
    def __init__(self, p_min, p_max, stroke_width=torch.tensor(1.0)):
        self.p_min = p_min
        self.p_max = p_max
        self.stroke_width = stroke_width


class LinearGradient:
    def __init__(self, begin, end, offsets, stop_colors):
        self.begin = begin
        self.end = end
        self.offsets = offsets
        self.stop_colors = stop_colors


class RadialGradient:
    def __init__(self, center, radius, offsets, stop_colors):
        self.center = center
        self.radius = radius
        self.offsets = offsets
        self.stop_colors = stop_colors


class ShapeGroup:
    def __init__(self, shape_ids, fill_color, use_even_odd_rule=True,
                 stroke_color=None, shape_to_canvas=None):
        self.shape_ids = shape_ids
        self.fill_color = fill_color
        self.use_even_odd_rule = use_even_odd_rule
        self.stroke_color = stroke_color
        if shape_to_canvas is None:
            self.shape_to_canvas = torch.eye(3)
        else:
            self.shape_to_canvas = shape_to_canvas


class PixelFilter:
    def __init__(self, type=FilterType.box, radius=0.5):
        self.type = type
        self.radius = radius if isinstance(radius, torch.Tensor) else torch.tensor(radius)


# ============================================================================
# Geometry Utilities
# ============================================================================

def transform_point(point: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply 3x3 affine transformation to 2D point(s)."""
    if point.dim() == 1:
        p = torch.cat([point, torch.ones(1, device=point.device, dtype=point.dtype)])
        transformed = matrix @ p
        return transformed[:2] / transformed[2]
    else:
        ones = torch.ones(point.shape[0], 1, device=point.device, dtype=point.dtype)
        p = torch.cat([point, ones], dim=1)
        transformed = (matrix @ p.T).T
        return transformed[:, :2] / transformed[:, 2:3]


def inverse_transform_point(point: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply inverse of 3x3 affine transformation to 2D point(s)."""
    inv_matrix = torch.linalg.inv(matrix)
    return transform_point(point, inv_matrix)


# ============================================================================
# Bezier Curve Utilities
# ============================================================================

def evaluate_quadratic_bezier(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
                               t: torch.Tensor) -> torch.Tensor:
    """Evaluate quadratic Bezier curve at parameter t."""
    one_minus_t = 1.0 - t
    return one_minus_t * one_minus_t * p0 + 2.0 * one_minus_t * t * p1 + t * t * p2


def evaluate_cubic_bezier(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
                          p3: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier curve at parameter t."""
    one_minus_t = 1.0 - t
    return (one_minus_t ** 3 * p0 +
            3.0 * one_minus_t ** 2 * t * p1 +
            3.0 * one_minus_t * t ** 2 * p2 +
            t ** 3 * p3)


def quadratic_bezier_derivative(p0: torch.Tensor, p1: torch.Tensor,
                                 p2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Derivative of quadratic Bezier curve at parameter t."""
    return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1)


def cubic_bezier_derivative(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor,
                            p3: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Derivative of cubic Bezier curve at parameter t."""
    one_minus_t = 1.0 - t
    return (3.0 * one_minus_t ** 2 * (p1 - p0) +
            6.0 * one_minus_t * t * (p2 - p1) +
            3.0 * t ** 2 * (p3 - p2))


# ============================================================================
# Distance Functions
# ============================================================================

def point_to_line_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute distance from point p to line segment ab."""
    ab = b - a
    ap = p - a
    t = torch.clamp(torch.dot(ap, ab) / (torch.dot(ab, ab) + 1e-10), 0.0, 1.0)
    closest = a + t * ab
    return torch.norm(p - closest)


def point_to_quadratic_bezier_distance(p: torch.Tensor, p0: torch.Tensor,
                                        p1: torch.Tensor, p2: torch.Tensor,
                                        num_samples: int = 16) -> torch.Tensor:
    """Compute approximate distance from point p to quadratic Bezier curve."""
    # Sample the curve and find minimum distance
    t_values = torch.linspace(0, 1, num_samples, device=p.device, dtype=p.dtype)
    min_dist = torch.tensor(float('inf'), device=p.device, dtype=p.dtype)

    for t in t_values:
        curve_point = evaluate_quadratic_bezier(p0, p1, p2, t)
        dist = torch.norm(p - curve_point)
        min_dist = torch.minimum(min_dist, dist)

    return min_dist


def point_to_cubic_bezier_distance(p: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
                                    p2: torch.Tensor, p3: torch.Tensor,
                                    num_samples: int = 16) -> torch.Tensor:
    """Compute approximate distance from point p to cubic Bezier curve."""
    t_values = torch.linspace(0, 1, num_samples, device=p.device, dtype=p.dtype)
    min_dist = torch.tensor(float('inf'), device=p.device, dtype=p.dtype)

    for t in t_values:
        curve_point = evaluate_cubic_bezier(p0, p1, p2, p3, t)
        dist = torch.norm(p - curve_point)
        min_dist = torch.minimum(min_dist, dist)

    return min_dist


# ============================================================================
# Winding Number Computation (Point-in-Polygon Testing)
# ============================================================================

def line_winding_number(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute winding number contribution from line segment ab for point p."""
    # Ray casting from point p in +x direction
    # Check if ray intersects the line segment

    # Early exit if segment is completely above or below the ray
    if (a[1] > p[1] and b[1] > p[1]) or (a[1] <= p[1] and b[1] <= p[1]):
        return torch.tensor(0.0, device=p.device, dtype=p.dtype)

    # Compute x-coordinate of intersection
    t = (p[1] - a[1]) / (b[1] - a[1] + 1e-10)
    x_intersect = a[0] + t * (b[0] - a[0])

    if x_intersect > p[0]:
        # Intersection is to the right of p
        if b[1] > a[1]:
            return torch.tensor(1.0, device=p.device, dtype=p.dtype)
        else:
            return torch.tensor(-1.0, device=p.device, dtype=p.dtype)

    return torch.tensor(0.0, device=p.device, dtype=p.dtype)


def quadratic_bezier_winding_number(p: torch.Tensor, p0: torch.Tensor,
                                     p1: torch.Tensor, p2: torch.Tensor,
                                     num_subdivisions: int = 8) -> torch.Tensor:
    """Compute winding number contribution from quadratic Bezier for point p."""
    # Subdivide into line segments and sum contributions
    winding = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    t_prev = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    prev_point = p0.clone()

    for i in range(1, num_subdivisions + 1):
        t = torch.tensor(i / num_subdivisions, device=p.device, dtype=p.dtype)
        curr_point = evaluate_quadratic_bezier(p0, p1, p2, t)
        winding = winding + line_winding_number(p, prev_point, curr_point)
        prev_point = curr_point

    return winding


def cubic_bezier_winding_number(p: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
                                 p2: torch.Tensor, p3: torch.Tensor,
                                 num_subdivisions: int = 8) -> torch.Tensor:
    """Compute winding number contribution from cubic Bezier for point p."""
    winding = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    prev_point = p0.clone()

    for i in range(1, num_subdivisions + 1):
        t = torch.tensor(i / num_subdivisions, device=p.device, dtype=p.dtype)
        curr_point = evaluate_cubic_bezier(p0, p1, p2, p3, t)
        winding = winding + line_winding_number(p, prev_point, curr_point)
        prev_point = curr_point

    return winding


# ============================================================================
# Shape Distance and Winding Number
# ============================================================================

def circle_signed_distance(p: torch.Tensor, center: torch.Tensor,
                           radius: torch.Tensor) -> torch.Tensor:
    """Signed distance from point p to circle boundary (negative inside)."""
    return torch.norm(p - center) - radius


def circle_winding_number(p: torch.Tensor, center: torch.Tensor,
                          radius: torch.Tensor) -> torch.Tensor:
    """Returns 1 if point is inside circle, 0 otherwise."""
    dist = torch.norm(p - center)
    return torch.where(dist < radius,
                       torch.tensor(1.0, device=p.device, dtype=p.dtype),
                       torch.tensor(0.0, device=p.device, dtype=p.dtype))


def ellipse_signed_distance(p: torch.Tensor, center: torch.Tensor,
                            radius: torch.Tensor) -> torch.Tensor:
    """Approximate signed distance from point p to ellipse boundary."""
    # Normalize point to unit circle space
    normalized = (p - center) / radius
    dist_normalized = torch.norm(normalized)
    # Approximate signed distance
    if dist_normalized < 1e-10:
        return -torch.min(radius)
    # Scale back to ellipse space (approximation)
    return (dist_normalized - 1.0) * torch.min(radius)


def ellipse_winding_number(p: torch.Tensor, center: torch.Tensor,
                           radius: torch.Tensor) -> torch.Tensor:
    """Returns 1 if point is inside ellipse, 0 otherwise."""
    normalized = (p - center) / radius
    dist_sq = torch.sum(normalized ** 2)
    return torch.where(dist_sq < 1.0,
                       torch.tensor(1.0, device=p.device, dtype=p.dtype),
                       torch.tensor(0.0, device=p.device, dtype=p.dtype))


def rect_signed_distance(p: torch.Tensor, p_min: torch.Tensor,
                         p_max: torch.Tensor) -> torch.Tensor:
    """Signed distance from point p to rectangle boundary."""
    center = (p_min + p_max) / 2.0
    half_size = (p_max - p_min) / 2.0

    d = torch.abs(p - center) - half_size
    outside = torch.norm(torch.clamp(d, min=0.0))
    inside = torch.min(torch.clamp(d, max=0.0))
    return outside + inside


def rect_winding_number(p: torch.Tensor, p_min: torch.Tensor,
                        p_max: torch.Tensor) -> torch.Tensor:
    """Returns 1 if point is inside rectangle, 0 otherwise."""
    inside = (p[0] >= p_min[0] and p[0] <= p_max[0] and
              p[1] >= p_min[1] and p[1] <= p_max[1])
    return torch.tensor(1.0 if inside else 0.0, device=p.device, dtype=p.dtype)


def path_signed_distance(p: torch.Tensor, num_control_points: torch.Tensor,
                         points: torch.Tensor, is_closed: bool,
                         num_samples: int = 16) -> torch.Tensor:
    """Compute signed distance from point p to path."""
    min_dist = torch.tensor(float('inf'), device=p.device, dtype=p.dtype)

    point_idx = 0
    num_segments = len(num_control_points)

    for seg_idx in range(num_segments):
        n_ctrl = int(num_control_points[seg_idx].item())

        if n_ctrl == 0:  # Line segment
            p0 = points[point_idx]
            if seg_idx == num_segments - 1 and is_closed:
                p1 = points[0]
            else:
                p1 = points[point_idx + 1]
            dist = point_to_line_distance(p, p0, p1)
            min_dist = torch.minimum(min_dist, dist)
            point_idx += 1

        elif n_ctrl == 1:  # Quadratic Bezier
            p0 = points[point_idx]
            p1 = points[point_idx + 1]  # Control point
            if seg_idx == num_segments - 1 and is_closed:
                p2 = points[0]
            else:
                p2 = points[point_idx + 2]
            dist = point_to_quadratic_bezier_distance(p, p0, p1, p2, num_samples)
            min_dist = torch.minimum(min_dist, dist)
            point_idx += 2

        elif n_ctrl == 2:  # Cubic Bezier
            p0 = points[point_idx]
            p1 = points[point_idx + 1]  # Control point 1
            p2 = points[point_idx + 2]  # Control point 2
            if seg_idx == num_segments - 1 and is_closed:
                p3 = points[0]
            else:
                p3 = points[point_idx + 3]
            dist = point_to_cubic_bezier_distance(p, p0, p1, p2, p3, num_samples)
            min_dist = torch.minimum(min_dist, dist)
            point_idx += 3

    return min_dist


def path_winding_number(p: torch.Tensor, num_control_points: torch.Tensor,
                        points: torch.Tensor, is_closed: bool,
                        num_subdivisions: int = 8) -> torch.Tensor:
    """Compute winding number for point p with respect to path."""
    if not is_closed:
        return torch.tensor(0.0, device=p.device, dtype=p.dtype)

    winding = torch.tensor(0.0, device=p.device, dtype=p.dtype)

    point_idx = 0
    num_segments = len(num_control_points)

    for seg_idx in range(num_segments):
        n_ctrl = int(num_control_points[seg_idx].item())

        if n_ctrl == 0:  # Line segment
            p0 = points[point_idx]
            if seg_idx == num_segments - 1:
                p1 = points[0]
            else:
                p1 = points[point_idx + 1]
            winding = winding + line_winding_number(p, p0, p1)
            point_idx += 1

        elif n_ctrl == 1:  # Quadratic Bezier
            p0 = points[point_idx]
            p1 = points[point_idx + 1]
            if seg_idx == num_segments - 1:
                p2 = points[0]
            else:
                p2 = points[point_idx + 2]
            winding = winding + quadratic_bezier_winding_number(p, p0, p1, p2, num_subdivisions)
            point_idx += 2

        elif n_ctrl == 2:  # Cubic Bezier
            p0 = points[point_idx]
            p1 = points[point_idx + 1]
            p2 = points[point_idx + 2]
            if seg_idx == num_segments - 1:
                p3 = points[0]
            else:
                p3 = points[point_idx + 3]
            winding = winding + cubic_bezier_winding_number(p, p0, p1, p2, p3, num_subdivisions)
            point_idx += 3

    return winding


# ============================================================================
# Color Evaluation
# ============================================================================

def evaluate_constant_color(color: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Evaluate constant color at point p."""
    return color


def evaluate_linear_gradient(gradient: LinearGradient, p: torch.Tensor) -> torch.Tensor:
    """Evaluate linear gradient color at point p."""
    begin = gradient.begin
    end = gradient.end

    # Project point onto gradient line
    direction = end - begin
    length_sq = torch.sum(direction ** 2) + 1e-10
    t = torch.clamp(torch.dot(p - begin, direction) / length_sq, 0.0, 1.0)

    # Find color stops that bracket t
    offsets = gradient.offsets
    stop_colors = gradient.stop_colors

    # Find the bracketing stops
    for i in range(len(offsets) - 1):
        if t >= offsets[i] and t <= offsets[i + 1]:
            local_t = (t - offsets[i]) / (offsets[i + 1] - offsets[i] + 1e-10)
            return (1.0 - local_t) * stop_colors[i] + local_t * stop_colors[i + 1]

    # Clamp to first or last color
    if t <= offsets[0]:
        return stop_colors[0]
    return stop_colors[-1]


def evaluate_radial_gradient(gradient: RadialGradient, p: torch.Tensor) -> torch.Tensor:
    """Evaluate radial gradient color at point p."""
    center = gradient.center
    radius = gradient.radius

    # Compute normalized distance from center
    diff = p - center
    normalized_diff = diff / (radius + 1e-10)
    t = torch.clamp(torch.norm(normalized_diff), 0.0, 1.0)

    # Find color stops that bracket t
    offsets = gradient.offsets
    stop_colors = gradient.stop_colors

    for i in range(len(offsets) - 1):
        if t >= offsets[i] and t <= offsets[i + 1]:
            local_t = (t - offsets[i]) / (offsets[i + 1] - offsets[i] + 1e-10)
            return (1.0 - local_t) * stop_colors[i] + local_t * stop_colors[i + 1]

    if t <= offsets[0]:
        return stop_colors[0]
    return stop_colors[-1]


def evaluate_color(color: Union[torch.Tensor, LinearGradient, RadialGradient, None],
                   p: torch.Tensor) -> Optional[torch.Tensor]:
    """Evaluate color at point p."""
    if color is None:
        return None
    elif isinstance(color, torch.Tensor):
        return color
    elif isinstance(color, LinearGradient):
        return evaluate_linear_gradient(color, p)
    elif isinstance(color, RadialGradient):
        return evaluate_radial_gradient(color, p)
    else:
        return None


# ============================================================================
# Filter Functions
# ============================================================================

def box_filter(x: torch.Tensor, y: torch.Tensor, radius: float) -> torch.Tensor:
    """Box filter weight."""
    if torch.abs(x) <= radius and torch.abs(y) <= radius:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return torch.tensor(0.0, device=x.device, dtype=x.dtype)


def tent_filter(x: torch.Tensor, y: torch.Tensor, radius: float) -> torch.Tensor:
    """Tent (linear) filter weight."""
    dist = torch.sqrt(x ** 2 + y ** 2)
    if dist <= radius:
        return 1.0 - dist / radius
    return torch.tensor(0.0, device=x.device, dtype=x.dtype)


def gaussian_filter(x: torch.Tensor, y: torch.Tensor, radius: float) -> torch.Tensor:
    """Gaussian filter weight."""
    sigma = radius / 3.0
    dist_sq = x ** 2 + y ** 2
    return torch.exp(-dist_sq / (2.0 * sigma ** 2))


def evaluate_filter(filter_type: FilterType, x: torch.Tensor, y: torch.Tensor,
                    radius: float) -> torch.Tensor:
    """Evaluate filter weight at offset (x, y)."""
    if filter_type == FilterType.box:
        return box_filter(x, y, radius)
    elif filter_type == FilterType.tent:
        return tent_filter(x, y, radius)
    elif filter_type == FilterType.gaussian:
        return gaussian_filter(x, y, radius)
    return box_filter(x, y, radius)


# ============================================================================
# Shape Rendering
# ============================================================================

def is_inside_shape(p: torch.Tensor, shape, use_even_odd_rule: bool,
                    transform: torch.Tensor) -> torch.Tensor:
    """Check if point p is inside the shape after applying inverse transform."""
    # Transform point to shape's local space
    p_local = inverse_transform_point(p, transform)

    if isinstance(shape, Circle):
        winding = circle_winding_number(p_local, shape.center, shape.radius)
    elif isinstance(shape, Ellipse):
        winding = ellipse_winding_number(p_local, shape.center, shape.radius)
    elif isinstance(shape, Rect):
        winding = rect_winding_number(p_local, shape.p_min, shape.p_max)
    elif isinstance(shape, (Path, Polygon)):
        if isinstance(shape, Polygon):
            # Convert polygon to path representation
            num_ctrl = torch.zeros(shape.points.shape[0] if shape.is_closed else shape.points.shape[0] - 1,
                                   dtype=torch.int32, device=p.device)
            points = shape.points
            is_closed = shape.is_closed
        else:
            num_ctrl = shape.num_control_points
            points = shape.points
            is_closed = shape.is_closed
        winding = path_winding_number(p_local, num_ctrl, points, is_closed)
    else:
        return torch.tensor(0.0, device=p.device, dtype=p.dtype)

    if use_even_odd_rule:
        # Even-odd rule: inside if winding number is odd
        return torch.abs(winding) % 2.0
    else:
        # Non-zero winding rule: inside if winding number is non-zero
        return torch.where(torch.abs(winding) > 0.5,
                          torch.tensor(1.0, device=p.device, dtype=p.dtype),
                          torch.tensor(0.0, device=p.device, dtype=p.dtype))


def distance_to_shape(p: torch.Tensor, shape, transform: torch.Tensor) -> torch.Tensor:
    """Compute distance from point p to shape boundary."""
    p_local = inverse_transform_point(p, transform)

    if isinstance(shape, Circle):
        return torch.abs(circle_signed_distance(p_local, shape.center, shape.radius))
    elif isinstance(shape, Ellipse):
        return torch.abs(ellipse_signed_distance(p_local, shape.center, shape.radius))
    elif isinstance(shape, Rect):
        return torch.abs(rect_signed_distance(p_local, shape.p_min, shape.p_max))
    elif isinstance(shape, (Path, Polygon)):
        if isinstance(shape, Polygon):
            num_ctrl = torch.zeros(shape.points.shape[0] if shape.is_closed else shape.points.shape[0] - 1,
                                   dtype=torch.int32, device=p.device)
            points = shape.points
            is_closed = shape.is_closed
        else:
            num_ctrl = shape.num_control_points
            points = shape.points
            is_closed = shape.is_closed
        return path_signed_distance(p_local, num_ctrl, points, is_closed)

    return torch.tensor(float('inf'), device=p.device, dtype=p.dtype)


def get_stroke_width(shape, p: torch.Tensor = None) -> torch.Tensor:
    """Get stroke width for shape (may vary along path)."""
    if hasattr(shape, 'stroke_width'):
        sw = shape.stroke_width
        if sw.dim() == 0 or (sw.dim() == 1 and sw.shape[0] == 1):
            return sw if sw.dim() == 0 else sw[0]
        # Variable stroke width - would need interpolation
        return sw.mean()  # Simplified
    return torch.tensor(1.0, device=p.device if p is not None else 'cpu')


# ============================================================================
# Alpha Compositing
# ============================================================================

def alpha_over(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Porter-Duff alpha-over compositing."""
    src_alpha = src[3:4]
    dst_alpha = dst[3:4]

    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)

    if out_alpha < 1e-10:
        return torch.zeros(4, device=src.device, dtype=src.dtype)

    out_rgb = (src[:3] * src_alpha + dst[:3] * dst_alpha * (1.0 - src_alpha)) / out_alpha
    return torch.cat([out_rgb, out_alpha])


# ============================================================================
# Main Render Function
# ============================================================================

def sample_pixel(x: float, y: float, shapes: List, shape_groups: List[ShapeGroup],
                 background: Optional[torch.Tensor], device: torch.device,
                 dtype: torch.dtype) -> torch.Tensor:
    """Sample color at pixel location (x, y)."""
    p = torch.tensor([x, y], device=device, dtype=dtype)

    # Start with background or transparent
    if background is not None:
        # Bilinear interpolation of background
        ix = int(x)
        iy = int(y)
        h, w = background.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            result = background[iy, ix].clone()
        else:
            result = torch.zeros(4, device=device, dtype=dtype)
    else:
        result = torch.zeros(4, device=device, dtype=dtype)

    # Iterate through shape groups (back to front)
    for group in shape_groups:
        transform = group.shape_to_canvas.to(device=device, dtype=dtype)

        # Check each shape in the group
        for shape_id in group.shape_ids:
            shape = shapes[shape_id]

            # Check fill
            if group.fill_color is not None:
                inside = is_inside_shape(p, shape, group.use_even_odd_rule, transform)
                if inside > 0.5:
                    # Transform point to local space for color evaluation
                    p_local = inverse_transform_point(p, transform)
                    fill_color = evaluate_color(group.fill_color, p_local)
                    if fill_color is not None:
                        fill_color = fill_color.to(device=device, dtype=dtype)
                        result = alpha_over(fill_color, result)

            # Check stroke
            if group.stroke_color is not None:
                stroke_width = get_stroke_width(shape, p)
                dist = distance_to_shape(p, shape, transform)
                half_width = stroke_width / 2.0

                if dist <= half_width:
                    p_local = inverse_transform_point(p, transform)
                    stroke_color = evaluate_color(group.stroke_color, p_local)
                    if stroke_color is not None:
                        stroke_color = stroke_color.to(device=device, dtype=dtype)
                        result = alpha_over(stroke_color, result)

    return result


def render_pytorch(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    shapes: List,
    shape_groups: List[ShapeGroup],
    background_image: Optional[torch.Tensor] = None,
    filter_type: FilterType = FilterType.box,
    filter_radius: float = 0.5,
    output_type: OutputType = OutputType.color,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Pure PyTorch implementation of diffvg's render function.

    Args:
        width: Output image width
        height: Output image height
        num_samples_x: Number of samples per pixel in x
        num_samples_y: Number of samples per pixel in y
        seed: Random seed for jittered sampling
        shapes: List of shape objects (Circle, Path, Rect, etc.)
        shape_groups: List of ShapeGroup objects with colors
        background_image: Optional background image [H, W, 4]
        filter_type: Pixel filter type (box, tent, gaussian)
        filter_radius: Pixel filter radius
        output_type: Output type (color or sdf)
        device: Torch device to use
        dtype: Torch dtype to use

    Returns:
        Rendered image tensor [height, width, 4] for color output
        or [height, width, 1] for SDF output
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(seed)

    if output_type == OutputType.color:
        result = torch.zeros(height, width, 4, device=device, dtype=dtype)
    else:
        result = torch.zeros(height, width, 1, device=device, dtype=dtype)

    total_samples = num_samples_x * num_samples_y

    # Generate jitter offsets for stratified sampling
    jitter_x = (torch.rand(num_samples_y, num_samples_x, device=device, dtype=dtype) - 0.5) / num_samples_x
    jitter_y = (torch.rand(num_samples_y, num_samples_x, device=device, dtype=dtype) - 0.5) / num_samples_y

    # Iterate over each pixel
    for py in range(height):
        for px in range(width):
            if output_type == OutputType.color:
                # Accumulate samples
                accumulated = torch.zeros(4, device=device, dtype=dtype)
                weight_sum = torch.tensor(0.0, device=device, dtype=dtype)

                for sy in range(num_samples_y):
                    for sx in range(num_samples_x):
                        # Sample position with stratified jittering
                        sample_x = px + (sx + 0.5) / num_samples_x + jitter_x[sy, sx].item()
                        sample_y = py + (sy + 0.5) / num_samples_y + jitter_y[sy, sx].item()

                        # Compute filter weight
                        offset_x = sample_x - (px + 0.5)
                        offset_y = sample_y - (py + 0.5)
                        weight = evaluate_filter(filter_type,
                                                torch.tensor(offset_x, device=device, dtype=dtype),
                                                torch.tensor(offset_y, device=device, dtype=dtype),
                                                filter_radius)

                        if weight > 0:
                            color = sample_pixel(sample_x, sample_y, shapes, shape_groups,
                                               background_image, device, dtype)
                            accumulated = accumulated + weight * color
                            weight_sum = weight_sum + weight

                if weight_sum > 0:
                    result[py, px] = accumulated / weight_sum
            else:
                # SDF output - compute minimum distance to any shape
                p = torch.tensor([px + 0.5, py + 0.5], device=device, dtype=dtype)
                min_dist = torch.tensor(float('inf'), device=device, dtype=dtype)

                for group in shape_groups:
                    transform = group.shape_to_canvas.to(device=device, dtype=dtype)
                    for shape_id in group.shape_ids:
                        shape = shapes[shape_id]
                        dist = distance_to_shape(p, shape, transform)

                        # Check if inside (make distance negative)
                        inside = is_inside_shape(p, shape, group.use_even_odd_rule, transform)
                        if inside > 0.5:
                            dist = -dist

                        if torch.abs(dist) < torch.abs(min_dist):
                            min_dist = dist

                result[py, px, 0] = min_dist

    return result


# ============================================================================
# Vectorized Render Function (More Efficient)
# ============================================================================

def render_pytorch_vectorized(
    width: int,
    height: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int,
    shapes: List,
    shape_groups: List[ShapeGroup],
    background_image: Optional[torch.Tensor] = None,
    filter_type: FilterType = FilterType.box,
    filter_radius: float = 0.5,
    output_type: OutputType = OutputType.color,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Vectorized pure PyTorch implementation of diffvg's render function.
    More efficient by processing all pixels in parallel where possible.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=dtype)
    x_coords = torch.arange(width, device=device, dtype=dtype)

    if output_type == OutputType.color:
        result = torch.zeros(height, width, 4, device=device, dtype=dtype)
    else:
        result = torch.zeros(height, width, 1, device=device, dtype=dtype)

    # Process samples
    total_samples = num_samples_x * num_samples_y
    weight_sum = torch.zeros(height, width, device=device, dtype=dtype)

    for sy in range(num_samples_y):
        for sx in range(num_samples_x):
            # Sample offset within pixel
            jitter_x = torch.rand(height, width, device=device, dtype=dtype) - 0.5
            jitter_y = torch.rand(height, width, device=device, dtype=dtype) - 0.5

            offset_x = (sx + 0.5) / num_samples_x + jitter_x / num_samples_x - 0.5
            offset_y = (sy + 0.5) / num_samples_y + jitter_y / num_samples_y - 0.5

            sample_x = x_coords.unsqueeze(0).expand(height, -1) + 0.5 + offset_x
            sample_y = y_coords.unsqueeze(1).expand(-1, width) + 0.5 + offset_y

            # Compute filter weights
            if filter_type == FilterType.box:
                weights = torch.ones(height, width, device=device, dtype=dtype)
            elif filter_type == FilterType.tent:
                dist = torch.sqrt(offset_x ** 2 + offset_y ** 2)
                weights = torch.clamp(1.0 - dist / filter_radius, min=0.0)
            else:  # Gaussian
                sigma = filter_radius / 3.0
                dist_sq = offset_x ** 2 + offset_y ** 2
                weights = torch.exp(-dist_sq / (2.0 * sigma ** 2))

            weight_sum = weight_sum + weights

            if output_type == OutputType.color:
                # Sample each pixel
                sample_colors = torch.zeros(height, width, 4, device=device, dtype=dtype)

                # Start with background
                if background_image is not None:
                    sample_colors = background_image.clone()

                # Render each shape group
                for group in shape_groups:
                    transform = group.shape_to_canvas.to(device=device, dtype=dtype)

                    for shape_id in group.shape_ids:
                        shape = shapes[shape_id]

                        # Vectorized inside test and color sampling
                        # For each pixel, check if inside shape
                        for py in range(height):
                            for px in range(width):
                                p = torch.tensor([sample_x[py, px], sample_y[py, px]],
                                               device=device, dtype=dtype)

                                # Fill
                                if group.fill_color is not None:
                                    inside = is_inside_shape(p, shape, group.use_even_odd_rule, transform)
                                    if inside > 0.5:
                                        p_local = inverse_transform_point(p, transform)
                                        fill_color = evaluate_color(group.fill_color, p_local)
                                        if fill_color is not None:
                                            fill_color = fill_color.to(device=device, dtype=dtype)
                                            sample_colors[py, px] = alpha_over(fill_color, sample_colors[py, px])

                                # Stroke
                                if group.stroke_color is not None:
                                    stroke_width = get_stroke_width(shape, p)
                                    dist = distance_to_shape(p, shape, transform)
                                    half_width = stroke_width / 2.0

                                    if dist <= half_width:
                                        p_local = inverse_transform_point(p, transform)
                                        stroke_color = evaluate_color(group.stroke_color, p_local)
                                        if stroke_color is not None:
                                            stroke_color = stroke_color.to(device=device, dtype=dtype)
                                            sample_colors[py, px] = alpha_over(stroke_color, sample_colors[py, px])

                result = result + weights.unsqueeze(-1) * sample_colors

    if output_type == OutputType.color:
        result = result / weight_sum.unsqueeze(-1).clamp(min=1e-10)

    return result


# ============================================================================
# PyTorch Autograd Function (for differentiability)
# ============================================================================

class RenderFunctionPure(torch.autograd.Function):
    """
    Pure PyTorch autograd function for differentiable vector graphics rendering.
    """

    @staticmethod
    def forward(ctx, width, height, num_samples_x, num_samples_y, seed,
                background_image, shapes, shape_groups, filter_type, filter_radius):
        """Forward pass - render the scene."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32

        # Render using the pure PyTorch implementation
        result = render_pytorch(
            width=width,
            height=height,
            num_samples_x=num_samples_x,
            num_samples_y=num_samples_y,
            seed=seed,
            shapes=shapes,
            shape_groups=shape_groups,
            background_image=background_image,
            filter_type=filter_type,
            filter_radius=filter_radius,
            output_type=OutputType.color,
            device=device,
            dtype=dtype
        )

        # Save for backward
        ctx.save_for_backward(result, background_image)
        ctx.shapes = shapes
        ctx.shape_groups = shape_groups
        ctx.width = width
        ctx.height = height
        ctx.num_samples_x = num_samples_x
        ctx.num_samples_y = num_samples_y
        ctx.seed = seed
        ctx.filter_type = filter_type
        ctx.filter_radius = filter_radius

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - compute gradients."""
        # For now, return None for non-differentiable parameters
        # Full gradient computation would require implementing the boundary sampling
        return (None, None, None, None, None,  # width, height, num_samples_x, num_samples_y, seed
                None, None, None, None, None)  # background, shapes, shape_groups, filter_type, filter_radius


# ============================================================================
# Convenience function matching pydiffvg API
# ============================================================================

def render(width: int,
           height: int,
           num_samples_x: int = 2,
           num_samples_y: int = 2,
           seed: int = 0,
           background_image: Optional[torch.Tensor] = None,
           shapes: List = None,
           shape_groups: List[ShapeGroup] = None,
           filter: PixelFilter = None,
           device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convenience function matching the pydiffvg.RenderFunction.apply API.

    Args:
        width: Output image width
        height: Output image height
        num_samples_x: Samples per pixel in x direction
        num_samples_y: Samples per pixel in y direction
        seed: Random seed
        background_image: Optional background [H, W, 4]
        shapes: List of shape objects
        shape_groups: List of ShapeGroup objects
        filter: PixelFilter object
        device: Torch device

    Returns:
        Rendered image [height, width, 4]
    """
    if shapes is None:
        shapes = []
    if shape_groups is None:
        shape_groups = []
    if filter is None:
        filter = PixelFilter(FilterType.box, 0.5)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return render_pytorch(
        width=width,
        height=height,
        num_samples_x=num_samples_x,
        num_samples_y=num_samples_y,
        seed=seed,
        shapes=shapes,
        shape_groups=shape_groups,
        background_image=background_image,
        filter_type=filter.type,
        filter_radius=float(filter.radius),
        output_type=OutputType.color,
        device=device,
        dtype=torch.float32
    )
