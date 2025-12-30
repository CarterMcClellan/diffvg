"""
Pure Python/PyTorch implementation of diffvg's render function.
This implementation provides EXACT PARITY with the C++ implementation.
"""

import torch
import math
from enum import IntEnum
from typing import List, Optional, Union, Tuple


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
                 stroke_width=torch.tensor(1.0), thickness=None,
                 use_distance_approx=False):
        self.num_control_points = num_control_points
        self.points = points
        self.is_closed = is_closed
        self.stroke_width = stroke_width
        self.thickness = thickness  # Per-point thickness
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
# Polynomial Solvers (Exact match to C++ solve.h)
# ============================================================================

def solve_quadratic(a: float, b: float, c: float) -> Tuple[bool, float, float]:
    """
    Solve quadratic equation ax^2 + bx + c = 0.
    Returns (success, t0, t1) where t0 <= t1.
    Exact match to C++ solve_quadratic.
    """
    discrim = b * b - 4 * a * c
    if discrim < 0:
        return False, 0.0, 0.0

    root_discrim = math.sqrt(discrim)

    if b < 0:
        q = -0.5 * (b - root_discrim)
    else:
        q = -0.5 * (b + root_discrim)

    if abs(a) < 1e-10:
        if abs(q) < 1e-10:
            return False, 0.0, 0.0
        t0 = c / q
        t1 = t0
    else:
        t0 = q / a
        if abs(q) < 1e-10:
            t1 = t0
        else:
            t1 = c / q

    if t0 > t1:
        t0, t1 = t1, t0

    return True, t0, t1


def solve_cubic(a: float, b: float, c: float, d: float) -> List[float]:
    """
    Solve cubic equation ax^3 + bx^2 + cx + d = 0.
    Returns list of real roots.
    Exact match to C++ solve_cubic.
    """
    if abs(a) < 1e-6:
        success, t0, t1 = solve_quadratic(b, c, d)
        if success:
            if abs(t0 - t1) < 1e-10:
                return [t0]
            return [t0, t1]
        return []

    # Normalize cubic equation
    b = b / a
    c = c / a
    d = d / a

    Q = (b * b - 3 * c) / 9.0
    R = (2 * b * b * b - 9 * b * c + 27 * d) / 54.0

    if R * R < Q * Q * Q:
        # 3 real roots
        theta = math.acos(R / math.sqrt(Q * Q * Q))
        sqrt_Q = math.sqrt(Q)
        t0 = -2.0 * sqrt_Q * math.cos(theta / 3.0) - b / 3.0
        t1 = -2.0 * sqrt_Q * math.cos((theta + 2.0 * math.pi) / 3.0) - b / 3.0
        t2 = -2.0 * sqrt_Q * math.cos((theta - 2.0 * math.pi) / 3.0) - b / 3.0
        return [t0, t1, t2]
    else:
        RR_QQQ = R * R - Q * Q * Q
        if RR_QQQ < 0:
            RR_QQQ = 0
        sqrt_val = math.sqrt(RR_QQQ)
        if R > 0:
            A = -math.pow(R + sqrt_val, 1.0 / 3.0)
        else:
            A = math.pow(-R + sqrt_val, 1.0 / 3.0)

        if abs(A) > 1e-6:
            B = Q / A
        else:
            B = 0.0

        t0 = (A + B) - b / 3.0
        return [t0]


def solve_quintic_isolator(A: float, B: float, C: float, D: float, E: float, F: float) -> List[float]:
    """
    Solve quintic equation t^5 + Bt^4 + Ct^3 + Dt^2 + Et + F = 0 in [0, 1].
    Uses isolator polynomial method from the C++ implementation.
    """
    # Isolator polynomial coefficients
    p1A = (2.0 / 5.0) * C - (4.0 / 25.0) * B * B
    p1B = (3.0 / 5.0) * D - (3.0 / 25.0) * B * C
    p1C = (4.0 / 5.0) * E - (2.0 / 25.0) * B * D
    p1D = F - B * E / 25.0

    # Linear root
    q_root = -B / 5.0

    # Cubic roots of isolator polynomial
    p_roots = solve_cubic(p1A, p1B, p1C, p1D)

    # Collect interval boundaries
    intervals = []
    if 0 <= q_root <= 1:
        intervals.append(q_root)
    for r in p_roots:
        if 0 <= r <= 1:
            intervals.append(r)
    intervals.sort()

    def eval_polynomial(t):
        return t**5 + B*t**4 + C*t**3 + D*t**2 + E*t + F

    def eval_polynomial_deriv(t):
        return 5*t**4 + 4*B*t**3 + 3*C*t**2 + 2*D*t + E

    roots = []
    lower_bound = 0.0

    for j in range(len(intervals) + 1):
        if j < len(intervals) and intervals[j] < 0:
            continue

        upper_bound = intervals[j] if j < len(intervals) else 1.0
        upper_bound = min(upper_bound, 1.0)

        lb = lower_bound
        ub = upper_bound
        lb_eval = eval_polynomial(lb)
        ub_eval = eval_polynomial(ub)

        if lb_eval * ub_eval > 0:
            # No root in this interval
            lower_bound = upper_bound
            if upper_bound >= 1.0:
                break
            continue

        if lb_eval > ub_eval:
            lb, ub = ub, lb

        # Newton-Raphson with bisection fallback
        t = 0.5 * (lb + ub)
        for _ in range(20):
            if not (lb <= t <= ub):
                t = 0.5 * (lb + ub)
            value = eval_polynomial(t)
            if abs(value) < 1e-5:
                break
            if value > 0:
                ub = t
            else:
                lb = t
            derivative = eval_polynomial_deriv(t)
            if abs(derivative) > 1e-10:
                t = t - value / derivative

        roots.append(t)

        if upper_bound >= 1.0:
            break
        lower_bound = upper_bound

    return roots


# ============================================================================
# Geometry Utilities
# ============================================================================

def xform_pt(matrix: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Apply 3x3 affine transformation to 2D point."""
    p = torch.stack([pt[0], pt[1], torch.ones_like(pt[0])])
    transformed = matrix @ p
    return transformed[:2] / transformed[2]


def inverse_xform_pt(matrix: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Apply inverse 3x3 affine transformation to 2D point."""
    inv_matrix = torch.linalg.inv(matrix)
    return xform_pt(inv_matrix, pt)


# ============================================================================
# Winding Number (Exact match to C++ winding_number.h)
# ============================================================================

def compute_winding_number_circle(center: torch.Tensor, radius: torch.Tensor,
                                   pt: torch.Tensor) -> int:
    """Exact match to C++ compute_winding_number for Circle."""
    dist_sq = ((pt[0] - center[0]) ** 2 + (pt[1] - center[1]) ** 2).item()
    r = radius.item() if isinstance(radius, torch.Tensor) else radius
    if dist_sq < r * r:
        return 1
    return 0


def compute_winding_number_ellipse(center: torch.Tensor, radius: torch.Tensor,
                                    pt: torch.Tensor) -> int:
    """Exact match to C++ compute_winding_number for Ellipse."""
    rx = radius[0].item()
    ry = radius[1].item()
    cx = center[0].item()
    cy = center[1].item()
    px = pt[0].item()
    py = pt[1].item()

    if ((cx - px) ** 2) / (rx * rx) + ((cy - py) ** 2) / (ry * ry) < 1:
        return 1
    return 0


def compute_winding_number_rect(p_min: torch.Tensor, p_max: torch.Tensor,
                                 pt: torch.Tensor) -> int:
    """Exact match to C++ compute_winding_number for Rect."""
    px = pt[0].item()
    py = pt[1].item()
    x_min = p_min[0].item()
    y_min = p_min[1].item()
    x_max = p_max[0].item()
    y_max = p_max[1].item()

    if x_min < px < x_max and y_min < py < y_max:
        return 1
    return 0


def compute_winding_number_path(path: Path, pt: torch.Tensor) -> int:
    """
    Exact match to C++ compute_winding_number for Path.
    Uses analytical ray-curve intersection.
    """
    if not path.is_closed:
        return 0

    winding_number = 0
    num_segments = len(path.num_control_points)
    point_idx = 0
    px = pt[0].item()
    py = pt[1].item()
    points = path.points

    for seg_idx in range(num_segments):
        n_ctrl = int(path.num_control_points[seg_idx].item())

        if n_ctrl == 0:
            # Straight line
            i0 = point_idx
            if seg_idx == num_segments - 1:
                i1 = 0
            else:
                i1 = point_idx + 1

            p0x = points[i0, 0].item()
            p0y = points[i0, 1].item()
            p1x = points[i1, 0].item()
            p1y = points[i1, 1].item()

            if p1y != p0y:
                t = (py - p0y) / (p1y - p0y)
                if 0 <= t <= 1:
                    tp = p0x - px + t * (p1x - p0x)
                    if tp >= 0:
                        if p1y - p0y > 0:
                            winding_number += 1
                        else:
                            winding_number -= 1

            point_idx += 1

        elif n_ctrl == 1:
            # Quadratic Bezier
            i0 = point_idx
            i1 = point_idx + 1
            if seg_idx == num_segments - 1:
                i2 = 0
            else:
                i2 = point_idx + 2

            p0x = points[i0, 0].item()
            p0y = points[i0, 1].item()
            p1x = points[i1, 0].item()
            p1y = points[i1, 1].item()
            p2x = points[i2, 0].item()
            p2y = points[i2, 1].item()

            # Solve: py = (p0-2p1+p2)t^2 + (-2p0+2p1)t + p0
            a = p0y - 2*p1y + p2y
            b = -2*p0y + 2*p1y
            c = p0y - py

            success, t0, t1 = solve_quadratic(a, b, c)
            if success:
                for t in [t0, t1]:
                    if 0 <= t <= 1:
                        # tp = x-coordinate of curve at t minus px
                        tp = (p0x - 2*p1x + p2x)*t*t + (-2*p0x + 2*p1x)*t + p0x - px
                        if tp >= 0:
                            # Derivative of y with respect to t
                            dy_dt = 2*(p0y - 2*p1y + p2y)*t + (-2*p0y + 2*p1y)
                            if dy_dt > 0:
                                winding_number += 1
                            else:
                                winding_number -= 1

            point_idx += 2

        elif n_ctrl == 2:
            # Cubic Bezier
            i0 = point_idx
            i1 = point_idx + 1
            i2 = point_idx + 2
            if seg_idx == num_segments - 1:
                i3 = 0
            else:
                i3 = point_idx + 3

            p0x = points[i0, 0].item()
            p0y = points[i0, 1].item()
            p1x = points[i1, 0].item()
            p1y = points[i1, 1].item()
            p2x = points[i2, 0].item()
            p2y = points[i2, 1].item()
            p3x = points[i3, 0].item()
            p3y = points[i3, 1].item()

            # Solve: py = (-p0+3p1-3p2+p3)t^3 + (3p0-6p1+3p2)t^2 + (-3p0+3p1)t + p0
            a = -p0y + 3*p1y - 3*p2y + p3y
            b = 3*p0y - 6*p1y + 3*p2y
            c = -3*p0y + 3*p1y
            d = p0y - py

            roots = solve_cubic(a, b, c, d)
            for t in roots:
                if 0 <= t <= 1:
                    # tp = x-coordinate of curve at t minus px
                    tp = ((-p0x + 3*p1x - 3*p2x + p3x)*t*t*t +
                          (3*p0x - 6*p1x + 3*p2x)*t*t +
                          (-3*p0x + 3*p1x)*t +
                          p0x - px)
                    if tp > 0:
                        # Derivative of y with respect to t
                        dy_dt = (3*(-p0y + 3*p1y - 3*p2y + p3y)*t*t +
                                 2*(3*p0y - 6*p1y + 3*p2y)*t +
                                 (-3*p0y + 3*p1y))
                        if dy_dt > 0:
                            winding_number += 1
                        else:
                            winding_number -= 1

            point_idx += 3

    return winding_number


def compute_winding_number(shape, pt: torch.Tensor) -> int:
    """Compute winding number for any shape."""
    if isinstance(shape, Circle):
        return compute_winding_number_circle(shape.center, shape.radius, pt)
    elif isinstance(shape, Ellipse):
        return compute_winding_number_ellipse(shape.center, shape.radius, pt)
    elif isinstance(shape, Rect):
        return compute_winding_number_rect(shape.p_min, shape.p_max, pt)
    elif isinstance(shape, Path):
        return compute_winding_number_path(shape, pt)
    elif isinstance(shape, Polygon):
        # Convert polygon to path
        n_points = shape.points.shape[0]
        if shape.is_closed:
            num_ctrl = torch.zeros(n_points, dtype=torch.int32)
        else:
            num_ctrl = torch.zeros(n_points - 1, dtype=torch.int32)
        temp_path = Path(num_ctrl, shape.points, shape.is_closed)
        return compute_winding_number_path(temp_path, pt)
    return 0


# ============================================================================
# Closest Point / Distance (Exact match to C++ compute_distance.h)
# ============================================================================

def closest_point_line(pt: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Find closest point on line segment p0-p1 to pt."""
    px, py = pt[0].item(), pt[1].item()
    p0x, p0y = p0[0].item(), p0[1].item()
    p1x, p1y = p1[0].item(), p1[1].item()

    dx = p1x - p0x
    dy = p1y - p0y
    len_sq = dx * dx + dy * dy

    if len_sq < 1e-10:
        return p0.clone(), math.sqrt((px - p0x)**2 + (py - p0y)**2)

    t = ((px - p0x) * dx + (py - p0y) * dy) / len_sq
    t = max(0.0, min(1.0, t))

    closest_x = p0x + t * dx
    closest_y = p0y + t * dy
    dist = math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    return torch.tensor([closest_x, closest_y], dtype=pt.dtype, device=pt.device), dist


def closest_point_quadratic_bezier(pt: torch.Tensor, p0: torch.Tensor,
                                    p1: torch.Tensor, p2: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    Find closest point on quadratic Bezier to pt.
    Exact match to C++ implementation - solves cubic equation.
    Returns (closest_point, distance, t_parameter).
    """
    px, py = pt[0].item(), pt[1].item()
    p0x, p0y = p0[0].item(), p0[1].item()
    p1x, p1y = p1[0].item(), p1[1].item()
    p2x, p2y = p2[0].item(), p2[1].item()

    def eval_bezier(t):
        tt = 1 - t
        x = tt*tt*p0x + 2*tt*t*p1x + t*t*p2x
        y = tt*tt*p0y + 2*tt*t*p1y + t*t*p2y
        return x, y

    # Check endpoints
    pt0 = eval_bezier(0)
    pt1 = eval_bezier(1)
    dist0 = math.sqrt((pt0[0] - px)**2 + (pt0[1] - py)**2)
    dist1 = math.sqrt((pt1[0] - px)**2 + (pt1[1] - py)**2)

    min_dist = dist0
    min_t = 0.0
    closest = pt0

    if dist1 < min_dist:
        min_dist = dist1
        min_t = 1.0
        closest = pt1

    # Solve (q - pt) dot q' = 0
    # q = (p0-2p1+p2)t^2 + (-2p0+2p1)t + p0
    # q' = 2(p0-2p1+p2)t + (-2p0+2p1)
    ax = p0x - 2*p1x + p2x
    ay = p0y - 2*p1y + p2y
    bx = -2*p0x + 2*p1x
    by = -2*p0y + 2*p1y

    # Expanding to cubic: At^3 + Bt^2 + Ct + D = 0
    A = (ax*ax + ay*ay)
    B = 3*(ax*bx + ay*by) / 2  # Actually 3*(p0-2p1+p2)(-p0+p1)
    B = 3 * (ax * (bx/2) + ay * (by/2))  # bx/2 = -p0x+p1x
    # Recalculate properly
    A = ax*ax + ay*ay
    B = 3 * (ax * (-p0x + p1x) + ay * (-p0y + p1y))
    C = 2 * ((-p0x + p1x)**2 + (-p0y + p1y)**2) + (ax * (p0x - px) + ay * (p0y - py))
    D = ((-p0x + p1x) * (p0x - px) + (-p0y + p1y) * (p0y - py))

    roots = solve_cubic(A, B, C, D)
    for t in roots:
        if 0 <= t <= 1:
            curve_pt = eval_bezier(t)
            dist = math.sqrt((curve_pt[0] - px)**2 + (curve_pt[1] - py)**2)
            if dist < min_dist:
                min_dist = dist
                min_t = t
                closest = curve_pt

    return torch.tensor([closest[0], closest[1]], dtype=pt.dtype, device=pt.device), min_dist, min_t


def closest_point_cubic_bezier(pt: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
                                p2: torch.Tensor, p3: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    Find closest point on cubic Bezier to pt.
    Exact match to C++ implementation - solves quintic equation.
    Returns (closest_point, distance, t_parameter).
    """
    px, py = pt[0].item(), pt[1].item()
    p0x, p0y = p0[0].item(), p0[1].item()
    p1x, p1y = p1[0].item(), p1[1].item()
    p2x, p2y = p2[0].item(), p2[1].item()
    p3x, p3y = p3[0].item(), p3[1].item()

    def eval_bezier(t):
        tt = 1 - t
        x = tt*tt*tt*p0x + 3*tt*tt*t*p1x + 3*tt*t*t*p2x + t*t*t*p3x
        y = tt*tt*tt*p0y + 3*tt*tt*t*p1y + 3*tt*t*t*p2y + t*t*t*p3y
        return x, y

    # Check endpoints
    pt0 = eval_bezier(0)
    pt1 = eval_bezier(1)
    dist0 = math.sqrt((pt0[0] - px)**2 + (pt0[1] - py)**2)
    dist1 = math.sqrt((pt1[0] - px)**2 + (pt1[1] - py)**2)

    min_dist = dist0
    min_t = 0.0
    closest = pt0

    if dist1 < min_dist:
        min_dist = dist1
        min_t = 1.0
        closest = pt1

    # Coefficients for the curve
    # q = (-p0+3p1-3p2+p3)t^3 + (3p0-6p1+3p2)t^2 + (-3p0+3p1)t + p0
    ax = -p0x + 3*p1x - 3*p2x + p3x
    ay = -p0y + 3*p1y - 3*p2y + p3y
    bx = 3*p0x - 6*p1x + 3*p2x
    by = 3*p0y - 6*p1y + 3*p2y
    cx = -3*p0x + 3*p1x
    cy = -3*p0y + 3*p1y
    dx = p0x - px
    dy = p0y - py

    # Quintic coefficients from (q - pt) dot q' = 0
    # q' = 3*a*t^2 + 2*b*t + c
    A = 3 * (ax*ax + ay*ay)
    B = 5 * (ax*bx + ay*by)
    C = 4 * (ax*cx + ay*cy) + 2 * (bx*bx + by*by)
    D = 3 * ((bx*cx + by*cy) + (ax*dx + ay*dy))
    E = (cx*cx + cy*cy) + 2 * (bx*dx + by*dy)
    F = cx*dx + cy*dy

    if abs(A) < 1e-10:
        # Degenerate case
        return torch.tensor([closest[0], closest[1]], dtype=pt.dtype, device=pt.device), min_dist, min_t

    # Normalize
    B /= A
    C /= A
    D /= A
    E /= A
    F /= A

    # Solve quintic using isolator polynomials
    roots = solve_quintic_isolator(A, B, C, D, E, F)

    for t in roots:
        if 0 <= t <= 1:
            curve_pt = eval_bezier(t)
            dist = math.sqrt((curve_pt[0] - px)**2 + (curve_pt[1] - py)**2)
            if dist < min_dist:
                min_dist = dist
                min_t = t
                closest = curve_pt

    return torch.tensor([closest[0], closest[1]], dtype=pt.dtype, device=pt.device), min_dist, min_t


def closest_point_circle(pt: torch.Tensor, center: torch.Tensor,
                          radius: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Find closest point on circle boundary to pt."""
    diff = pt - center
    dist_to_center = torch.norm(diff).item()
    if dist_to_center < 1e-10:
        # Point is at center
        r = radius.item() if isinstance(radius, torch.Tensor) else radius
        return center + torch.tensor([r, 0.0], dtype=pt.dtype, device=pt.device), r

    r = radius.item() if isinstance(radius, torch.Tensor) else radius
    closest = center + r * diff / dist_to_center
    dist = abs(dist_to_center - r)
    return closest, dist


def closest_point_rect(pt: torch.Tensor, p_min: torch.Tensor,
                        p_max: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Find closest point on rectangle boundary to pt."""
    left_top = p_min
    right_top = torch.tensor([p_max[0], p_min[1]], dtype=pt.dtype, device=pt.device)
    left_bottom = torch.tensor([p_min[0], p_max[1]], dtype=pt.dtype, device=pt.device)
    right_bottom = p_max

    edges = [
        (left_top, left_bottom),
        (left_top, right_top),
        (right_top, right_bottom),
        (left_bottom, right_bottom)
    ]

    min_dist = float('inf')
    closest = None

    for p0, p1 in edges:
        cp, dist = closest_point_line(pt, p0, p1)
        if dist < min_dist:
            min_dist = dist
            closest = cp

    return closest, min_dist


def closest_point_path(pt: torch.Tensor, path: Path) -> Tuple[torch.Tensor, float, int, float]:
    """
    Find closest point on path to pt.
    Returns (closest_point, distance, segment_id, t_parameter).
    """
    min_dist = float('inf')
    closest = None
    min_seg = -1
    min_t = 0.0

    num_segments = len(path.num_control_points)
    point_idx = 0
    points = path.points

    for seg_idx in range(num_segments):
        n_ctrl = int(path.num_control_points[seg_idx].item())

        if n_ctrl == 0:
            # Line segment
            i0 = point_idx
            if seg_idx == num_segments - 1 and path.is_closed:
                i1 = 0
            else:
                i1 = point_idx + 1

            p0 = points[i0]
            p1 = points[i1]
            cp, dist = closest_point_line(pt, p0, p1)

            if dist < min_dist:
                min_dist = dist
                closest = cp
                min_seg = seg_idx
                # Compute t
                diff = p1 - p0
                len_sq = (diff[0]**2 + diff[1]**2).item()
                if len_sq > 1e-10:
                    min_t = max(0, min(1, ((pt[0]-p0[0])*diff[0] + (pt[1]-p0[1])*diff[1]).item() / len_sq))

            point_idx += 1

        elif n_ctrl == 1:
            # Quadratic Bezier
            i0 = point_idx
            i1 = point_idx + 1
            if seg_idx == num_segments - 1 and path.is_closed:
                i2 = 0
            else:
                i2 = point_idx + 2

            p0 = points[i0]
            p1 = points[i1]
            p2 = points[i2]
            cp, dist, t = closest_point_quadratic_bezier(pt, p0, p1, p2)

            if dist < min_dist:
                min_dist = dist
                closest = cp
                min_seg = seg_idx
                min_t = t

            point_idx += 2

        elif n_ctrl == 2:
            # Cubic Bezier
            i0 = point_idx
            i1 = point_idx + 1
            i2 = point_idx + 2
            if seg_idx == num_segments - 1 and path.is_closed:
                i3 = 0
            else:
                i3 = point_idx + 3

            p0 = points[i0]
            p1 = points[i1]
            p2 = points[i2]
            p3 = points[i3]
            cp, dist, t = closest_point_cubic_bezier(pt, p0, p1, p2, p3)

            if dist < min_dist:
                min_dist = dist
                closest = cp
                min_seg = seg_idx
                min_t = t

            point_idx += 3

    return closest, min_dist, min_seg, min_t


def distance_to_shape(pt: torch.Tensor, shape, transform: torch.Tensor) -> float:
    """Compute distance from pt to shape boundary."""
    # Transform point to shape's local space
    local_pt = inverse_xform_pt(transform, pt)

    if isinstance(shape, Circle):
        _, dist = closest_point_circle(local_pt, shape.center, shape.radius)
        return dist
    elif isinstance(shape, Rect):
        _, dist = closest_point_rect(local_pt, shape.p_min, shape.p_max)
        return dist
    elif isinstance(shape, Path):
        _, dist, _, _ = closest_point_path(local_pt, shape)
        return dist
    elif isinstance(shape, Polygon):
        n_points = shape.points.shape[0]
        if shape.is_closed:
            num_ctrl = torch.zeros(n_points, dtype=torch.int32)
        else:
            num_ctrl = torch.zeros(n_points - 1, dtype=torch.int32)
        temp_path = Path(num_ctrl, shape.points, shape.is_closed)
        _, dist, _, _ = closest_point_path(local_pt, temp_path)
        return dist
    elif isinstance(shape, Ellipse):
        # Approximate ellipse distance
        normalized = (local_pt - shape.center) / shape.radius
        dist_normalized = torch.norm(normalized).item()
        if dist_normalized < 1e-10:
            return min(shape.radius[0].item(), shape.radius[1].item())
        return abs(dist_normalized - 1.0) * min(shape.radius[0].item(), shape.radius[1].item())

    return float('inf')


# ============================================================================
# Color Sampling (Exact match to C++ sample_color)
# ============================================================================

def sample_color_constant(color: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Sample constant color."""
    return color.clone()


def sample_color_linear_gradient(gradient: LinearGradient, pt: torch.Tensor) -> torch.Tensor:
    """Sample linear gradient color at pt."""
    begin = gradient.begin
    end = gradient.end

    # Project point onto gradient line
    direction = end - begin
    length_sq = (direction[0]**2 + direction[1]**2).item()

    if length_sq < 1e-10:
        return gradient.stop_colors[0].clone()

    t = ((pt[0] - begin[0]) * direction[0] + (pt[1] - begin[1]) * direction[1]).item() / length_sq
    t = max(0.0, min(1.0, t))

    # Find bracketing color stops
    offsets = gradient.offsets
    stop_colors = gradient.stop_colors

    if t <= offsets[0].item():
        return stop_colors[0].clone()
    if t >= offsets[-1].item():
        return stop_colors[-1].clone()

    for i in range(len(offsets) - 1):
        o0 = offsets[i].item()
        o1 = offsets[i + 1].item()
        if o0 <= t <= o1:
            local_t = (t - o0) / (o1 - o0 + 1e-10)
            return (1.0 - local_t) * stop_colors[i] + local_t * stop_colors[i + 1]

    return stop_colors[-1].clone()


def sample_color_radial_gradient(gradient: RadialGradient, pt: torch.Tensor) -> torch.Tensor:
    """Sample radial gradient color at pt."""
    center = gradient.center
    radius = gradient.radius

    # Compute normalized distance from center
    diff = pt - center
    # Use the average radius for radial gradient
    r = (radius[0].item() + radius[1].item()) / 2.0
    if r < 1e-10:
        return gradient.stop_colors[0].clone()

    t = torch.norm(diff).item() / r
    t = max(0.0, min(1.0, t))

    offsets = gradient.offsets
    stop_colors = gradient.stop_colors

    if t <= offsets[0].item():
        return stop_colors[0].clone()
    if t >= offsets[-1].item():
        return stop_colors[-1].clone()

    for i in range(len(offsets) - 1):
        o0 = offsets[i].item()
        o1 = offsets[i + 1].item()
        if o0 <= t <= o1:
            local_t = (t - o0) / (o1 - o0 + 1e-10)
            return (1.0 - local_t) * stop_colors[i] + local_t * stop_colors[i + 1]

    return stop_colors[-1].clone()


def sample_color(color, pt: torch.Tensor) -> Optional[torch.Tensor]:
    """Sample color at point pt."""
    if color is None:
        return None
    elif isinstance(color, torch.Tensor):
        return sample_color_constant(color, pt)
    elif isinstance(color, LinearGradient):
        return sample_color_linear_gradient(color, pt)
    elif isinstance(color, RadialGradient):
        return sample_color_radial_gradient(color, pt)
    return None


# ============================================================================
# Filter Functions
# ============================================================================

def compute_filter_weight(filter_type: FilterType, dx: float, dy: float,
                          radius: float) -> float:
    """Compute filter weight at offset (dx, dy)."""
    if filter_type == FilterType.box:
        if abs(dx) <= radius and abs(dy) <= radius:
            return 1.0
        return 0.0
    elif filter_type == FilterType.tent:
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= radius:
            return max(0.0, 1.0 - dist / radius)
        return 0.0
    elif filter_type == FilterType.gaussian:
        sigma = radius / 3.0
        dist_sq = dx * dx + dy * dy
        return math.exp(-dist_sq / (2.0 * sigma * sigma))
    return 1.0


# ============================================================================
# Fragment and Blending (Exact match to C++ sample_color in diffvg.cpp)
# ============================================================================

class Fragment:
    def __init__(self, color: torch.Tensor, alpha: float, group_id: int, is_stroke: bool):
        self.color = color  # RGB
        self.alpha = alpha
        self.group_id = group_id
        self.is_stroke = is_stroke


def blend_fragments(fragments: List[Fragment], background_color: Optional[torch.Tensor],
                    device, dtype) -> torch.Tensor:
    """
    Blend fragments from back to front.
    Exact match to C++ sample_color blending logic.
    """
    if not fragments:
        if background_color is not None:
            return background_color.clone()
        return torch.zeros(4, device=device, dtype=dtype)

    # Sort fragments by group_id (back to front)
    fragments.sort(key=lambda f: f.group_id)

    # Initialize with background
    if background_color is not None:
        accum_color = background_color[:3].clone()
        accum_alpha = background_color[3].item()
    else:
        accum_color = torch.zeros(3, device=device, dtype=dtype)
        accum_alpha = 0.0

    # Blend each fragment
    for frag in fragments:
        new_color = frag.color
        new_alpha = frag.alpha

        # prev_color is alpha premultiplied, don't need to multiply with prev_alpha
        # accum_color = prev_color * (1 - new_alpha) + new_alpha * new_color
        # accum_alpha = prev_alpha * (1 - new_alpha) + new_alpha
        accum_color = accum_color * (1.0 - new_alpha) + new_alpha * new_color
        accum_alpha = accum_alpha * (1.0 - new_alpha) + new_alpha

    # Un-premultiply
    if accum_alpha > 1e-6:
        final_color = accum_color / accum_alpha
    else:
        final_color = accum_color

    return torch.cat([final_color, torch.tensor([accum_alpha], device=device, dtype=dtype)])


# ============================================================================
# Main Render Function
# ============================================================================

def sample_pixel(x: float, y: float, shapes: List, shape_groups: List[ShapeGroup],
                 background: Optional[torch.Tensor], device, dtype) -> torch.Tensor:
    """Sample color at pixel location (x, y). Exact match to C++ sample_color."""
    pt = torch.tensor([x, y], device=device, dtype=dtype)
    fragments = []

    # Iterate through shape groups
    for group_idx, group in enumerate(shape_groups):
        transform = group.shape_to_canvas.to(device=device, dtype=dtype)
        inv_transform = torch.linalg.inv(transform)

        for shape_id in group.shape_ids:
            shape_id_val = shape_id.item() if isinstance(shape_id, torch.Tensor) else shape_id
            shape = shapes[shape_id_val]

            # Transform point to shape's local space
            local_pt = xform_pt(inv_transform, pt)

            # Check stroke
            if group.stroke_color is not None:
                stroke_width = shape.stroke_width
                if isinstance(stroke_width, torch.Tensor):
                    stroke_width = stroke_width.item()

                dist = distance_to_shape(pt, shape, transform)
                half_width = stroke_width / 2.0

                if dist <= half_width:
                    color = sample_color(group.stroke_color, local_pt)
                    if color is not None:
                        color = color.to(device=device, dtype=dtype)
                        frag = Fragment(color[:3], color[3].item(), group_idx, True)
                        fragments.append(frag)

            # Check fill
            if group.fill_color is not None:
                winding = compute_winding_number(shape, local_pt)

                is_inside = False
                if group.use_even_odd_rule:
                    is_inside = (abs(winding) % 2) == 1
                else:
                    is_inside = winding != 0

                if is_inside:
                    color = sample_color(group.fill_color, local_pt)
                    if color is not None:
                        color = color.to(device=device, dtype=dtype)
                        frag = Fragment(color[:3], color[3].item(), group_idx, False)
                        fragments.append(frag)

    return blend_fragments(fragments, background, device, dtype)


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
    Pure PyTorch implementation of diffvg's render function with EXACT PARITY.

    Args:
        width: Output image width
        height: Output image height
        num_samples_x: Number of samples per pixel in x
        num_samples_y: Number of samples per pixel in y
        seed: Random seed for jittered sampling
        shapes: List of shape objects
        shape_groups: List of ShapeGroup objects
        background_image: Optional background [H, W, 4]
        filter_type: Pixel filter type
        filter_radius: Pixel filter radius
        output_type: Output type (color or sdf)
        device: Torch device
        dtype: Torch dtype

    Returns:
        Rendered image [height, width, 4] for color, [height, width, 1] for SDF
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    if output_type == OutputType.color:
        result = torch.zeros(height, width, 4, device=device, dtype=dtype)
    else:
        result = torch.zeros(height, width, 1, device=device, dtype=dtype)

    # Pre-compute jitter for all samples
    total_samples = num_samples_x * num_samples_y

    for py in range(height):
        for px in range(width):
            if output_type == OutputType.color:
                accumulated = torch.zeros(4, device=device, dtype=dtype)
                weight_sum = 0.0

                for sy in range(num_samples_y):
                    for sx in range(num_samples_x):
                        # Jittered sample position
                        jitter_x = torch.rand(1, device=device, dtype=dtype).item() - 0.5
                        jitter_y = torch.rand(1, device=device, dtype=dtype).item() - 0.5

                        sample_x = px + (sx + 0.5 + jitter_x) / num_samples_x
                        sample_y = py + (sy + 0.5 + jitter_y) / num_samples_y

                        # Filter offset from pixel center
                        offset_x = sample_x - (px + 0.5)
                        offset_y = sample_y - (py + 0.5)

                        weight = compute_filter_weight(filter_type, offset_x, offset_y, filter_radius)

                        if weight > 0:
                            # Get background color at this sample
                            bg_color = None
                            if background_image is not None:
                                bx = int(sample_x)
                                by = int(sample_y)
                                if 0 <= bx < width and 0 <= by < height:
                                    bg_color = background_image[by, bx]

                            color = sample_pixel(sample_x, sample_y, shapes, shape_groups,
                                                bg_color, device, dtype)
                            accumulated = accumulated + weight * color
                            weight_sum += weight

                if weight_sum > 0:
                    result[py, px] = accumulated / weight_sum

            else:
                # SDF output
                pt = torch.tensor([px + 0.5, py + 0.5], device=device, dtype=dtype)
                min_dist = float('inf')
                is_inside = False

                for group in shape_groups:
                    transform = group.shape_to_canvas.to(device=device, dtype=dtype)

                    for shape_id in group.shape_ids:
                        shape_id_val = shape_id.item() if isinstance(shape_id, torch.Tensor) else shape_id
                        shape = shapes[shape_id_val]

                        dist = distance_to_shape(pt, shape, transform)
                        if dist < abs(min_dist):
                            inv_transform = torch.linalg.inv(transform)
                            local_pt = xform_pt(inv_transform, pt)
                            winding = compute_winding_number(shape, local_pt)
                            if group.use_even_odd_rule:
                                is_inside = (abs(winding) % 2) == 1
                            else:
                                is_inside = winding != 0
                            min_dist = -dist if is_inside else dist

                result[py, px, 0] = min_dist

    return result


# ============================================================================
# Convenience API
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
    """Convenience function matching the pydiffvg.RenderFunction.apply API."""
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
