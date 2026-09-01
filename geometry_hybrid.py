"""Geometry primitives for atrial CS/VoM tube construction.

The functions in this module are anatomy-agnostic.  They provide robust ring,
vector and centreline operations used by the higher-level tube workflow.

A central design rule is that a ring normal must not be trusted without an
orientation reference: a best-fit plane has two equally valid normals, ``n``
and ``-n``.  Functions therefore accept a ``reference_direction`` whenever
anatomical orientation matters.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

_EPS = 1.0e-12


def normalize_vector(vector: Sequence[float], *, name: str = "vector") -> np.ndarray:
    """Return a unit vector, raising a useful error for a zero-length input."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")

    norm = np.linalg.norm(vector)
    if norm < _EPS:
        raise ValueError(f"{name} has zero or near-zero length")
    return vector / norm


def ring_center(ring: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the centroid of a ring."""
    ring = _as_ring(ring)
    return np.mean(ring, axis=0)


def ring_radius(ring: Sequence[Sequence[float]]) -> float:
    """Return the mean radial distance of ring points from their centroid."""
    ring = _as_ring(ring)
    center = np.mean(ring, axis=0)
    return float(np.mean(np.linalg.norm(ring - center, axis=1)))


def compute_ring_normal(
    ring: Sequence[Sequence[float]],
    reference_direction: Sequence[float] | None = None,
) -> np.ndarray:
    """Fit a plane to a ring and return a unit normal.

    SVD determines the plane but not the sign of its normal.  When
    ``reference_direction`` is supplied, the returned normal is guaranteed to
    have a non-negative dot product with that direction.

    If no reference is supplied, a deterministic sign convention is used only
    to make repeated runs reproducible.  Anatomical code should normally pass
    a reference direction.
    """
    ring = _as_ring(ring)
    centered = ring - np.mean(ring, axis=0)

    if np.linalg.matrix_rank(centered, tol=_EPS) < 2:
        raise ValueError("Cannot compute a ring normal from collinear points")

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize_vector(vh[-1], name="ring normal")

    if reference_direction is not None:
        reference = normalize_vector(reference_direction, name="reference_direction")
        if np.dot(normal, reference) < 0.0:
            normal = -normal
    else:
        # SVD may return either sign.  This gives a reproducible fallback, but
        # it is deliberately not presented as an anatomical orientation rule.
        dominant_component = int(np.argmax(np.abs(normal)))
        if normal[dominant_component] < 0.0:
            normal = -normal

    return normal


def ordered_ring_normal(ring: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the normal implied by the winding of an already ordered ring."""
    ring = _as_ring(ring)
    center = np.mean(ring, axis=0)
    centered = ring - center
    next_centered = np.roll(centered, -1, axis=0)
    normal = np.sum(np.cross(centered, next_centered), axis=0)

    if np.linalg.norm(normal) < _EPS:
        raise ValueError("Ordered ring has no well-defined polygon normal")
    return normalize_vector(normal, name="ordered ring normal")


def order_ring_points(
    ring: Sequence[Sequence[float]],
    *,
    reference_direction: Sequence[float] | None = None,
    start_reference: Sequence[float] | None = None,
) -> np.ndarray:
    """Order an unordered planar ring by angle around its centroid.

    Parameters
    ----------
    ring
        Unordered ``(N, 3)`` ring points.
    reference_direction
        Desired sign of the ring normal.  Strongly recommended for anatomical
        use.  If supplied, the final winding follows this direction.
    start_reference
        Optional 3-D point used to choose which ring point becomes index 0.
        This makes later hinge/landmark indexing reproducible.
    """
    ring = _as_ring(ring)
    center = np.mean(ring, axis=0)
    centered = ring - center
    normal = compute_ring_normal(ring, reference_direction)

    if start_reference is not None:
        in_plane_reference = np.asarray(start_reference, dtype=float) - center
        in_plane_reference -= normal * np.dot(in_plane_reference, normal)
    else:
        # Use the farthest ring point as a stable in-plane direction.
        in_plane_reference = centered[np.argmax(np.linalg.norm(centered, axis=1))].copy()
        in_plane_reference -= normal * np.dot(in_plane_reference, normal)

    if np.linalg.norm(in_plane_reference) < _EPS:
        # Rare fallback: choose the first point with a usable in-plane vector.
        for candidate in centered:
            candidate = candidate - normal * np.dot(candidate, normal)
            if np.linalg.norm(candidate) >= _EPS:
                in_plane_reference = candidate
                break
        else:
            raise ValueError("Could not establish an in-plane ring reference")

    axis_u = normalize_vector(in_plane_reference, name="ring in-plane reference")
    axis_v = normalize_vector(np.cross(normal, axis_u), name="ring secondary axis")

    angles = np.arctan2(centered @ axis_v, centered @ axis_u)
    ordered = ring[np.argsort(angles)]

    # Numerical details or a fallback normal should never silently invert the
    # requested winding.
    if reference_direction is not None:
        reference = normalize_vector(reference_direction, name="reference_direction")
        if np.dot(ordered_ring_normal(ordered), reference) < 0.0:
            ordered = ordered[::-1]

    if start_reference is not None:
        target = np.asarray(start_reference, dtype=float)
        start_index = int(np.argmin(np.linalg.norm(ordered - target, axis=1)))
        ordered = np.roll(ordered, -start_index, axis=0)

    return ordered


def resample_closed_ring(
    ring: Sequence[Sequence[float]],
    *,
    spacing: float | None = None,
    n_points: int | None = None,
) -> np.ndarray:
    """Uniformly resample a closed ring by arc length.

    Exactly one of ``spacing`` or ``n_points`` must be supplied.  Winding and
    the location of the first point are preserved.
    """
    ring = _as_ring(ring)

    if (spacing is None) == (n_points is None):
        raise ValueError("Provide exactly one of spacing or n_points")

    # Remove an explicitly duplicated closing point if present.
    if len(ring) > 3 and np.linalg.norm(ring[0] - ring[-1]) < _EPS:
        ring = ring[:-1]

    closed = np.vstack([ring, ring[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(segment_lengths < _EPS):
        raise ValueError("Ring contains duplicate consecutive points")

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = float(cumulative[-1])

    if n_points is None:
        if spacing is None or spacing <= 0.0:
            raise ValueError("spacing must be positive")
        n_points = max(3, int(np.ceil(perimeter / float(spacing))))
    elif n_points < 3:
        raise ValueError("n_points must be at least 3")

    sample_distances = np.linspace(0.0, perimeter, int(n_points), endpoint=False)
    return _interpolate_polyline(closed, cumulative, sample_distances)


def resample_open_path(
    path: Sequence[Sequence[float]],
    spacing: float,
    *,
    include_endpoint: bool = True,
) -> np.ndarray:
    """Uniformly resample an open 3-D path by arc length."""
    path = _as_path(path)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(segment_lengths < _EPS):
        raise ValueError("Path contains duplicate consecutive points")

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])

    sample_distances = np.arange(0.0, total_length, float(spacing))
    if include_endpoint and (
        len(sample_distances) == 0 or not np.isclose(sample_distances[-1], total_length)
    ):
        sample_distances = np.append(sample_distances, total_length)

    return _interpolate_polyline(path, cumulative, sample_distances)


def smooth_path(path: Sequence[Sequence[float]], sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth an open path while preserving its two endpoints."""
    path = _as_path(path)
    if sigma <= 0.0:
        return path.copy()

    smoothed = gaussian_filter1d(path, sigma=sigma, axis=0, mode="nearest")
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed


def path_tangents(path: Sequence[Sequence[float]]) -> np.ndarray:
    """Return unit tangents using forward/backward endpoints and central differences."""
    path = _as_path(path)
    tangents = np.empty_like(path)
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    if len(path) > 2:
        tangents[1:-1] = path[2:] - path[:-2]

    norms = np.linalg.norm(tangents, axis=1)
    if np.any(norms < _EPS):
        raise ValueError("Path contains a location with an undefined tangent")
    return tangents / norms[:, None]


def rotation_matrix_from_vectors(
    source: Sequence[float],
    target: Sequence[float],
) -> np.ndarray:
    """Return the minimal 3-D rotation mapping ``source`` onto ``target``.

    Unlike the old implementation, this explicitly handles antiparallel
    vectors, for which the cross product is zero but a 180-degree rotation is
    required.
    """
    source = normalize_vector(source, name="source")
    target = normalize_vector(target, name="target")
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 1.0 - 1.0e-12:
        return np.eye(3)

    if dot < -1.0 + 1.0e-12:
        # Choose the Cartesian axis least parallel to source, then construct a
        # guaranteed perpendicular rotation axis.
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = normalize_vector(np.cross(source, basis), name="180-degree rotation axis")
        return Rotation.from_rotvec(np.pi * axis).as_matrix()

    axis = normalize_vector(np.cross(source, target), name="rotation axis")
    angle = float(np.arccos(dot))
    return Rotation.from_rotvec(angle * axis).as_matrix()


def align_ring_normal(
    ring: Sequence[Sequence[float]],
    target_direction: Sequence[float],
    *,
    current_normal: Sequence[float] | None = None,
) -> np.ndarray:
    """Rotate a ring about its centroid so its normal matches ``target_direction``."""
    ring = _as_ring(ring)
    target = normalize_vector(target_direction, name="target_direction")

    if current_normal is None:
        try:
            current = ordered_ring_normal(ring)
        except ValueError:
            current = compute_ring_normal(ring, reference_direction=target)
    else:
        current = normalize_vector(current_normal, name="current_normal")

    rotation = rotation_matrix_from_vectors(current, target)
    center = np.mean(ring, axis=0)
    return (ring - center) @ rotation.T + center


def parallel_transport_frames(
    path: Sequence[Sequence[float]],
    *,
    initial_radial: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct rotation-minimizing frames along a centreline.

    Returns ``(tangents, radial, binormal)``.  ``radial`` and ``binormal`` span
    the cross-section plane.  The radial vector is propagated from one path
    point to the next with the minimal rotation that aligns consecutive
    tangents, avoiding the arbitrary global-up-vector twist of the old code.
    """
    path = _as_path(path)
    tangents = path_tangents(path)

    if initial_radial is None:
        radial0 = _stable_perpendicular(tangents[0])
    else:
        radial0 = np.asarray(initial_radial, dtype=float)
        radial0 = radial0 - tangents[0] * np.dot(radial0, tangents[0])
        if np.linalg.norm(radial0) < _EPS:
            raise ValueError("initial_radial is parallel to the first path tangent")
        radial0 = normalize_vector(radial0, name="initial_radial")

    radial = np.empty_like(path)
    binormal = np.empty_like(path)
    radial[0] = radial0
    binormal[0] = normalize_vector(
        np.cross(tangents[0], radial[0]), name="initial binormal"
    )
    radial[0] = normalize_vector(
        np.cross(binormal[0], tangents[0]), name="initial radial"
    )

    for index in range(1, len(path)):
        rotation = rotation_matrix_from_vectors(tangents[index - 1], tangents[index])
        candidate = rotation @ radial[index - 1]

        # Remove numerical drift from the tangent direction.
        candidate -= tangents[index] * np.dot(candidate, tangents[index])
        if np.linalg.norm(candidate) < _EPS:
            candidate = _stable_perpendicular(tangents[index])
        radial[index] = normalize_vector(candidate, name="transported radial")
        binormal[index] = normalize_vector(
            np.cross(tangents[index], radial[index]), name="transported binormal"
        )
        radial[index] = normalize_vector(
            np.cross(binormal[index], tangents[index]), name="transported radial"
        )

    return tangents, radial, binormal


def match_ring_indexing(
    reference_ring: Sequence[Sequence[float]],
    target_ring: Sequence[Sequence[float]],
    *,
    allow_reverse: bool = False,
) -> np.ndarray:
    """Cyclically align target-ring point indices with a reference ring.

    Matching is performed after centring and normalising each ring so that
    translation and radius differences do not dominate the correspondence.
    By default winding is preserved; reversal must be requested explicitly.
    """
    reference = _as_ring(reference_ring)
    target = _as_ring(target_ring)
    if reference.shape != target.shape:
        raise ValueError("reference_ring and target_ring must have the same shape")

    reference_shape = _normalised_ring_shape(reference)
    candidates = [(target, False)]
    if allow_reverse:
        candidates.append((target[::-1], True))

    best_ring: np.ndarray | None = None
    best_error = np.inf

    for candidate, _ in candidates:
        candidate_shape = _normalised_ring_shape(candidate)
        for shift in range(len(candidate)):
            shifted_shape = np.roll(candidate_shape, -shift, axis=0)
            error = float(np.mean(np.sum((reference_shape - shifted_shape) ** 2, axis=1)))
            if error < best_error:
                best_error = error
                best_ring = np.roll(candidate, -shift, axis=0)

    if best_ring is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to match ring indexing")
    return best_ring


def stitch_rings(rings: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """Create VTK-style triangular faces between consecutive equal-sized rings.

    The returned array is flat and has the same ``[3, i0, i1, i2, ...]`` form
    expected by the existing CARP surface writer.
    """
    rings_array = np.asarray(rings, dtype=float)
    if rings_array.ndim != 3 or rings_array.shape[2] != 3:
        raise ValueError("rings must have shape (N_rings, N_points, 3)")
    if rings_array.shape[0] < 2:
        raise ValueError("At least two rings are required for stitching")
    if rings_array.shape[1] < 3:
        raise ValueError("Each ring must contain at least three points")

    n_rings, n_points, _ = rings_array.shape
    faces: list[list[int]] = []

    for ring_index in range(n_rings - 1):
        current_offset = ring_index * n_points
        next_offset = (ring_index + 1) * n_points
        for point_index in range(n_points):
            next_point = (point_index + 1) % n_points
            a = current_offset + point_index
            b = current_offset + next_point
            c = next_offset + next_point
            d = next_offset + point_index
            faces.append([3, a, b, c])
            faces.append([3, a, c, d])

    return np.asarray(faces, dtype=np.int32).reshape(-1)


def closest_point_index(points: Sequence[Sequence[float]], target: Sequence[float]) -> tuple[int, float]:
    """Return the index and Euclidean distance of the point nearest ``target``."""
    points = np.asarray(points, dtype=float)
    target = np.asarray(target, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if target.shape != (3,):
        raise ValueError("target must have shape (3,)")

    distances = np.linalg.norm(points - target, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _stable_perpendicular(vector: Sequence[float]) -> np.ndarray:
    vector = normalize_vector(vector)
    basis = np.eye(3)[int(np.argmin(np.abs(vector)))]
    return normalize_vector(np.cross(vector, basis), name="perpendicular vector")


def _normalised_ring_shape(ring: np.ndarray) -> np.ndarray:
    centered = ring - np.mean(ring, axis=0)
    scale = np.mean(np.linalg.norm(centered, axis=1))
    if scale < _EPS:
        raise ValueError("Ring radius is zero")
    return centered / scale


def _interpolate_polyline(
    points: np.ndarray,
    cumulative_distance: np.ndarray,
    sample_distance: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(sample_distance), 3), dtype=float)
    for coordinate in range(3):
        result[:, coordinate] = np.interp(
            sample_distance,
            cumulative_distance,
            points[:, coordinate],
        )
    return result


def _as_ring(ring: Sequence[Sequence[float]]) -> np.ndarray:
    ring = np.asarray(ring, dtype=float)
    if ring.ndim != 2 or ring.shape[1] != 3:
        raise ValueError(f"ring must have shape (N, 3), got {ring.shape}")
    if len(ring) < 3:
        raise ValueError("ring must contain at least three points")
    if not np.all(np.isfinite(ring)):
        raise ValueError("ring contains NaN or infinite values")
    return ring


def _as_path(path: Sequence[Sequence[float]]) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"path must have shape (N, 3), got {path.shape}")
    if len(path) < 2:
        raise ValueError("path must contain at least two points")
    if not np.all(np.isfinite(path)):
        raise ValueError("path contains NaN or infinite values")
    return path

# -----------------------------------------------------------------------------
# Surface construction primitives
# -----------------------------------------------------------------------------


def make_circular_ring(
    center: Sequence[float],
    tangent: Sequence[float],
    radius: float,
    n_points: int,
    *,
    radial_reference: Sequence[float] | None = None,
) -> np.ndarray:
    """Create an ordered circular ring perpendicular to ``tangent``.

    Point ordering follows the right-hand rule around ``tangent``.  When a
    radial reference is supplied it determines point 0 after projection onto
    the cross-section plane.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if int(n_points) < 3:
        raise ValueError("n_points must be at least 3")

    tangent = normalize_vector(tangent, name="tangent")
    if radial_reference is None:
        radial = _stable_perpendicular(tangent)
    else:
        radial = np.asarray(radial_reference, dtype=float)
        radial -= tangent * np.dot(radial, tangent)
        if np.linalg.norm(radial) < _EPS:
            radial = _stable_perpendicular(tangent)
        else:
            radial = normalize_vector(radial, name="radial_reference")
    binormal = normalize_vector(np.cross(tangent, radial), name="ring binormal")
    radial = normalize_vector(np.cross(binormal, tangent), name="ring radial")

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)
    offsets = (
        np.cos(angles)[:, None] * radial[None, :]
        + np.sin(angles)[:, None] * binormal[None, :]
    )
    return center[None, :] + float(radius) * offsets


def radius_profile(
    n_points: int,
    start_radius: float,
    end_radius: float,
    *,
    taper_start_fraction: float = 0.5,
) -> np.ndarray:
    """Return a radius profile that stays constant then tapers linearly."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if start_radius <= 0.0 or end_radius <= 0.0:
        raise ValueError("radii must be positive")
    if not 0.0 <= taper_start_fraction < 1.0:
        raise ValueError("taper_start_fraction must satisfy 0 <= value < 1")

    t = np.linspace(0.0, 1.0, int(n_points))
    radii = np.full(int(n_points), float(start_radius), dtype=float)
    mask = t > taper_start_fraction
    if np.any(mask):
        local = (t[mask] - taper_start_fraction) / (1.0 - taper_start_fraction)
        radii[mask] = start_radius + local * (end_radius - start_radius)
    radii[-1] = float(end_radius)
    return radii


def build_tube_along_path(
    path: Sequence[Sequence[float]],
    *,
    start_radius: float,
    end_radius: float,
    n_ring_points: int,
    taper_start_fraction: float = 0.5,
    initial_radial: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a circular tube using rotation-minimizing frames.

    Returns ``(points, faces, first_ring, last_ring, tangents)``.
    No duplicate first ring is created.
    """
    path = _as_path(path)
    tangents, radial, binormal = parallel_transport_frames(
        path, initial_radial=initial_radial
    )
    radii = radius_profile(
        len(path),
        start_radius,
        end_radius,
        taper_start_fraction=taper_start_fraction,
    )

    angles = np.linspace(0.0, 2.0 * np.pi, int(n_ring_points), endpoint=False)
    ca = np.cos(angles)
    sa = np.sin(angles)
    rings = []
    for center, radius, r, b in zip(path, radii, radial, binormal):
        ring = center[None, :] + float(radius) * (
            ca[:, None] * r[None, :] + sa[:, None] * b[None, :]
        )
        rings.append(ring)

    rings_array = np.asarray(rings, dtype=float)
    points = rings_array.reshape(-1, 3)
    faces = stitch_rings(rings_array)
    return points, faces, rings_array[0].copy(), rings_array[-1].copy(), tangents


def orient_path_start_near(
    path: Sequence[Sequence[float]], reference_point: Sequence[float]
) -> np.ndarray:
    """Orient an open path so its first endpoint is nearest ``reference_point``."""
    path = _as_path(path).copy()
    reference = np.asarray(reference_point, dtype=float)
    if reference.shape != (3,):
        raise ValueError("reference_point must have shape (3,)")
    if np.linalg.norm(path[-1] - reference) < np.linalg.norm(path[0] - reference):
        path = path[::-1].copy()
    return path


def hermite_connector_path(
    start: Sequence[float],
    end: Sequence[float],
    start_tangent: Sequence[float],
    end_tangent: Sequence[float],
    *,
    n_points: int = 12,
    handle_scale: float = 0.35,
) -> np.ndarray:
    """Create a cubic Hermite connector between two centres.

    ``handle_scale`` multiplies the straight-line endpoint distance and controls
    how strongly the supplied endpoint tangents influence the bend.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start and end must each have shape (3,)")
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    if handle_scale <= 0.0:
        raise ValueError("handle_scale must be positive")

    chord = end - start
    distance = float(np.linalg.norm(chord))
    if distance < _EPS:
        raise ValueError("Elbow endpoints are coincident")

    m0 = normalize_vector(start_tangent, name="start_tangent") * distance * handle_scale
    m1 = normalize_vector(end_tangent, name="end_tangent") * distance * handle_scale

    t = np.linspace(0.0, 1.0, int(n_points))
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00[:, None] * start[None, :]
        + h10[:, None] * m0[None, :]
        + h01[:, None] * end[None, :]
        + h11[:, None] * m1[None, :]
    )


def _signed_angle_about_axis(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    source = source - axis * np.dot(source, axis)
    target = target - axis * np.dot(target, axis)
    source = normalize_vector(source, name="angle source")
    target = normalize_vector(target, name="angle target")
    x = float(np.clip(np.dot(source, target), -1.0, 1.0))
    y = float(np.dot(axis, np.cross(source, target)))
    return float(np.arctan2(y, x))


def build_elbow_between_rings(
    start_ring: Sequence[Sequence[float]],
    end_ring: Sequence[Sequence[float]],
    start_tangent: Sequence[float],
    end_tangent: Sequence[float],
    *,
    n_centerline_points: int = 12,
    handle_scale: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth connector whose endpoint rings are exact.

    The centreline is cubic Hermite.  Intermediate cross-sections use a
    rotation-minimizing frame, with a gradual axial twist so the final ring
    indexing agrees with ``end_ring``.  The exact start and end rings replace
    the generated endpoint circles, preventing gaps at component interfaces.

    Returns ``(points, faces, centreline)``.
    """
    start_ring = _as_ring(start_ring).copy()
    end_ring = _as_ring(end_ring).copy()
    if start_ring.shape != end_ring.shape:
        raise ValueError("start_ring and end_ring must contain the same number of points")

    start_center = ring_center(start_ring)
    end_center = ring_center(end_ring)
    start_tangent = normalize_vector(start_tangent, name="start_tangent")
    end_tangent = normalize_vector(end_tangent, name="end_tangent")

    path = hermite_connector_path(
        start_center,
        end_center,
        start_tangent,
        end_tangent,
        n_points=n_centerline_points,
        handle_scale=handle_scale,
    )

    # Establish point 0 of the transported circles from start-ring point 0.
    start_radial = start_ring[0] - start_center
    start_radial -= start_tangent * np.dot(start_radial, start_tangent)
    if np.linalg.norm(start_radial) < _EPS:
        start_radial = _stable_perpendicular(start_tangent)

    tangents, radial, binormal = parallel_transport_frames(
        path, initial_radial=start_radial
    )

    # Gradually twist the transported frame so its final phase agrees with the
    # target ring.  This prevents a concentrated seam twist at the final band.
    target_radial = end_ring[0] - end_center
    target_radial -= end_tangent * np.dot(target_radial, end_tangent)
    if np.linalg.norm(target_radial) < _EPS:
        target_radial = radial[-1]
    twist = _signed_angle_about_axis(radial[-1], target_radial, end_tangent)

    fractions = np.linspace(0.0, 1.0, len(path))
    adjusted_radial = np.empty_like(radial)
    adjusted_binormal = np.empty_like(binormal)
    for i, (frac, tangent, r, b) in enumerate(zip(fractions, tangents, radial, binormal)):
        angle = twist * frac
        rr = np.cos(angle) * r + np.sin(angle) * b
        bb = np.cross(tangent, rr)
        adjusted_radial[i] = normalize_vector(rr, name="elbow radial")
        adjusted_binormal[i] = normalize_vector(bb, name="elbow binormal")

    start_radius = ring_radius(start_ring)
    end_radius = ring_radius(end_ring)
    radii = np.linspace(start_radius, end_radius, len(path))
    n_ring_points = len(start_ring)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring_points, endpoint=False)
    ca = np.cos(angles)
    sa = np.sin(angles)

    rings = []
    for center, radius, r, b in zip(path, radii, adjusted_radial, adjusted_binormal):
        rings.append(
            center[None, :] + float(radius) * (
                ca[:, None] * r[None, :] + sa[:, None] * b[None, :]
            )
        )

    rings[0] = start_ring.copy()
    # Preserve winding but cyclically align point indices to the penultimate
    # generated ring before inserting the exact endpoint ring.
    rings[-1] = match_ring_indexing(rings[-2], end_ring, allow_reverse=False)

    rings_array = np.asarray(rings, dtype=float)
    points = rings_array.reshape(-1, 3)
    faces = stitch_rings(rings_array)
    return points, faces, path


def merge_surface_components(
    components: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    weld_tolerance: float = 1.0e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge triangular surface components and weld coincident interface points.

    A very small tolerance should be used here: this function is intended to
    identify rings deliberately copied between adjacent components, not to
    merge anatomically nearby but distinct surfaces.
    """
    if weld_tolerance <= 0.0:
        raise ValueError("weld_tolerance must be positive")
    if not components:
        raise ValueError("At least one component is required")

    unique_points: list[np.ndarray] = []
    buckets: dict[tuple[int, int, int], list[int]] = {}
    triangles_out: list[list[int]] = []

    for points, faces in components:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("component points must have shape (N, 3)")
        flat = np.asarray(faces, dtype=int)
        if flat.ndim != 1 or len(flat) % 4 != 0:
            raise ValueError("component faces must be flat VTK triangles")
        vtk = flat.reshape(-1, 4)
        if not np.all(vtk[:, 0] == 3):
            raise ValueError("component faces contain non-triangle entries")

        remap = np.empty(len(points), dtype=int)
        for local_idx, point in enumerate(points):
            key = tuple(np.rint(point / weld_tolerance).astype(np.int64))
            chosen = None
            for existing_idx in buckets.get(key, []):
                if np.linalg.norm(unique_points[existing_idx] - point) <= weld_tolerance:
                    chosen = existing_idx
                    break
            if chosen is None:
                chosen = len(unique_points)
                unique_points.append(point.copy())
                buckets.setdefault(key, []).append(chosen)
            remap[local_idx] = chosen

        for row in vtk:
            i0, i1, i2 = remap[row[1:4]]
            if len({int(i0), int(i1), int(i2)}) < 3:
                continue
            triangles_out.append([3, int(i0), int(i1), int(i2)])

    return np.vstack(unique_points), np.asarray(triangles_out, dtype=np.int32).reshape(-1)


def surface_quality(points: np.ndarray, faces: np.ndarray) -> dict[str, float | int]:
    """Return basic triangle and edge diagnostics for an open surface."""
    points = np.asarray(points, dtype=float)
    vtk = np.asarray(faces, dtype=int).reshape(-1, 4)
    triangles = vtk[:, 1:4]
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    return {
        "n_points": int(len(points)),
        "n_triangles": int(len(triangles)),
        "min_triangle_area": float(np.min(areas)) if len(areas) else 0.0,
        "mean_triangle_area": float(np.mean(areas)) if len(areas) else 0.0,
        "zero_area_triangles": int(np.sum(areas <= 1.0e-12)),
        "boundary_edges": int(boundary_edges),
        "nonmanifold_edges": int(nonmanifold_edges),
    }


def orient_ring_winding(
    ring: Sequence[Sequence[float]], reference_direction: Sequence[float]
) -> np.ndarray:
    """Return a ring whose winding normal follows ``reference_direction``.

    Point 0 is preserved when reversal is required; only traversal direction is
    changed.  This is useful at tube/elbow interfaces where face connectivity
    requires a consistent cyclic orientation.
    """
    ring = _as_ring(ring).copy()
    reference = normalize_vector(reference_direction, name="reference_direction")
    if np.dot(ordered_ring_normal(ring), reference) < 0.0:
        ring = np.concatenate([ring[:1], ring[:0:-1]], axis=0)
    return ring

# -----------------------------------------------------------------------------
# LA-surface offset / clearance primitives
# -----------------------------------------------------------------------------


def local_surface_normals_from_point_cloud(
    query_points: Sequence[Sequence[float]],
    surface_points: Sequence[Sequence[float]],
    *,
    k_neighbors: int = 30,
    reference_center: Sequence[float] | None = None,
    smoothing_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate outward local surface normals at points near an LA surface.

    The local tangent plane is fitted by PCA/SVD to the ``k_neighbors`` nearest
    LA vertices.  The sign ambiguity of the fitted plane normal is resolved by
    choosing the direction that points away from ``reference_center``.  When no
    reference center is supplied, the centroid of the complete surface point
    cloud is used.

    Returns
    -------
    anchors
        The nearest LA vertex to each query point.  These anchors are used as
        the surface reference positions for offsetting.
    normals
        Smooth, unit, sign-consistent local normals pointing away from the LA
        reference center.

    Notes
    -----
    This intentionally uses only the LA point cloud, not triangle winding.
    Atrial surface meshes can contain inconsistent element orientations, while
    local PCA is insensitive to face winding.  The away-from-centroid rule is a
    first anatomical sign heuristic and should be validated on several cases.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree

    query = np.asarray(query_points, dtype=float)
    surface = np.asarray(surface_points, dtype=float)
    if query.ndim != 2 or query.shape[1] != 3 or len(query) < 2:
        raise ValueError("query_points must have shape (N, 3) with N >= 2")
    if surface.ndim != 2 or surface.shape[1] != 3 or len(surface) < 3:
        raise ValueError("surface_points must have shape (M, 3) with M >= 3")

    k = int(k_neighbors)
    if k < 3:
        raise ValueError("k_neighbors must be at least 3")
    k = min(k, len(surface))

    if reference_center is None:
        ref_center = np.mean(surface, axis=0)
    else:
        ref_center = np.asarray(reference_center, dtype=float)
        if ref_center.shape != (3,):
            raise ValueError("reference_center must have shape (3,)")

    tree = cKDTree(surface)
    _, nearest_idx = tree.query(query, k=1)
    anchors = surface[np.asarray(nearest_idx, dtype=int)]

    normals = np.empty_like(query)
    previous = None
    for i, anchor in enumerate(anchors):
        _, idx = tree.query(anchor, k=k)
        neighbors = surface[np.atleast_1d(idx).astype(int)]
        centered = neighbors - np.mean(neighbors, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = normalize_vector(vh[-1], name="local LA normal")

        # Primary sign rule: point away from the LA centroid/reference center.
        outward_hint = anchor - ref_center
        hint_norm = np.linalg.norm(outward_hint)
        if hint_norm > _EPS and np.dot(normal, outward_hint) < 0.0:
            normal = -normal

        # Enforce continuity in near-ambiguous regions where local PCA may
        # numerically flip sign from one sample to the next.
        if previous is not None and np.dot(normal, previous) < 0.0:
            normal = -normal

        normals[i] = normal
        previous = normal

    if smoothing_sigma > 0.0 and len(normals) > 2:
        normals = gaussian_filter1d(
            normals,
            sigma=float(smoothing_sigma),
            axis=0,
            mode="nearest",
        )
        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms < _EPS):
            raise ValueError("Normal smoothing produced a zero-length normal")
        normals = normals / norms[:, None]

        # Reapply the outward sign after smoothing and preserve continuity.
        previous = None
        for i in range(len(normals)):
            outward_hint = anchors[i] - ref_center
            if np.linalg.norm(outward_hint) > _EPS and np.dot(normals[i], outward_hint) < 0.0:
                normals[i] *= -1.0
            if previous is not None and np.dot(normals[i], previous) < 0.0:
                normals[i] *= -1.0
            previous = normals[i]

    return anchors, normals


def offset_path_from_surface(
    guide_path: Sequence[Sequence[float]],
    surface_points: Sequence[Sequence[float]],
    offset_distance: float | Sequence[float],
    *,
    k_neighbors: int = 30,
    reference_center: Sequence[float] | None = None,
    normal_smoothing_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a guide onto the LA point cloud and offset it outward.

    ``offset_distance`` is the *centreline-to-LA* distance.  For a tube that
    must clear the LA by ``clearance``, callers should normally supply
    ``tube_radius + clearance`` at every path sample.

    Returns ``(offset_path, surface_anchors, outward_normals)``.
    """
    guide = _as_path(guide_path)
    anchors, normals = local_surface_normals_from_point_cloud(
        guide,
        surface_points,
        k_neighbors=k_neighbors,
        reference_center=reference_center,
        smoothing_sigma=normal_smoothing_sigma,
    )

    distances = np.asarray(offset_distance, dtype=float)
    if distances.ndim == 0:
        distances = np.full(len(guide), float(distances), dtype=float)
    if distances.shape != (len(guide),):
        raise ValueError(
            "offset_distance must be a scalar or contain one value per guide point"
        )
    if np.any(distances <= 0.0):
        raise ValueError("All offset distances must be positive")

    offset = anchors + distances[:, None] * normals
    return offset, anchors, normals


def nearest_vertex_clearance(
    points: Sequence[Sequence[float]],
    surface_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return distance from each point to its nearest LA surface vertex.

    This is a fast diagnostic, not an exact point-to-triangle distance.  It is
    useful before optional PyVista/VTK surface-distance validation.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float)
    surface = np.asarray(surface_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError("surface_points must have shape (M, 3)")
    distances, _ = cKDTree(surface).query(points, k=1)
    return np.asarray(distances, dtype=float)

# -----------------------------------------------------------------------------
# Clearance-aware continuous CS/VoM construction
# -----------------------------------------------------------------------------


def angle_between_vectors(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the unsigned angle between two vectors in radians."""
    aa = normalize_vector(a, name="angle vector a")
    bb = normalize_vector(b, name="angle vector b")
    return float(np.arccos(np.clip(np.dot(aa, bb), -1.0, 1.0)))


def polyline_length(path: Sequence[Sequence[float]]) -> float:
    """Return total Euclidean arc length of an open polyline."""
    path = _as_path(path)
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def clearance_aware_connector_path(
    start: Sequence[float],
    end: Sequence[float],
    start_tangent: Sequence[float],
    end_tangent: Sequence[float],
    surface_points: Sequence[Sequence[float]],
    *,
    start_radius: float,
    end_radius: float,
    clearance: float,
    resolution: float = 350.0,
    handle_scales: Sequence[float] = (0.20, 0.30, 0.40, 0.55, 0.75, 1.00),
    minimum_points: int = 10,
    k_neighbors: int = 30,
    normal_smoothing_sigma: float = 1.5,
    max_iterations: int = 8,
    connector_smoothing_sigma: float = 0.65,
    connector_smoothing_weight: float = 0.15,
    start_clearance_ramp_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Construct a smooth connector and automatically keep it outside the LA.

    A small family of cubic-Hermite candidates is generated from the supplied
    endpoint positions and tangents.  Each candidate is iteratively pushed
    outward along locally estimated LA normals wherever its centreline is too
    close to accommodate the local tube radius plus the requested clearance.

    ``start_clearance_ramp_fraction`` is useful for the CS ostium connector:
    clearance may grow smoothly from zero at the exact CS-ring centre to the
    full requested value over the first part of the connector.  This allows the
    tube to remain attached to the anatomical CS opening rather than forcing
    that endpoint away from the atrial boundary.

    Returns
    -------
    path
        Selected connector centreline.
    radii
        Linear radius profile along that connector.
    diagnostics
        Selected handle scale and clearance/curvature diagnostics.
    """
    from scipy.ndimage import gaussian_filter1d

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start and end must each have shape (3,)")
    if start_radius <= 0.0 or end_radius <= 0.0:
        raise ValueError("connector radii must be positive")
    if clearance < 0.0:
        raise ValueError("clearance must be non-negative")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    if not 0.0 <= start_clearance_ramp_fraction < 1.0:
        raise ValueError("start_clearance_ramp_fraction must satisfy 0 <= value < 1")

    start_tangent = normalize_vector(start_tangent, name="start_tangent")
    end_tangent = normalize_vector(end_tangent, name="end_tangent")
    chord = float(np.linalg.norm(end - start))
    if chord < _EPS:
        raise ValueError("connector endpoints are coincident")

    n_points = max(int(minimum_points), int(np.ceil(chord / float(resolution))) + 1)
    n_points = min(n_points, 80)
    t = np.linspace(0.0, 1.0, n_points)
    radii = np.linspace(float(start_radius), float(end_radius), n_points)

    if start_clearance_ramp_fraction > 0.0:
        clearance_fraction = _smoothstep01(t / float(start_clearance_ramp_fraction))
    else:
        clearance_fraction = np.ones_like(t)
    required_wall_clearance = float(clearance) * clearance_fraction
    required_center_distance = radii + required_wall_clearance

    best: tuple[float, np.ndarray, dict[str, float | int]] | None = None

    for handle_scale in handle_scales:
        if handle_scale <= 0.0:
            continue
        path = hermite_connector_path(
            start,
            end,
            start_tangent,
            end_tangent,
            n_points=n_points,
            handle_scale=float(handle_scale),
        )

        # Keep exact endpoints.  Interior samples are allowed to move outward.
        for _ in range(int(max_iterations)):
            anchors, normals = local_surface_normals_from_point_cloud(
                path,
                surface_points,
                k_neighbors=k_neighbors,
                smoothing_sigma=normal_smoothing_sigma,
            )
            signed_distance = np.einsum("ij,ij->i", path - anchors, normals)
            deficit = required_center_distance - signed_distance
            correction = np.maximum(deficit, 0.0)[:, None] * normals
            correction[0] = 0.0
            correction[-1] = 0.0

            if np.max(np.linalg.norm(correction[1:-1], axis=1), initial=0.0) < 1.0e-4:
                break

            path[1:-1] += correction[1:-1]

            if connector_smoothing_weight > 0.0 and len(path) > 4:
                smoothed = gaussian_filter1d(
                    path,
                    sigma=float(connector_smoothing_sigma),
                    axis=0,
                    mode="nearest",
                )
                w = float(np.clip(connector_smoothing_weight, 0.0, 0.45))
                path[1:-1] = (1.0 - w) * path[1:-1] + w * smoothed[1:-1]
                path[0] = start
                path[-1] = end

            # Preserve the prescribed interface tangents exactly.  This avoids
            # introducing a kink when the connector is concatenated with the
            # preceding/following centreline segment after clearance pushing.
            if len(path) >= 4:
                d_start = max(float(np.linalg.norm(path[1] - start)), _EPS)
                d_end = max(float(np.linalg.norm(path[-2] - end)), _EPS)
                path[1] = start + d_start * start_tangent
                path[-2] = end - d_end * end_tangent

        # One final tangent lock in case the loop converged before smoothing.
        if len(path) >= 4:
            d_start = max(float(np.linalg.norm(path[1] - start)), _EPS)
            d_end = max(float(np.linalg.norm(path[-2] - end)), _EPS)
            path[1] = start + d_start * start_tangent
            path[-2] = end - d_end * end_tangent

        anchors, normals = local_surface_normals_from_point_cloud(
            path,
            surface_points,
            k_neighbors=k_neighbors,
            smoothing_sigma=normal_smoothing_sigma,
        )
        signed_distance = np.einsum("ij,ij->i", path - anchors, normals)
        wall_clearance = signed_distance - radii
        violation = np.maximum(required_wall_clearance - wall_clearance, 0.0)

        # The exact endpoints are fixed interface constraints.  The first CS
        # point in particular belongs to the anatomical opening, so score the
        # movable interior separately from the interface itself.
        interior = slice(1, -1) if len(path) > 2 else slice(0, len(path))
        interior_violation = violation[interior]
        max_violation = float(np.max(interior_violation)) if len(interior_violation) else 0.0
        mean_violation = float(np.mean(interior_violation)) if len(interior_violation) else 0.0

        tangents = path_tangents(path)
        start_angle = angle_between_vectors(tangents[0], start_tangent)
        end_angle = angle_between_vectors(tangents[-1], end_tangent)
        local_turn = [
            angle_between_vectors(tangents[i - 1], tangents[i])
            for i in range(1, len(tangents))
        ]
        curvature = float(np.sum(local_turn))
        length = polyline_length(path)

        clearance_scale = max(float(clearance), 1.0)
        score = (
            1000.0 * max_violation / clearance_scale
            + 200.0 * mean_violation / clearance_scale
            + 20.0 * (start_angle + end_angle)
            + 2.0 * curvature
            + 0.10 * (length / chord)
        )

        diagnostics: dict[str, float | int] = {
            "handle_scale": float(handle_scale),
            "n_points": int(n_points),
            "max_clearance_violation": max_violation,
            "mean_clearance_violation": mean_violation,
            "minimum_estimated_wall_clearance": float(np.min(wall_clearance[1:])),
            "start_tangent_error_deg": float(np.rad2deg(start_angle)),
            "end_tangent_error_deg": float(np.rad2deg(end_angle)),
            "total_turn_deg": float(np.rad2deg(curvature)),
            "path_length": length,
            "score": float(score),
        }
        if best is None or score < best[0]:
            best = (score, path.copy(), diagnostics)

    if best is None:
        raise RuntimeError("No valid connector candidate was generated")

    _, path, diagnostics = best
    return path, radii, diagnostics


def concatenate_paths_and_radii(
    segments: Sequence[tuple[Sequence[Sequence[float]], Sequence[float]]],
    *,
    join_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Join path/radius segments while removing duplicated interface samples.

    Returns ``(path, radii, ranges)``.  Each entry in ``ranges`` is the
    half-open ``[start, stop)`` index range occupied by the corresponding
    segment in the combined arrays.
    """
    if not segments:
        raise ValueError("At least one segment is required")

    path_parts: list[np.ndarray] = []
    radius_parts: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    previous_end: np.ndarray | None = None

    for path_like, radius_like in segments:
        path = _as_path(path_like).copy()
        radii = np.asarray(radius_like, dtype=float)
        if radii.shape != (len(path),):
            raise ValueError("Each radius array must contain one value per path point")
        if np.any(radii <= 0.0):
            raise ValueError("All radii must be positive")

        if previous_end is not None:
            gap = float(np.linalg.norm(path[0] - previous_end))
            if gap > join_tolerance:
                raise ValueError(f"Path segments do not meet (gap={gap:.6g})")
            path = path[1:]
            radii = radii[1:]

        if len(path) == 0:
            ranges.append((cursor, cursor))
            continue

        start_idx = cursor
        path_parts.append(path)
        radius_parts.append(radii)
        cursor += len(path)
        ranges.append((start_idx, cursor))
        previous_end = path[-1].copy()

    return np.vstack(path_parts), np.concatenate(radius_parts), ranges


def build_continuous_tube_from_initial_ring(
    path: Sequence[Sequence[float]],
    radii: Sequence[float],
    initial_ring: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep one continuous circular tube beginning with the exact CS ring.

    The actual CS boundary is used as ring 0.  All subsequent rings are
    circular and use a single rotation-minimizing frame propagated along the
    *whole* CS→mitral→VoM centreline.  Consequently there is no independent
    elbow/tube reorientation and no internal interface seam.

    Returns ``(points, faces, last_ring, tangents)``.
    """
    path = _as_path(path)
    radii = np.asarray(radii, dtype=float)
    if radii.shape != (len(path),):
        raise ValueError("radii must contain one value per path point")
    if np.any(radii <= 0.0):
        raise ValueError("All radii must be positive")

    initial_ring = _as_ring(initial_ring).copy()
    start_center = ring_center(initial_ring)
    if np.linalg.norm(path[0] - start_center) > 1.0e-5:
        raise ValueError("The continuous path must begin at the CS-ring centre")

    tangents = path_tangents(path)
    initial_ring = orient_ring_winding(initial_ring, tangents[0])

    start_radial = initial_ring[0] - start_center
    start_radial -= tangents[0] * np.dot(start_radial, tangents[0])
    if np.linalg.norm(start_radial) < _EPS:
        start_radial = _stable_perpendicular(tangents[0])

    _, radial, binormal = parallel_transport_frames(
        path,
        initial_radial=start_radial,
    )

    n_ring_points = len(initial_ring)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring_points, endpoint=False)
    ca = np.cos(angles)
    sa = np.sin(angles)

    rings: list[np.ndarray] = [initial_ring]
    for i in range(1, len(path)):
        ring = path[i][None, :] + float(radii[i]) * (
            ca[:, None] * radial[i][None, :] + sa[:, None] * binormal[i][None, :]
        )
        rings.append(ring)

    rings_array = np.asarray(rings, dtype=float)
    points = rings_array.reshape(-1, 3)
    faces = stitch_rings(rings_array)
    return points, faces, rings_array[-1].copy(), tangents

# -----------------------------------------------------------------------------
# Hybrid legacy-sequential construction helpers
# -----------------------------------------------------------------------------


def oriented_ring_normal_away_from_landmark(
    ring: Sequence[Sequence[float]],
    landmark: Sequence[float],
) -> np.ndarray:
    """Return the ring normal whose sign points away from ``landmark``.

    The landmark is intended to be a nearby point on the RA side of the CS
    ostium (``Regions.txt`` second row / Python index 1).  It resolves only the
    +/- ambiguity of the best-fit ring normal; the output remains perpendicular
    to the ring rather than using the possibly oblique landmark-to-centre vector
    directly.
    """
    ring = _as_ring(ring)
    landmark = np.asarray(landmark, dtype=float)
    if landmark.shape != (3,):
        raise ValueError("landmark must have shape (3,)")
    center = ring_center(ring)
    away = center - landmark
    if np.linalg.norm(away) < _EPS:
        raise ValueError("landmark is coincident with the ring centre")
    normal = compute_ring_normal(ring, reference_direction=away)
    return normalize_vector(normal, name="oriented ring normal")


def ring_local_tangent(ring: Sequence[Sequence[float]], index: int) -> np.ndarray:
    """Estimate the tangent of an ordered closed ring at one vertex."""
    ring = _as_ring(ring)
    n = len(ring)
    if n < 3:
        raise ValueError("ring needs at least three points")
    i = int(index) % n
    tangent = ring[(i + 1) % n] - ring[(i - 1) % n]
    return normalize_vector(tangent, name="ring local tangent")


def rotating_ring_elbow(
    start_ring: Sequence[Sequence[float]],
    *,
    hinge_index: int,
    angle_deg: float,
    hinge_offset: float,
    desired_start_tangent: Sequence[float],
    resolution: float = 350.0,
    minimum_rings: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build a legacy-style elbow by rigidly rotating the complete start ring.

    This keeps point correspondence exactly as in the original implementation,
    but fixes two important details:

    * angle zero is included exactly once (no duplicate first ring), and
    * the local hinge-axis sign is chosen automatically so positive rotation
      initially moves the ring centre along ``desired_start_tangent``.

    Returns
    -------
    points, faces, last_ring, centerline, rings, diagnostics
    """
    ring = _as_ring(start_ring).copy()
    center = ring_center(ring)
    n = len(ring)
    i = int(hinge_index) % n

    radial = ring[i] - center
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < _EPS:
        raise ValueError("hinge point is coincident with ring centre")
    radial_unit = radial / radial_norm
    pivot = ring[i] + float(hinge_offset) * radial_unit

    axis = ring_local_tangent(ring, i)
    desired = normalize_vector(desired_start_tangent, name="desired_start_tangent")

    # Derivative of R_axis(theta)(center-pivot) at theta=0 is
    # axis x (center-pivot).  Flip the axis so positive theta leaves the ring
    # in the requested anatomical direction.
    initial_arc_tangent = np.cross(axis, center - pivot)
    if np.linalg.norm(initial_arc_tangent) < _EPS:
        raise ValueError("degenerate elbow hinge axis")
    if np.dot(initial_arc_tangent, desired) < 0.0:
        axis = -axis
        initial_arc_tangent = -initial_arc_tangent
    initial_arc_tangent = normalize_vector(initial_arc_tangent, name="elbow initial tangent")

    bend_radius = float(np.linalg.norm(center - pivot))
    total_angle = float(np.deg2rad(angle_deg))
    arc_length = abs(total_angle) * bend_radius
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    n_rings = max(int(minimum_rings), int(np.ceil(arc_length / float(resolution))) + 1)

    angles = np.linspace(0.0, total_angle, n_rings)
    rings = []
    centers = []
    for angle in angles:
        rot = Rotation.from_rotvec(float(angle) * axis)
        rotated_ring = rot.apply(ring - pivot) + pivot
        rings.append(rotated_ring)
        centers.append(ring_center(rotated_ring))

    rings_array = np.asarray(rings, dtype=float)
    centers_array = np.asarray(centers, dtype=float)
    points = rings_array.reshape(-1, 3)
    faces = stitch_rings(rings_array)
    center_tangents = path_tangents(centers_array)

    diagnostics = {
        "bend_radius": bend_radius,
        "arc_length": arc_length,
        "start_tangent_error_deg": float(np.rad2deg(angle_between_vectors(center_tangents[0], desired))),
        "n_rings": int(n_rings),
    }
    return (
        points,
        faces,
        rings_array[-1].copy(),
        centers_array,
        rings_array,
        diagnostics,
    )


def rigid_align_path_to_start(
    path: Sequence[Sequence[float]],
    target_center: Sequence[float],
    target_tangent: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly align an open guide path to a new start centre/tangent.

    Shape and curvature are preserved.  Returns ``(aligned_path, R, t)`` where
    ``aligned = original @ R.T + t``.
    """
    path = _as_path(path).copy()
    target_center = np.asarray(target_center, dtype=float)
    if target_center.shape != (3,):
        raise ValueError("target_center must have shape (3,)")
    target_tangent = normalize_vector(target_tangent, name="target_tangent")

    source_tangent = normalize_vector(path[1] - path[0], name="guide start tangent")
    rot = rotation_matrix_from_vectors(source_tangent, target_tangent)
    rotated = path @ rot.T
    translation = target_center - rotated[0]
    aligned = rotated + translation
    aligned[0] = target_center
    return aligned, rot, translation


def apply_rigid_transform(
    points: Sequence[Sequence[float]],
    rotation: np.ndarray,
    translation: Sequence[float],
) -> np.ndarray:
    """Apply the transform returned by :func:`rigid_align_path_to_start`."""
    pts = np.asarray(points, dtype=float)
    rot = np.asarray(rotation, dtype=float)
    trans = np.asarray(translation, dtype=float)
    if rot.shape != (3, 3) or trans.shape != (3,):
        raise ValueError("rotation must be (3,3) and translation must be (3,)")
    return pts @ rot.T + trans


def radius_profile_from_start_ring(
    n_points: int,
    start_radius: float,
    end_radius: float,
    *,
    taper_start_fraction: float = 0.5,
) -> np.ndarray:
    """Alias with an explicit name for sequential inherited-ring sweeps."""
    return radius_profile(
        n_points,
        start_radius,
        end_radius,
        taper_start_fraction=taper_start_fraction,
    )


def build_sequential_tube_from_ring(
    start_ring: Sequence[Sequence[float]],
    path: Sequence[Sequence[float]],
    *,
    end_radius: float,
    taper_start_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep a tube by propagating the *previous* ring along the path.

    Unlike rebuilding every cross-section from an independent global frame,
    each ring is obtained by the minimal rotation from the previous path tangent
    to the next.  This preserves ring indexing/winding and exactly inherits the
    elbow's final ring as the tube's first ring.

    Returns ``(points, faces, last_ring, tangents, rings, radii)``.
    """
    path = _as_path(path).copy()
    first_ring = _as_ring(start_ring).copy()
    first_center = ring_center(first_ring)
    path += first_center - path[0]
    path[0] = first_center

    tangents = path_tangents(path)
    start_radius = ring_radius(first_ring)
    radii = radius_profile_from_start_ring(
        len(path),
        start_radius,
        float(end_radius),
        taper_start_fraction=taper_start_fraction,
    )

    # Make sure the geometric plane of the inherited first ring and the path's
    # first tangent use the same normal direction.  We do not change winding or
    # point order; this is only used for diagnostics/transport direction.
    first_normal = compute_ring_normal(first_ring, reference_direction=tangents[0])
    if np.dot(first_normal, tangents[0]) < 0.0:
        tangents = -tangents

    rings = [first_ring]
    previous_ring = first_ring
    previous_center = first_center
    previous_radius = start_radius

    for i in range(1, len(path)):
        rot = rotation_matrix_from_vectors(tangents[i - 1], tangents[i])
        offsets = previous_ring - previous_center
        offsets = offsets @ rot.T
        if previous_radius < _EPS:
            raise ValueError("previous ring radius became zero")
        offsets *= float(radii[i] / previous_radius)
        current_ring = path[i][None, :] + offsets
        rings.append(current_ring)
        previous_ring = current_ring
        previous_center = path[i]
        previous_radius = float(radii[i])

    rings_array = np.asarray(rings, dtype=float)
    points = rings_array.reshape(-1, 3)
    faces = stitch_rings(rings_array)
    return points, faces, rings_array[-1].copy(), tangents, rings_array, radii


def discrete_curvature_radius(path: Sequence[Sequence[float]]) -> np.ndarray:
    """Estimate the local circumcircle radius for each interior path sample."""
    p = _as_path(path)
    radii = np.full(len(p), np.inf, dtype=float)
    for i in range(1, len(p) - 1):
        a = float(np.linalg.norm(p[i] - p[i - 1]))
        b = float(np.linalg.norm(p[i + 1] - p[i]))
        c = float(np.linalg.norm(p[i + 1] - p[i - 1]))
        cross = np.cross(p[i] - p[i - 1], p[i + 1] - p[i - 1])
        twice_area = float(np.linalg.norm(cross))
        if twice_area < _EPS:
            radii[i] = np.inf
        else:
            # R = abc / (4A), and twice_area = 2A.
            radii[i] = (a * b * c) / (2.0 * twice_area)
    return radii


def minimum_curvature_ratio(
    path: Sequence[Sequence[float]],
    tube_radii: float | Sequence[float],
) -> float:
    """Return min(local centreline curvature radius / local tube radius)."""
    p = _as_path(path)
    r = np.asarray(tube_radii, dtype=float)
    if r.ndim == 0:
        r = np.full(len(p), float(r), dtype=float)
    if r.shape != (len(p),):
        raise ValueError("tube_radii must be scalar or one value per path point")
    local = discrete_curvature_radius(p)
    mask = np.isfinite(local) & (r > _EPS)
    if not np.any(mask):
        return float("inf")
    return float(np.min(local[mask] / r[mask]))


def nearest_vertex_distances(
    points: Sequence[Sequence[float]],
    surface_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Distance from query points to nearest LA vertex (fast diagnostic)."""
    from scipy.spatial import cKDTree
    q = np.asarray(points, dtype=float)
    s = np.asarray(surface_points, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or s.ndim != 2 or s.shape[1] != 3:
        raise ValueError("points and surface_points must both have shape (N,3)")
    d, _ = cKDTree(s).query(q, k=1)
    return np.asarray(d, dtype=float)


def combine_ring_sequences(*ring_sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join sequential ring arrays, dropping duplicate interface rings."""
    combined: list[np.ndarray] = []
    for seq in ring_sequences:
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("each ring sequence must have shape (M,N,3)")
        if not combined:
            combined.extend(arr)
        else:
            if arr.shape[1] != combined[-1].shape[0]:
                raise ValueError("all ring sequences must use the same ring point count")
            if np.allclose(arr[0], combined[-1], atol=1.0e-6, rtol=0.0):
                combined.extend(arr[1:])
            else:
                raise ValueError("adjacent ring sequences do not share the exact interface ring")
    rings = np.asarray(combined, dtype=float)
    points = rings.reshape(-1, 3)
    faces = stitch_rings(rings)
    return points, faces, rings

# -----------------------------------------------------------------------------
# Generic graph/path helpers used by build_biatrial_connections.py
# -----------------------------------------------------------------------------


def triangle_boundary_loops(faces):
    """Return oriented boundary loops of a manifold triangular surface.

    Parameters
    ----------
    faces : array-like, shape (N, 3)
        Triangle vertex indices.

    Returns
    -------
    list[numpy.ndarray]
        Boundary vertex loops, sorted from longest to shortest.
    """
    from collections import defaultdict

    triangles = np.asarray(faces, dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (N, 3)")

    edge_rows = defaultdict(list)
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            edge_rows[key].append((int(u), int(v)))

    directed_boundary = [rows[0] for rows in edge_rows.values() if len(rows) == 1]
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for u, v in directed_boundary:
        outgoing[u].append(v)
        incoming[v].append(u)

    boundary_vertices = set(outgoing) | set(incoming)
    bad = [
        v for v in boundary_vertices
        if len(outgoing.get(v, [])) != 1 or len(incoming.get(v, [])) != 1
    ]
    if bad:
        raise ValueError(
            "Boundary is not a collection of simple oriented loops; "
            f"{len(bad)} boundary vertices have degree != 2"
        )

    next_vertex = {u: values[0] for u, values in outgoing.items()}
    unused = set(next_vertex)
    loops = []

    while unused:
        start = min(unused)
        current = start
        loop = [start]
        for _ in range(len(next_vertex) + 1):
            nxt = next_vertex[current]
            if nxt == start:
                break
            if nxt in loop:
                raise ValueError("Boundary loop revisited itself before closing")
            loop.append(nxt)
            current = nxt
        else:
            raise ValueError("Boundary loop failed to close")

        for vertex in loop:
            unused.discard(vertex)
        loops.append(np.asarray(loop, dtype=int))

    loops.sort(key=lambda loop: (-len(loop), int(np.min(loop))))
    return loops


def nearest_reference_vertices(query_points, reference_points, *, k=1):
    """Return Euclidean nearest-neighbour distances and reference indices.

    This is a thin cKDTree wrapper that accepts either one 3-D query point or
    an ``(N, 3)`` query array.
    """
    from scipy.spatial import cKDTree

    query = np.asarray(query_points, dtype=float)
    reference = np.asarray(reference_points, dtype=float)

    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
        raise ValueError("reference_points must have shape (N, 3) with N > 0")
    if query.shape == (3,):
        pass
    elif query.ndim == 2 and query.shape[1] == 3:
        pass
    else:
        raise ValueError("query_points must have shape (3,) or (N, 3)")

    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1")
    k = min(k, len(reference))

    distances, indices = cKDTree(reference).query(query, k=k)
    return np.asarray(distances), np.asarray(indices, dtype=int)


def mesh_shortest_path_indices(points, faces, start_index, end_index):
    """Return the vertex-index shortest path along triangular-mesh edges.

    Edge weights are Euclidean edge lengths, so the result is a discrete
    surface geodesic along the existing mesh connectivity.
    """
    import heapq
    from collections import defaultdict

    pts = np.asarray(points, dtype=float)
    triangles = np.asarray(faces, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")

    start = int(start_index)
    end = int(end_index)
    if not (0 <= start < len(pts)) or not (0 <= end < len(pts)):
        raise IndexError("start_index/end_index is outside the point array")
    if start == end:
        return np.asarray([start], dtype=int)

    adjacency = defaultdict(dict)
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            u = int(u)
            v = int(v)
            weight = float(np.linalg.norm(pts[u] - pts[v]))
            old = adjacency[u].get(v)
            if old is None or weight < old:
                adjacency[u][v] = weight
                adjacency[v][u] = weight

    distance = np.full(len(pts), np.inf, dtype=float)
    previous = np.full(len(pts), -1, dtype=int)
    distance[start] = 0.0
    heap = [(0.0, start)]

    while heap:
        current_distance, u = heapq.heappop(heap)
        if current_distance != distance[u]:
            continue
        if u == end:
            break
        for v, weight in adjacency.get(u, {}).items():
            candidate = current_distance + weight
            if candidate < distance[v]:
                distance[v] = candidate
                previous[v] = u
                heapq.heappush(heap, (candidate, v))

    if not np.isfinite(distance[end]):
        raise ValueError(f"No mesh-edge path exists between vertices {start} and {end}")

    path = []
    current = end
    while current != -1:
        path.append(current)
        if current == start:
            break
        current = int(previous[current])
    path.reverse()

    if not path or path[0] != start or path[-1] != end:
        raise ValueError("Failed to reconstruct shortest mesh path")
    return np.asarray(path, dtype=int)


def path_vertex_indices_at_fractions(path_points, fractions):
    """Return path vertex positions nearest requested normalized arc lengths.

    ``0.0`` selects the first path vertex and ``1.0`` the final path vertex.
    Returned indices are positions within ``path_points`` rather than global
    mesh vertex IDs.
    """
    path = np.asarray(path_points, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) == 0:
        raise ValueError("path_points must have shape (N, 3) with N > 0")

    requested = np.asarray(list(fractions), dtype=float)
    if requested.ndim != 1:
        raise ValueError("fractions must be one-dimensional")
    if np.any(~np.isfinite(requested)) or np.any(requested < 0.0) or np.any(requested > 1.0):
        raise ValueError("all path fractions must lie in [0, 1]")
    if len(requested) == 0:
        return np.empty(0, dtype=int)
    if len(path) == 1:
        return np.zeros(len(requested), dtype=int)

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= _EPS:
        raise ValueError("path has zero physical arc length")

    targets = requested * total
    result = []
    for target in targets:
        right = int(np.searchsorted(cumulative, target, side="left"))
        if right <= 0:
            result.append(0)
        elif right >= len(cumulative):
            result.append(len(cumulative) - 1)
        else:
            left = right - 1
            if abs(target - cumulative[left]) <= abs(cumulative[right] - target):
                result.append(left)
            else:
                result.append(right)
    return np.asarray(result, dtype=int)

# -----------------------------------------------------------------------------
# Generic surface-component / ruled-strip helpers used by BB bridge construction
# -----------------------------------------------------------------------------


def triangle_vertex_components(faces, n_points=None):
    """Return edge-connected vertex components of a triangular mesh.

    Only vertices referenced by ``faces`` are included.  Components are sorted
    from largest to smallest.  Vertex IDs are kept in the indexing of the input
    mesh.
    """
    from collections import defaultdict, deque

    triangles = np.asarray(faces, dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (N, 3)")
    if len(triangles) == 0:
        return []

    if n_points is None:
        n_points = int(np.max(triangles)) + 1
    n_points = int(n_points)
    if np.any(triangles < 0) or np.any(triangles >= n_points):
        raise IndexError("triangle vertex index is outside n_points")

    adjacency = defaultdict(set)
    used = set(map(int, np.unique(triangles)))
    for a, b, c in triangles:
        a, b, c = int(a), int(b), int(c)
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    remaining = set(used)
    components = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        component = []
        while queue:
            u = queue.popleft()
            component.append(u)
            for v in adjacency.get(u, ()):
                if v in remaining:
                    remaining.remove(v)
                    queue.append(v)
        components.append(np.asarray(sorted(component), dtype=int))

    components.sort(key=lambda x: (-len(x), int(x[0]) if len(x) else -1))
    return components


def faces_within_vertex_set(faces, vertex_indices):
    """Return triangles whose three vertices are all in ``vertex_indices``."""
    triangles = np.asarray(faces, dtype=int)
    vertices = np.asarray(vertex_indices, dtype=int).reshape(-1)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (N, 3)")
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=int)
    mask = np.all(np.isin(triangles, vertices), axis=1)
    return triangles[mask]


def nearest_vertex_to_centroid(points, vertex_indices):
    """Return the component vertex closest to its Euclidean centroid."""
    pts = np.asarray(points, dtype=float)
    vertices = np.asarray(vertex_indices, dtype=int).reshape(-1)
    if len(vertices) == 0:
        raise ValueError("vertex_indices is empty")
    centroid = np.mean(pts[vertices], axis=0)
    local = int(np.argmin(np.linalg.norm(pts[vertices] - centroid, axis=1)))
    return int(vertices[local])


def local_pca_normal(points, center_index, *, candidate_indices=None, k_neighbors=30):
    """Estimate an unsigned local surface normal around one mesh vertex by PCA.

    The sign is intentionally unspecified because callers that only project a
    direction onto the tangent plane do not need an oriented normal.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=float)
    center = int(center_index)
    if not (0 <= center < len(pts)):
        raise IndexError("center_index is outside points")

    if candidate_indices is None:
        candidates = np.arange(len(pts), dtype=int)
    else:
        candidates = np.asarray(candidate_indices, dtype=int).reshape(-1)
    if len(candidates) < 3:
        raise ValueError("at least three candidate points are required for PCA")

    k = min(max(3, int(k_neighbors)), len(candidates))
    tree = cKDTree(pts[candidates])
    _, local_idx = tree.query(pts[center], k=k)
    local_idx = np.asarray(local_idx, dtype=int).reshape(-1)
    cloud = pts[candidates[local_idx]]
    centered = cloud - np.mean(cloud, axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("local point cloud is degenerate; cannot estimate tangent plane")
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return normalize_vector(vh[-1], name="local PCA normal")


def polyline_arc_fractions(path):
    """Return normalized cumulative arc length at every polyline vertex."""
    line = np.asarray(path, dtype=float)
    if line.ndim != 2 or line.shape[1] != 3 or len(line) == 0:
        raise ValueError("path must have shape (N, 3) with N > 0")
    if len(line) == 1:
        return np.asarray([0.0], dtype=float)
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= _EPS:
        raise ValueError("polyline has zero arc length")
    return cumulative / total


def sample_polyline_at_fractions(path, fractions):
    """Sample an open polyline at normalized arc-length fractions.

    Sampling is piecewise linear along the original polyline edges.  Therefore,
    if the polyline is a mesh-edge geodesic, every returned point remains on
    those mesh edges.
    """
    line = np.asarray(path, dtype=float)
    requested = np.asarray(fractions, dtype=float).reshape(-1)
    if line.ndim != 2 or line.shape[1] != 3 or len(line) == 0:
        raise ValueError("path must have shape (N, 3) with N > 0")
    if np.any(~np.isfinite(requested)) or np.any(requested < 0.0) or np.any(requested > 1.0):
        raise ValueError("fractions must lie in [0, 1]")
    if len(requested) == 0:
        return np.empty((0, 3), dtype=float)
    if len(line) == 1:
        return np.repeat(line, len(requested), axis=0)

    fractions_at_vertices = polyline_arc_fractions(line)
    out = np.empty((len(requested), 3), dtype=float)
    for i, value in enumerate(requested):
        right = int(np.searchsorted(fractions_at_vertices, value, side="left"))
        if right <= 0:
            out[i] = line[0]
            continue
        if right >= len(line):
            out[i] = line[-1]
            continue
        left = right - 1
        f0 = float(fractions_at_vertices[left])
        f1 = float(fractions_at_vertices[right])
        if np.isclose(value, f0, atol=1.0e-12, rtol=0.0):
            out[i] = line[left]
        elif np.isclose(value, f1, atol=1.0e-12, rtol=0.0):
            out[i] = line[right]
        else:
            alpha = (float(value) - f0) / max(f1 - f0, _EPS)
            out[i] = (1.0 - alpha) * line[left] + alpha * line[right]
    return out


def ruled_surface_between_geodesics(path_a, path_b, *, cross_spacing=350.0):
    """Build a triangulated ruled surface between two open attachment paths.

    All original vertices from both input paths are retained on the two boundary
    rows.  A common along-path parameterization is formed from the union of the
    original normalized arc-length positions, so no attachment vertex is
    discarded even when the two paths contain different numbers of vertices.

    Parameters
    ----------
    path_a, path_b : array-like, shape (N,3)/(M,3)
        Ordered attachment polylines.
    cross_spacing : float
        Maximum requested spacing between interpolated rows across the bridge,
        in the same coordinate units as the input geometry.

    Returns
    -------
    vertices : (K,3) ndarray
    faces : (F,3) ndarray
    face_cross_fraction : (F,) ndarray
        Mid-strip normalized position from path_a (0) to path_b (1), useful for
        assigning a thickness/scalar profile.
    common_fractions : (P,) ndarray
        Along-path fractions used by both attachment rows.
    """
    a = np.asarray(path_a, dtype=float)
    b = np.asarray(path_b, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 2:
        raise ValueError("path_a must have shape (N,3) with N >= 2")
    if b.ndim != 2 or b.shape[1] != 3 or len(b) < 2:
        raise ValueError("path_b must have shape (M,3) with M >= 2")
    spacing = float(cross_spacing)
    if spacing <= 0.0:
        raise ValueError("cross_spacing must be positive")

    fa = polyline_arc_fractions(a)
    fb = polyline_arc_fractions(b)
    # Keep the exact normalized positions of every original vertex from both
    # paths.  This guarantees that every geodesic attachment vertex appears on
    # the connector boundary, even when the two paths have different counts.
    common = np.unique(np.concatenate((fa, fb)))
    common[0] = 0.0
    common[-1] = 1.0

    aa = sample_polyline_at_fractions(a, common)
    bb = sample_polyline_at_fractions(b, common)
    gaps = np.linalg.norm(bb - aa, axis=1)
    n_cross_intervals = max(1, int(np.ceil(float(np.max(gaps)) / spacing)))

    rows = []
    for i in range(n_cross_intervals + 1):
        alpha = i / n_cross_intervals
        rows.append((1.0 - alpha) * aa + alpha * bb)
    grid = np.asarray(rows, dtype=float)

    n_rows, n_along, _ = grid.shape
    vertices = grid.reshape(-1, 3)
    faces = []
    cross_fraction = []
    for i in range(n_rows - 1):
        strip_mid = (i + 0.5) / (n_rows - 1)
        for j in range(n_along - 1):
            v0 = i * n_along + j
            v1 = v0 + 1
            v3 = (i + 1) * n_along + j
            v2 = v3 + 1
            faces.append([v0, v1, v3])
            faces.append([v1, v2, v3])
            cross_fraction.extend((strip_mid, strip_mid))

    return (
        vertices,
        np.asarray(faces, dtype=int),
        np.asarray(cross_fraction, dtype=float),
        common,
    )

# -----------------------------------------------------------------------------
# Generic paired-surface helpers used by the BB connector workflow
# -----------------------------------------------------------------------------


def boundary_vertex_set(faces, *, warn_irregular=False):
    """Return vertices on triangle edges used by exactly one face.

    Unlike :func:`triangle_boundary_loops`, this helper does not require a
    perfectly simple closed boundary.  It is therefore useful for mapped
    patient surfaces that may contain a small slit or branched boundary.
    """
    from collections import Counter, defaultdict

    triangles = np.asarray(faces, dtype=int)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("faces must be a non-empty (N, 3) triangle array")

    counts = Counter()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            counts[(min(int(u), int(v)), max(int(u), int(v)))] += 1
    edges = [edge for edge, count in counts.items() if count == 1]
    if not edges:
        raise ValueError("surface has no one-sided triangle edges")

    boundary = np.unique(np.asarray(edges, dtype=int).reshape(-1)).astype(int)
    irregular = []
    if warn_irregular:
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        irregular = sorted(v for v, d in degree.items() if d != 2)
    return boundary, irregular


def assign_two_mesh_components_by_references(
    points,
    faces,
    reference_a,
    reference_b,
):
    """Assign the two largest triangle components to two spatial references.

    The assignment minimizes the sum of mean nearest-point distances.  Returns
    ``(component_a, component_b, total_component_count)`` using original mesh
    vertex indices.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=float)
    ref_a = np.asarray(reference_a, dtype=float)
    ref_b = np.asarray(reference_b, dtype=float)
    components = triangle_vertex_components(faces, n_points=len(pts))
    if len(components) < 2:
        raise ValueError("mesh has fewer than two connected triangle components")
    c0, c1 = components[:2]
    tree_a = cKDTree(ref_a)
    tree_b = cKDTree(ref_b)

    def mean_dist(component, tree):
        d, _ = tree.query(pts[component], k=1)
        return float(np.mean(d))

    direct = mean_dist(c0, tree_a) + mean_dist(c1, tree_b)
    swapped = mean_dist(c0, tree_b) + mean_dist(c1, tree_a)
    if direct <= swapped:
        return c0, c1, len(components)
    return c1, c0, len(components)


def concatenate_paths_through_center(path_center_to_a, path_center_to_b):
    """Return ``a -> center -> b`` without duplicating the center vertex."""
    left = np.asarray(path_center_to_a, dtype=int)[::-1]
    right = np.asarray(path_center_to_b, dtype=int)
    if len(left) == 0 or len(right) == 0:
        raise ValueError("center paths must be non-empty")
    return np.concatenate((left, right[1:])).astype(int)


def orient_second_polyline_for_pairing(line_a, line_b):
    """Orient line_b to minimize the sum of paired endpoint distances."""
    a = np.asarray(line_a, dtype=float)
    b = np.asarray(line_b, dtype=float)
    direct = float(np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[-1] - b[-1]))
    reverse = float(np.linalg.norm(a[0] - b[-1]) + np.linalg.norm(a[-1] - b[0]))
    if reverse < direct:
        return b[::-1].copy(), True, direct, reverse
    return b, False, direct, reverse


def nearest_component_contact_pairs(points, component_a, component_b, *, max_pairs=200):
    """Return closest unique vertex pairs sampled in both directions."""
    from scipy.spatial import cKDTree

    if int(max_pairs) < 2:
        raise ValueError("max_pairs must be at least 2")
    pts = np.asarray(points, dtype=float)
    comp_a = np.asarray(component_a, dtype=int)
    comp_b = np.asarray(component_b, dtype=int)
    tree_a = cKDTree(pts[comp_a])
    tree_b = cKDTree(pts[comp_b])
    d_b, i_a = tree_a.query(pts[comp_b], k=1)
    d_a, i_b = tree_b.query(pts[comp_a], k=1)

    pairs = {}
    for b_local, (dist, a_local) in enumerate(zip(d_b, i_a)):
        pair = (int(comp_a[int(a_local)]), int(comp_b[b_local]))
        pairs[pair] = min(float(dist), pairs.get(pair, np.inf))
    for a_local, (dist, b_local) in enumerate(zip(d_a, i_b)):
        pair = (int(comp_a[a_local]), int(comp_b[int(b_local)]))
        pairs[pair] = min(float(dist), pairs.get(pair, np.inf))

    ordered = sorted((dist, a, b) for (a, b), dist in pairs.items())
    return ordered[: min(int(max_pairs), len(ordered))]


def principal_axis_from_paired_midpoints(points, pairs):
    """Return the dominant PCA axis of midpoint coordinates of vertex pairs."""
    pts = np.asarray(points, dtype=float)
    if len(pairs) < 2:
        raise ValueError("at least two paired points are required")
    mid = np.asarray([0.5 * (pts[a] + pts[b]) for _, a, b in pairs], dtype=float)
    centered = mid - np.mean(mid, axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular) == 0 or singular[0] < 1.0e-12:
        raise ValueError("paired midpoint cloud has no resolvable principal axis")
    return normalize_vector(vh[0], name="paired midpoint principal axis")


def geodesic_through_candidate_band(points, faces, candidate_vertices, axis):
    """Build a mesh-edge shortest path between directional extremes of candidates."""
    pts = np.asarray(points, dtype=float)
    candidates = np.unique(np.asarray(candidate_vertices, dtype=int))
    if len(candidates) < 2:
        raise ValueError("candidate band contains fewer than two vertices")
    direction = normalize_vector(axis, name="candidate band axis")
    center = np.mean(pts[candidates], axis=0)
    projection = (pts[candidates] - center) @ direction
    start = int(candidates[int(np.argmin(projection))])
    end = int(candidates[int(np.argmax(projection))])
    if start == end:
        xyz = pts[candidates]
        d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
        i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
        start, end = int(candidates[i]), int(candidates[j])
    path = mesh_shortest_path_indices(pts, faces, start, end)
    return start, end, np.asarray(path, dtype=int)


def geodesic_line_through_centroid_toward_reference(
    points,
    faces,
    component_vertices,
    boundary_vertices,
    reference_point,
):
    """Geodesic boundary-to-boundary line through a component centroid.

    One endpoint is the boundary vertex closest to ``reference_point``.  The
    second is selected on the opposite side of the centroid along the same
    chord direction.
    """
    pts = np.asarray(points, dtype=float)
    component = np.asarray(component_vertices, dtype=int)
    boundary = np.asarray(boundary_vertices, dtype=int)
    ref = np.asarray(reference_point, dtype=float)
    center = nearest_vertex_to_centroid(pts, component)
    endpoint_a = int(boundary[int(np.argmin(np.linalg.norm(pts[boundary] - ref, axis=1)))])

    axis_to_a = normalize_vector(pts[endpoint_a] - pts[center], name="centroid reference axis")
    opposite_axis = -axis_to_a
    vectors = pts[boundary] - pts[center]
    projection = vectors @ opposite_axis
    lateral = np.linalg.norm(vectors - projection[:, None] * opposite_axis[None, :], axis=1)
    valid = boundary != endpoint_a
    if not np.any(valid):
        raise ValueError("boundary contains only one usable endpoint")
    max_projection = float(np.max(projection[valid]))
    tol = max(1.0e-9, 1.0e-8 * max(1.0, abs(max_projection)))
    tied = np.flatnonzero(valid & (projection >= max_projection - tol))
    endpoint_b = int(boundary[tied[int(np.argmin(lateral[tied]))]])

    path_a = np.asarray(mesh_shortest_path_indices(pts, faces, center, endpoint_a), dtype=int)
    path_b = np.asarray(mesh_shortest_path_indices(pts, faces, center, endpoint_b), dtype=int)
    line = concatenate_paths_through_center(path_a, path_b)
    return center, endpoint_a, endpoint_b, line, path_a, path_b


def parallel_geodesic_line_through_centroid(
    points,
    faces,
    component_vertices,
    boundary_vertices,
    reference_direction,
    *,
    target_negative_vertices,
    target_positive_vertices,
    normal_neighbors=30,
):
    """Geodesic line through a centroid, parallel to a projected reference direction.

    Full directional boundary paths are found first.  Each side is then limited
    by the requested geodesic vertex counts, allowing one surface to inherit the
    attachment scale of another surface.
    """
    pts = np.asarray(points, dtype=float)
    component = np.asarray(component_vertices, dtype=int)
    boundary = np.asarray(boundary_vertices, dtype=int)
    center = nearest_vertex_to_centroid(pts, component)
    normal = local_pca_normal(
        pts, center, candidate_indices=component, k_neighbors=int(normal_neighbors)
    )
    direction = np.asarray(reference_direction, dtype=float)
    projected = direction - normal * float(np.dot(direction, normal))
    if np.linalg.norm(projected) < 1.0e-12:
        cloud = pts[component]
        centered = cloud - np.mean(cloud, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        options = [vh[0], vh[1]]
        projected = max(options, key=lambda v: abs(float(np.dot(v, direction))))
        if np.dot(projected, direction) < 0.0:
            projected = -projected
    axis = normalize_vector(projected, name="projected reference direction")

    vectors = pts[boundary] - pts[center]
    projection = vectors @ axis
    lateral = np.linalg.norm(vectors - projection[:, None] * axis[None, :], axis=1)
    pmin, pmax = float(np.min(projection)), float(np.max(projection))
    tol = max(1.0e-9, 1.0e-8 * max(1.0, abs(pmin), abs(pmax)))
    neg_ids = np.flatnonzero(projection <= pmin + tol)
    pos_ids = np.flatnonzero(projection >= pmax - tol)
    negative_boundary = int(boundary[neg_ids[int(np.argmin(lateral[neg_ids]))]])
    positive_boundary = int(boundary[pos_ids[int(np.argmin(lateral[pos_ids]))]])

    full_negative = np.asarray(
        mesh_shortest_path_indices(pts, faces, center, negative_boundary), dtype=int
    )
    full_positive = np.asarray(
        mesh_shortest_path_indices(pts, faces, center, positive_boundary), dtype=int
    )
    n_neg = max(2, int(target_negative_vertices))
    n_pos = max(2, int(target_positive_vertices))
    negative = full_negative[:n_neg]
    positive = full_positive[:n_pos]
    line = concatenate_paths_through_center(negative, positive)
    return {
        "center": center,
        "negative_attachment": int(negative[-1]),
        "positive_attachment": int(positive[-1]),
        "line_indices": line,
        "axis": axis,
        "normal": normal,
        "negative_boundary": negative_boundary,
        "positive_boundary": positive_boundary,
        "full_negative_count": len(full_negative),
        "full_positive_count": len(full_positive),
        "negative_limited": len(full_negative) < n_neg,
        "positive_limited": len(full_positive) < n_pos,
    }
