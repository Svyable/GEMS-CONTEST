"""P4: orientation-aware curved continuation (curvature-prior geodesic paths).

Research: docs/research/METRIC_AWARE_RESEARCH_2026-10-10.md §P4-P5
(Chen et al., curvature-prior geodesic paths, 2023). The straight
endpoint-extension control (ENDPOINT-EXT-02) was measured against a full
baseline and REJECTED (deltas ~1e-7–1e-8 vs the 1.5× promotion margin), so
the control comparison is closed. This candidate tests whether continuations
that *bend toward probability support* recover continuations the straight
version missed.

Fixed budgets (per the research doc: "Fix endpoint/path budgets and penalize
unsupported bridges"):
- ``max_paths``: cap on continuations painted per map.
- ``max_extension_m``: metric cap on path length (default 300 m, the metric's
  own tolerance radius).
- ``min_prob``: per-pixel support floor; stepping onto p < min_prob ends the
  path, so bridges across empty background are never grown.
- ``support_threshold``: mean probability along the *finished* path must clear
  this, otherwise the whole path is discarded (unsupported bridges rejected,
  not painted).
- ``curvature_penalty``: cost weight on squared heading change per step — the
  curvature prior that distinguishes this from straight extension.
- ``beam_width``: beam size for the path search.

Everything here is pure numpy (no skimage): the endpoint/orientation helpers
are re-implemented here rather than imported from
``gems.endpoint_extension`` so the module (and its tests) run on CPU-only
hosts without the skimage dependency. The math is the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurvedContinuationParams:
    """Fixed budgets for one curved-continuation run."""

    max_extension_m: float = 300.0
    pixel_size_m: float = 100.0
    curvature_penalty: float = 0.5
    beam_width: int = 3
    min_prob: float = 0.3
    support_threshold: float = 0.4
    min_path_length: int = 3
    max_paths: int = 200
    decay_rate: float = 0.15
    window_size: int = 15

    def __post_init__(self) -> None:
        if not self.max_extension_m > 0:
            raise ValueError("max_extension_m must be positive")
        if not self.pixel_size_m > 0:
            raise ValueError("pixel_size_m must be positive")
        if not self.curvature_penalty >= 0:
            raise ValueError("curvature_penalty must be non-negative")
        if not self.beam_width >= 1:
            raise ValueError("beam_width must be >= 1")
        if not 0.0 <= self.min_prob < 1.0:
            raise ValueError("min_prob must lie in [0, 1)")
        if not 0.0 <= self.support_threshold <= 1.0:
            raise ValueError("support_threshold must lie in [0, 1]")
        if not self.min_path_length >= 1:
            raise ValueError("min_path_length must be >= 1")
        if not self.max_paths >= 1:
            raise ValueError("max_paths must be >= 1")
        if not 0.0 < self.decay_rate < 1.0:
            raise ValueError("decay_rate must lie in (0, 1)")
        if not (self.window_size >= 3 and self.window_size % 2 == 1):
            raise ValueError("window_size must be odd and >= 3")

    @property
    def max_steps(self) -> int:
        return max(1, round(self.max_extension_m / self.pixel_size_m))


# 8-neighborhood headings: (dr, dc), heading angle in radians (row grows down).
_HEADINGS: tuple[tuple[int, int], ...] = (
    (0, 1),    # 0: east
    (1, 1),    # 1: south-east
    (1, 0),    # 2: south
    (1, -1),   # 3: south-west
    (0, -1),   # 4: west
    (-1, -1),  # 5: north-west
    (-1, 0),   # 6: north
    (-1, 1),   # 7: north-east
)


def _zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """One-pixel-wide skeleton via Zhang–Suen thinning (numpy only).

    Deterministic; identical input gives identical output. Used instead of
    skimage.morphology.skeletonize so this module stays importable without
    optional dependencies.
    """
    img = binary.astype(bool).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1)
            # 8-neighborhood in P2..P9 order (clockwise from north).
            p = [padded[1:-1, 2:], padded[:-2, 2:], padded[:-2, 1:-1],
                 padded[:-2, :-2], padded[1:-1, :-2], padded[2:, :-2],
                 padded[2:, 1:-1], padded[2:, 2:]]
            # A(P1): 0->1 transitions in the ordered neighborhood.
            transitions = sum(
                (p[k] == 0) & (p[(k + 1) % 8] == 1) for k in range(8)
            )
            # B(P1): number of non-zero neighbors, must be in [2, 6].
            neighbors = sum(p)
            if step == 0:
                cond = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                cond = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            doomed = img & (transitions == 1) & (neighbors >= 2) & (neighbors <= 6) & cond
            if doomed.any():
                img[doomed] = False
                changed = True
    return img


def find_skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Pixels with exactly one 8-connected skeleton neighbor."""
    skel = skeleton.astype(bool)
    padded = np.pad(skel, 1)
    counts = sum(
        padded[r : r + skel.shape[0], c : c + skel.shape[1]]
        for r in range(3)
        for c in range(3)
        if (r, c) != (1, 1)
    )
    rows, cols = np.nonzero(skel & (counts == 1))
    return list(zip(rows.tolist(), cols.tolist()))


def estimate_endpoint_orientation(
    skeleton: np.ndarray,
    endpoint: tuple[int, int],
    window_size: int = 15,
) -> tuple[float, float] | None:
    """Local (dy, dx) unit orientation at an endpoint, pointed away from the body.

    PCA on skeleton pixels in a square window (same math as
    gems.endpoint_extension.compute_structure_tensor_orientation, numpy only).
    Returns None when support is insufficient.
    """
    row, col = endpoint
    h, w = skeleton.shape
    r_min = max(0, row - window_size)
    r_max = min(h, row + window_size + 1)
    c_min = max(0, col - window_size)
    c_max = min(w, col + window_size + 1)
    window = skeleton[r_min:r_max, c_min:c_max]
    if window.sum() < 3:
        return None
    skel_rows, skel_cols = np.nonzero(window)
    if len(skel_rows) < 3:
        return None
    centered = np.vstack(
        [skel_rows - (row - r_min), skel_cols - (col - c_min)]
    ).astype(float)
    cov = np.cov(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    dy, dx = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.hypot(dy, dx))
    if norm < 1e-6:
        return None
    dy, dx = dy / norm, dx / norm
    # Orient away from the skeleton body: the pixels *behind* the endpoint
    # (opposite the direction) should be the body. If they are empty, the
    # direction currently points into the body, so flip it.
    behind = sum(
        skeleton[int(row - off * dy), int(col - off * dx)]
        for off in (1, 2, 3)
        if 0 <= int(row - off * dy) < h and 0 <= int(col - off * dx) < w
    )
    if behind == 0:
        dy, dx = -dy, -dx
    return float(dy), float(dx)


def grow_curved_path(
    prob: np.ndarray,
    start: tuple[int, int],
    direction: tuple[float, float],
    params: CurvedContinuationParams,
    valid_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Grow one curvature-prior continuation from a skeleton endpoint.

    Beam search over 8-neighborhood steps. Per-step cost is
    ``-log(p) + curvature_penalty * turn²``: the walk is pulled toward
    probability support and penalized for bending. Turns are limited to
    ±45° per step (the curvature prior); stepping onto ``p < min_prob``
    ends the path (unsupported bridges are never grown). Among accepted
    paths the *longest* (then cheapest) is returned — a continuation should
    reach as far as support allows, not stop at the cheapest two pixels.
    Returns [] when no path of at least ``min_path_length`` steps exists.
    """
    h, w = prob.shape
    if valid_mask is None:
        valid_mask = np.ones_like(prob, dtype=bool)
    dy, dx = direction
    heading = int(np.round(np.arctan2(dy, dx) / (np.pi / 4))) % 8
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w and valid_mask[sr, sc]):
        return []
    eps = 1e-6

    # Beam entries: (cumulative_cost, r, c, heading, path, visited-set)
    beam: list[tuple[float, int, int, int, list[tuple[int, int]], set[tuple[int, int]]]] = [
        (0.0, sr, sc, heading, [], {(sr, sc)})
    ]
    best_path: list[tuple[int, int]] = []
    best_cost = float("inf")
    for _depth in range(params.max_steps):
        nxt: list[tuple[float, int, int, int, list[tuple[int, int]], set[tuple[int, int]]]] = []
        for cost, r, c, hd, path, visited in beam:
            for turn in (-1, 0, 1):
                nh = (hd + turn) % 8
                dr, dc = _HEADINGS[nh]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if not valid_mask[nr, nc] or (nr, nc) in visited:
                    continue
                p = float(prob[nr, nc])
                if p < params.min_prob:
                    continue  # unsupported: do not grow into this pixel
                turn_rad = turn * (np.pi / 4)
                step_cost = -np.log(p + eps) + params.curvature_penalty * turn_rad**2
                new_path = path + [(nr, nc)]
                new_cost = cost + step_cost
                nxt.append((new_cost, nr, nc, nh, new_path, visited | {(nr, nc)}))
                if len(new_path) >= params.min_path_length and (
                    len(new_path) > len(best_path) or new_cost < best_cost
                ):
                    best_cost = new_cost
                    best_path = new_path
        if not nxt:
            break
        nxt.sort(key=lambda e: e[0])
        beam = nxt[: params.beam_width]
    return best_path


def apply_curved_continuation(
    probabilities: np.ndarray,
    *,
    threshold: float,
    params: CurvedContinuationParams,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Paint curvature-prior continuations onto a probability raster.

    Skeletonize at ``threshold`` (same per-fold LOFO thresholds the box uses
    for the straight-extension control), grow a curved path from each ranked
    endpoint through *weak support* (``min_prob <= p < threshold`` — signal the
    model saw but the skeleton threshold cut off), and paint accepted paths
    with decaying probability anchored at ``threshold``. Painted values never
    exceed ``threshold`` and never overwrite higher existing probability: the
    candidate promotes weak continuations to the skeleton level, it cannot
    erase the model's own predictions.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    prob = probabilities.astype(np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(prob)
    else:
        valid_mask = valid_mask.astype(bool)
    out = prob.copy()

    skeleton = _zhang_suen_thinning((prob >= threshold) & valid_mask)
    endpoints = find_skeleton_endpoints(skeleton)
    if not endpoints:
        return out

    # Budget: rank endpoints by local probability, keep the top max_paths.
    def local_mean(ep: tuple[int, int]) -> float:
        r, c = ep
        patch = prob[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
        return float(patch.mean())

    endpoints.sort(key=local_mean, reverse=True)
    endpoints = endpoints[: params.max_paths]

    for endpoint in endpoints:
        direction = estimate_endpoint_orientation(
            skeleton, endpoint, window_size=params.window_size
        )
        if direction is None:
            continue
        path = grow_curved_path(
            prob, endpoint, direction, params, valid_mask=valid_mask
        )
        if not path:
            continue
        # Penalize unsupported bridges: the finished path must carry support.
        mean_support = float(np.mean([prob[r, c] for r, c in path]))
        if mean_support < params.support_threshold:
            continue
        for k, (r, c) in enumerate(path):
            paint = threshold * float(np.exp(-params.decay_rate * k))
            out[r, c] = max(out[r, c], paint)
    return out


def diagnostics(
    probabilities: np.ndarray,
    *,
    threshold: float,
    params: CurvedContinuationParams,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """Small report for tests/QA: skeleton size, endpoints, paths accepted."""
    prob = probabilities.astype(np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(prob)
    skeleton = _zhang_suen_thinning((prob >= threshold) & valid_mask)
    endpoints = find_skeleton_endpoints(skeleton)
    return {
        "skeleton_pixels": int(skeleton.sum()),
        "endpoints": len(endpoints),
        "budget": params.max_paths,
        "max_steps": params.max_steps,
    }
