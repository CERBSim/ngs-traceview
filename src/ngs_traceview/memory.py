"""Aggregation of memory allocations over timer stacks (sunburst data)."""

import dataclasses

import numpy as np

from .paje import MemoryData, MemoryKind


def format_bytes(b: float) -> str:
    sign = "-" if b < 0 else ""
    b = abs(float(b))
    for factor, unit in ((1024.0**4, "TB"), (1024.0**3, "GB"), (1024.0**2, "MB"), (1024.0, "KB")):
        if b >= factor:
            return f"{sign}{b / factor:.3g} {unit}"
    return f"{sign}{b:.0f} B"


def alive_mask(k: MemoryKind, t: float) -> np.ndarray:
    """Allocations alive at time t."""
    return (k.t_alloc <= t) & (k.t_free > t)


def growth_mask(k: MemoryKind, t0: float, t1: float) -> np.ndarray:
    """Allocations made inside [t0, t1] that are still alive at t1."""
    return (k.t_alloc >= t0) & (k.t_alloc <= t1) & (k.t_free > t1)


def self_bytes(mem: MemoryData, k: MemoryKind, mask: np.ndarray) -> np.ndarray:
    """Bytes per stack node allocated directly in that stack."""
    return np.bincount(k.stack[mask], weights=k.size[mask], minlength=mem.n_stacks)


def inclusive_bytes(mem: MemoryData, own: np.ndarray) -> np.ndarray:
    """Bytes per stack node including all deeper stacks."""
    incl = own.astype(np.float64).copy()
    depth = mem.stack_depth
    for d in range(int(depth.max()) if len(depth) else 0, 0, -1):
        sel = np.flatnonzero(depth == d)
        np.add.at(incl, mem.stack_parent[sel], incl[sel])
    return incl


def children_lists(mem: MemoryData) -> tuple[np.ndarray, np.ndarray]:
    """CSR children lists: ``order[start[p]:start[p+1]]`` are the children of p."""
    order = np.argsort(mem.stack_parent[1:], kind="stable") + 1
    counts = np.bincount(mem.stack_parent[1:], minlength=mem.n_stacks)
    start = np.zeros(mem.n_stacks + 1, np.int64)
    start[1:] = np.cumsum(counts)
    return order, start


@dataclasses.dataclass
class Arc:
    sid: int  # stack id, or the parent's id for an "other" arc
    level: int  # ring index, 1 = children of the focus node
    a0: float  # angle range in radians, clockwise from 12 o'clock
    a1: float
    value: float  # inclusive bytes
    other: int = 0  # > 0: folded arc, number of small siblings it stands for
    own: bool = False  # bytes allocated directly in sid, outside its child timers


def sunburst(
    mem: MemoryData,
    incl: np.ndarray,
    focus: int = 0,
    rings: int = 5,
    min_frac: float = 0.004,
    max_children: int = 40,
) -> list[Arc]:
    """Arcs of the sunburst rooted at ``focus``; angles are proportional to
    inclusive bytes. Siblings below ``min_frac`` of the focus total, or past
    ``max_children``, fold into one grey "other" arc per parent. The bytes a
    node allocates outside its child timers get an ``own`` arc after its
    children, so every ring below a node is filled (for the focus always)."""
    order, start = children_lists(mem)
    total = float(incl[focus])
    arcs: list[Arc] = []
    if total <= 0:
        return arcs
    two_pi = 2 * np.pi

    def walk(node, level, a0, a1):
        kids = order[start[node] : start[node + 1]]
        kids = kids[incl[kids] > 0]
        own = float(incl[node]) - float(incl[kids].sum())
        own = own if own > 0.5 else 0.0
        if len(kids) == 0:
            if node == focus and own:
                arcs.append(Arc(int(node), level, a0, a1, own, own=True))
            return
        kids = kids[np.argsort(-incl[kids], kind="stable")]
        span = a1 - a0
        scale = span / float(incl[node])
        small = incl[kids] < min_frac * total
        small[max_children:] = True
        if small.sum() < 2:  # folding a single child hides nothing
            small[:] = False
        a = a0
        for sid in kids[~small]:
            v = float(incl[sid])
            arcs.append(Arc(int(sid), level, a, a + v * scale, v))
            if level < rings:
                walk(int(sid), level + 1, a, a + v * scale)
            a += v * scale
        if small.any():
            v = float(incl[kids[small]].sum())
            arcs.append(Arc(int(node), level, a, a + v * scale, v, other=int(small.sum())))
            a += v * scale
        if own:
            arcs.append(Arc(int(node), level, a, a1, own, own=True))

    walk(int(focus), 1, 0.0, two_pi)
    return arcs
