"""Per-function aggregate statistics over a trace (ViTE-like)."""

import dataclasses

import numpy as np

from .paje import TraceData


@dataclasses.dataclass
class FunctionStat:
    value: int  # entity-value id (index into trace.names / colors)
    name: str
    color: tuple  # (r, g, b) 0..1
    count: int
    total: float  # ms
    mean: float  # ms
    min: float  # ms
    max: float  # ms
    percent: float  # of the aggregated time span


def compute(trace: TraceData, t0: float | None = None, t1: float | None = None):
    """Aggregate interval statistics per function.

    With ``t0``/``t1`` given, only the part of each interval inside the
    ``[t0, t1]`` window contributes (intervals are clipped to the window),
    matching ViTE's "statistics of the current selection".
    """
    n = len(trace.names)
    value = trace.value
    start, end = trace.start, trace.end

    if t0 is not None and t1 is not None:
        mask = (end > t0) & (start < t1)
        value = value[mask]
        dur = np.minimum(end[mask], t1) - np.maximum(start[mask], t0)
        span = max(t1 - t0, 1e-12)
    else:
        dur = end - start
        span = max(trace.tmax - trace.tmin, 1e-12)

    count = np.bincount(value, minlength=n).astype(np.int64)
    total = np.bincount(value, weights=dur, minlength=n)
    vmax = np.zeros(n)
    np.maximum.at(vmax, value, dur)
    vmin = np.full(n, np.inf)
    np.minimum.at(vmin, value, dur)

    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    for v in range(n):
        if count[v] == 0:
            continue
        out.append(
            FunctionStat(
                value=v,
                name=trace.names[v],
                color=tuple(float(c) for c in trace.colors[v, :3]),
                count=int(count[v]),
                total=float(total[v]),
                mean=float(mean[v]),
                min=float(vmin[v]),
                max=float(vmax[v]),
                percent=100.0 * float(total[v]) / span,
            )
        )
    out.sort(key=lambda s: s.total, reverse=True)
    return out
