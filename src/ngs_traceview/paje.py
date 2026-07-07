"""Parser for Paje trace files as written by ngcore's PajeTrace (NGSolve).

Reads the whole file in a single streaming pass, batching the frequent
PushState/PopState events into numpy arrays. Push/pop matching is done
vectorized per (container, depth) level: within one container, pushes and
pops of the same nesting level strictly alternate in time, so after
computing the nesting depth with a cumulative sum they pair up elementwise.

Times in the file are in milliseconds (ngcore ConvertTime).
"""

import dataclasses
import multiprocessing as mp
import os
import time

import numpy as np

# Paje event ids as defined in the ngcore header
DEFINE_CONTAINER_TYPE = 0
DEFINE_VARIABLE_TYPE = 1
DEFINE_STATE_TYPE = 2
DEFINE_EVENT_TYPE = 3
DEFINE_LINK_TYPE = 4
DEFINE_ENTITY_VALUE = 5
CREATE_CONTAINER = 6
DESTROY_CONTAINER = 7
SET_VARIABLE = 8
ADD_VARIABLE = 9
SUB_VARIABLE = 10
SET_STATE = 11
PUSH_STATE = 12
POP_STATE = 13

_CHUNK = 1 << 20  # batch-convert push/pop columns every ~1M events


def _unquote(s: bytes) -> str:
    s = s.strip()
    if len(s) >= 2 and s[:1] == b'"' and s[-1:] == b'"':
        s = s[1:-1]
    return s.decode("utf-8", errors="replace")


@dataclasses.dataclass
class Container:
    alias: str
    type_alias: str
    parent: str | None
    name: str
    children: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Row:
    """One horizontal lane in the timeline display."""

    name: str
    container: str  # container alias
    kind: str  # container type name ("Thread", "Jobs", ...)
    max_depth: int = 1  # deepest nesting level present (>= 1)


@dataclasses.dataclass
class TraceData:
    # one entry per state interval, all rows combined,
    # sorted by (depth, start) so that nested states draw on top
    start: np.ndarray  # float64, ms
    end: np.ndarray  # float64, ms
    row: np.ndarray  # uint32, index into rows
    depth: np.ndarray  # uint8, 0-based nesting level
    value: np.ndarray  # uint32, index into names/colors

    names: list[str]  # entity value names
    colors: np.ndarray  # (n_names, 4) float32 rgba
    rows: list[Row]
    tmin: float
    tmax: float
    # variable curves keyed by "container name / variable name"
    variables: dict[str, tuple[np.ndarray, np.ndarray]]
    parse_time: float

    @property
    def n_intervals(self) -> int:
        return len(self.start)


class _PushPopAccumulator:
    """Collects push/pop event columns and batch-converts them to numpy."""

    def __init__(self):
        self._t: list[bytes] = []
        self._cont: list[bytes] = []
        self._val: list[bytes] = []  # b"" marks a pop
        self.t_parts: list[np.ndarray] = []
        self.cont_parts: list[np.ndarray] = []
        self.val_parts: list[np.ndarray] = []

    def push(self, t: bytes, cont: bytes, val: bytes):
        self._t.append(t)
        self._cont.append(cont)
        self._val.append(val)
        if len(self._t) >= _CHUNK:
            self._flush()

    def pop(self, t: bytes, cont: bytes):
        self.push(t, cont, b"")

    def _flush(self):
        if not self._t:
            return
        self.t_parts.append(np.asarray(self._t).astype(np.float64))
        self.cont_parts.append(np.asarray(self._cont))
        self.val_parts.append(np.asarray(self._val))
        self._t.clear()
        self._cont.clear()
        self._val.clear()

    def arrays(self):
        self._flush()
        if not self.t_parts:
            empty_s = np.asarray([], dtype="S1")
            return np.empty(0), empty_s, empty_s
        t = np.concatenate(self.t_parts)
        # concatenate with a common (max) itemsize
        cont = np.concatenate([p.astype(max(p.dtype for p in self.cont_parts)) for p in self.cont_parts])
        val = np.concatenate([p.astype(max(p.dtype for p in self.val_parts)) for p in self.val_parts])
        return t, cont, val


def _classify(line: bytes, acc: "_PushPopAccumulator", meta: list):
    """Route one raw line to the push/pop accumulator or the meta bucket."""
    code = line[: line.find(b"\t")]
    if code == b"12":
        p = line.split(b"\t")
        acc.push(p[1], p[3], p[4])
    elif code == b"13":
        p = line.split(b"\t")
        acc.pop(p[1], p[3].rstrip())
    elif line[:1] == b"%":
        return
    else:
        meta.append(line)


def _parse_range(args):
    """Worker: parse byte range [start, end) of the trace file.

    Runs in a forked child; touches only ``open()`` + numpy, so it is safe to
    fork from the (threaded) app server. Returns push/pop columns + meta lines.
    """
    path, start, end = args
    acc = _PushPopAccumulator()
    meta: list[bytes] = []
    with open(path, "rb") as f:
        if start:
            # discard the line straddling the start boundary (owned by the
            # previous chunk); seek to start-1 so an exact boundary consumes
            # only the trailing newline, never a whole valid line.
            f.seek(start - 1)
            f.readline()
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            _classify(line, acc, meta)
    return (*acc.arrays(), meta)


def _read_events_serial(path, progress):
    acc = _PushPopAccumulator()
    meta: list[bytes] = []
    with open(path, "rb") as f:
        for line in f:
            _classify(line, acc, meta)
    if progress:
        progress(0.8, "parsing events")
    return (*acc.arrays(), meta)


def _read_events_parallel(path, progress, workers):
    size = os.path.getsize(path)
    nchunks = workers
    bounds = [size * i // nchunks for i in range(nchunks + 1)]
    tasks = [(path, bounds[i], bounds[i + 1]) for i in range(nchunks)]
    results = [None] * nchunks

    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = mp.get_context("fork")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_parse_range, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if progress:
                progress(0.05 + 0.75 * done / nchunks,
                         f"parsing events · {done}/{nchunks} chunks")

    ts, cs, vs, meta = [], [], [], []
    for t, c, v, m in results:
        if len(t):
            ts.append(t)
            cs.append(c)
            vs.append(v)
        meta.extend(m)
    if not ts:
        empty = np.asarray([], dtype="S1")
        return np.empty(0), empty, empty, meta
    t = np.concatenate(ts)
    cw = max(a.dtype.itemsize for a in cs)
    vw = max(a.dtype.itemsize for a in vs)
    cont = np.concatenate([a.astype(f"S{cw}") for a in cs])
    val = np.concatenate([a.astype(f"S{vw}") for a in vs])
    return t, cont, val, meta


def _read_events(path, progress=None):
    """Read push/pop events + meta lines, in parallel when possible."""
    try:
        workers = min(os.cpu_count() or 1, 16)
    except Exception:
        workers = 1
    # only worth forking for reasonably large files
    if workers > 1 and os.path.getsize(path) > 8 * 1024 * 1024:
        try:
            return _read_events_parallel(path, progress, workers)
        except Exception as e:  # pragma: no cover - defensive fallback
            print(f"parallel parse failed ({e!r}); falling back to serial")
    return _read_events_serial(path, progress)


def _process_meta(meta_lines):
    """Turn the (few hundred) non-event lines into the entity/container tables."""
    entity_names: dict[bytes, str] = {}
    entity_colors: dict[bytes, tuple] = {}
    type_names: dict[bytes, str] = {}
    containers: dict[bytes, Container] = {}
    root_containers: list[Container] = []
    var_events: dict[tuple[bytes, bytes], list[tuple[float, float]]] = {}

    for line in meta_lines:
        p = [c.strip() for c in line.split(b"\t")]
        if not p or not p[0] or not p[0].isdigit():
            continue
        ev = int(p[0])
        if ev == DEFINE_ENTITY_VALUE:
            entity_names[p[1]] = _unquote(p[3])
            entity_colors[p[1]] = tuple(float(c) for c in _unquote(p[4]).split())
        elif ev == CREATE_CONTAINER:
            c = Container(
                alias=p[2].decode(),
                type_alias=p[3].decode(),
                parent=p[4].decode(),
                name=_unquote(p[5]),
            )
            containers[p[2]] = c
            parent = containers.get(p[4])
            (parent.children if parent is not None else root_containers).append(c)
        elif ev in (
            DEFINE_CONTAINER_TYPE,
            DEFINE_VARIABLE_TYPE,
            DEFINE_STATE_TYPE,
            DEFINE_EVENT_TYPE,
        ):
            type_names[p[1]] = _unquote(p[3])
        elif ev in (SET_VARIABLE, ADD_VARIABLE, SUB_VARIABLE):
            key = (p[3], p[2])
            t = float(p[1])
            v = float(p[4])
            lst = var_events.setdefault(key, [])
            if ev == SET_VARIABLE:
                lst.append((t, v))
            else:
                prev = lst[-1][1] if lst else 0.0
                lst.append((t, prev + v if ev == ADD_VARIABLE else prev - v))
        # DESTROY_CONTAINER / links / SET_STATE: not needed for the timeline
    return entity_names, entity_colors, type_names, containers, root_containers, var_events


def parse(path: str, progress=None) -> TraceData:
    """Parse a Paje trace. ``progress(fraction, message)`` is called if given."""
    t_begin = time.time()
    if progress:
        progress(0.02, "reading file")

    t_events, cont_col, val_col, meta_lines = _read_events(path, progress)

    if progress:
        progress(0.82, "building intervals")

    (
        entity_names,
        entity_colors,
        type_names,
        containers,
        root_containers,
        var_events,
    ) = _process_meta(meta_lines)

    n = len(t_events)
    is_push = val_col != b""

    # map container aliases and entity values to small integer ids
    cont_aliases, cont_idx = np.unique(cont_col, return_inverse=True)
    val_aliases, val_idx = np.unique(val_col, return_inverse=True)

    # entity table: known aliases from the header + any unaliased (quoted) values
    names: list[str] = []
    colors: list[tuple] = []
    val_map = np.zeros(len(val_aliases), dtype=np.uint32)
    for i, alias in enumerate(val_aliases):
        if alias == b"":
            continue  # pop marker, never used as a value
        if alias in entity_names:
            name = entity_names[alias]
            col = entity_colors.get(alias, (0.5, 0.5, 0.5))
        else:
            name = _unquote(alias)
            col = (0.5, 0.5, 0.5)
        val_map[i] = len(names)
        names.append(name)
        colors.append((*col[:3], 1.0))
    value_of_event = val_map[val_idx]

    # display rows: depth-first through the container tree in creation order,
    # keeping only containers that actually carry state events
    counts = np.bincount(cont_idx[is_push], minlength=len(cont_aliases))
    active = {a.decode() for a, c in zip(cont_aliases, counts) if c > 0}
    rows: list[Row] = []
    row_of_alias: dict[str, int] = {}

    def _walk(container: Container):
        if container.alias in active:
            row_of_alias[container.alias] = len(rows)
            rows.append(
                Row(
                    name=container.name,
                    container=container.alias,
                    kind=type_names.get(container.type_alias.encode(), ""),
                )
            )
        for child in container.children:
            _walk(child)

    for c in root_containers:
        _walk(c)

    row_map = np.full(len(cont_aliases), -1, dtype=np.int64)
    for i, alias in enumerate(cont_aliases):
        row_map[i] = row_of_alias.get(alias.decode(), -1)
    row_of_event = row_map[cont_idx]

    tmax = float(t_events.max()) if n else 0.0
    tmin = min(0.0, float(t_events.min())) if n else 0.0

    # vectorized push/pop pairing per (container, depth)
    kind = np.where(is_push, np.int64(1), np.int64(-1))
    starts_l, ends_l, rows_l, depths_l, values_l = [], [], [], [], []
    for ci in range(len(cont_aliases)):
        r = row_map[ci]
        if r < 0:
            continue
        sel = np.flatnonzero(cont_idx == ci)
        k = kind[sel]
        d = np.cumsum(k)  # depth after the event; a push to level L gives d == L
        if d.min() < 0:
            raise ValueError(f"unbalanced pop in container {cont_aliases[ci]!r}")
        t_sel = t_events[sel]
        v_sel = value_of_event[sel]
        for level in range(1, int(d.max()) + 1):
            push_i = np.flatnonzero((k == 1) & (d == level))
            pop_i = np.flatnonzero((k == -1) & (d == level - 1))
            n_pairs = len(push_i)
            if len(pop_i) < n_pairs:  # unclosed states at trace end
                pop_t = np.concatenate(
                    [t_sel[pop_i], np.full(n_pairs - len(pop_i), tmax)]
                )
            else:
                pop_t = t_sel[pop_i]
            starts_l.append(t_sel[push_i])
            ends_l.append(pop_t)
            rows_l.append(np.full(n_pairs, r, dtype=np.uint32))
            depths_l.append(np.full(n_pairs, level - 1, dtype=np.uint8))
            values_l.append(v_sel[push_i])
            rows[r].max_depth = max(rows[r].max_depth, level)

    if starts_l:
        start = np.concatenate(starts_l)
        end = np.concatenate(ends_l)
        row = np.concatenate(rows_l)
        depth = np.concatenate(depths_l)
        value = np.concatenate(values_l).astype(np.uint32)
    else:
        start = end = np.empty(0)
        row = value = np.empty(0, dtype=np.uint32)
        depth = np.empty(0, dtype=np.uint8)

    if np.any(end < start):
        bad = int(np.sum(end < start))
        raise ValueError(f"{bad} intervals with negative duration — unsorted trace?")

    # draw order: shallow first, so nested states paint on top
    order = np.lexsort((start, depth))
    start, end, row, depth, value = (
        start[order],
        end[order],
        row[order],
        depth[order],
        value[order],
    )

    variables = {}
    for (cont, var_type), events in var_events.items():
        cname = containers[cont].name if cont in containers else cont.decode()
        vname = type_names.get(var_type, var_type.decode())
        arr = np.asarray(events)
        variables[f"{cname} / {vname}"] = (arr[:, 0], arr[:, 1])

    return TraceData(
        start=start,
        end=end,
        row=row,
        depth=depth,
        value=value,
        names=names,
        colors=np.asarray(colors, dtype=np.float32).reshape(-1, 4),
        rows=rows,
        tmin=tmin,
        tmax=tmax,
        variables=variables,
        parse_time=time.time() - t_begin,
    )
