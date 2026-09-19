"""Parser for Paje trace files as written by ngcore's PajeTrace (NGSolve).

Reads the whole file in a single streaming pass, batching the frequent
PushState/PopState events into numpy arrays. Push/pop matching is done
vectorized per (container, depth) level: within one container, pushes and
pops of the same nesting level strictly alternate in time, so after
computing the nesting depth with a cumulative sum they pair up elementwise.

Times in the file are in milliseconds (ngcore ConvertTime).
"""

import colorsys
import dataclasses
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ASCII byte constants used by the vectorized reader
_NL = 10  # \n
_TAB = 9  # \t
_C1 = ord("1")
_C2 = ord("2")
_C3 = ord("3")
_C0 = ord("0")
_PCT = ord("%")


def _unquote(s: bytes) -> str:
    s = s.strip()
    if len(s) >= 2 and s[:1] == b'"' and s[-1:] == b'"':
        s = s[1:-1]
    return s.decode("utf-8", errors="replace")


def _auto_color(name: str) -> tuple[float, float, float]:
    """Deterministic, well-spread color for values that carry no defined color.

    Hues come from a stable hash of the name (hashlib, not the salted built-in
    hash) so a given state gets the same color across runs and sessions.
    """
    h = int.from_bytes(hashlib.md5(name.encode("utf-8")).digest()[:4], "big")
    hue = (h % 1000) / 1000.0
    return colorsys.hsv_to_rgb(hue, 0.55, 0.9)


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
class MemoryKind:
    """Allocations of one memory kind (host or device), one entry per alloc."""

    t_alloc: np.ndarray  # float64, ms
    t_free: np.ndarray  # float64, ms; inf when never freed inside the trace
    size: np.ndarray  # float64, bytes
    stack: np.ndarray  # int64, timer-stack id (index into MemoryData.stack_*)
    # frees without a matching alloc (memory from before the trace started)
    free_t: np.ndarray
    free_size: np.ndarray
    # allocated bytes over time as a step function (relative to trace start)
    curve_t: np.ndarray
    curve_y: np.ndarray
    row: int  # timeline row index

    @property
    def n_alloc(self) -> int:
        return len(self.t_alloc)

    def value_at(self, t: float) -> float:
        i = int(np.searchsorted(self.curve_t, t, side="right")) - 1
        return float(self.curve_y[i]) if i >= 0 else 0.0

    def peak(self) -> tuple[float, float]:
        if len(self.curve_y) == 0:
            return 0.0, 0.0
        i = int(np.argmax(self.curve_y))
        return float(self.curve_t[i]), float(self.curve_y[i])


@dataclasses.dataclass
class MemoryData:
    """Memory events attributed to timer stacks.

    Stacks form a tree (calling-context tree): node 0 is the empty stack, every
    other node is (parent, timer value id); parents always have smaller ids.
    """

    stack_parent: np.ndarray  # int64, -1 for the root
    stack_value: np.ndarray  # int64, index into TraceData.names (-1 for the root)
    stack_depth: np.ndarray  # int64, 0 for the root
    host: MemoryKind
    device: MemoryKind
    n_events: int

    @property
    def n_stacks(self) -> int:
        return len(self.stack_parent)

    def kind(self, name: str) -> MemoryKind:
        return self.host if name == "host" else self.device

    def kind_of_row(self, row: int) -> str | None:
        if row == self.host.row:
            return "host"
        if row == self.device.row:
            return "device"
        return None

    def path(self, sid: int) -> list[int]:
        """Stack ids from the root's first child down to ``sid`` (excludes root)."""
        out = []
        while sid > 0:
            out.append(sid)
            sid = int(self.stack_parent[sid])
        return out[::-1]


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
    memory: MemoryData | None = None

    @property
    def n_intervals(self) -> int:
        return len(self.start)


def _gather(sub, starts, ends):
    """Extract the variable-length byte ranges ``sub[starts[i]:ends[i]]`` into a
    fixed-width ``S`` array (null-padded). Pure numpy, so it releases the GIL and
    parallelizes across threads."""
    lens = ends - starts
    lens[lens < 0] = 0
    W = int(lens.max()) if len(lens) else 1
    W = max(W, 1)
    j = np.arange(W)
    src = starts[:, None] + j
    valid = j < lens[:, None]
    g = np.where(valid, sub[np.clip(src, 0, len(sub) - 1)], 0).astype(np.uint8)
    return np.ascontiguousarray(g).view(f"S{W}").ravel()


def _parse_hex(sub, starts, ends):
    """Vectorized parse of ``0x``-prefixed hex fields into uint64."""
    n = len(starts)
    out = np.zeros(n, dtype=np.uint64)
    if n == 0:
        return out
    m = len(sub)
    has_pre = (sub[np.minimum(starts, m - 1)] == ord("0")) & (
        (sub[np.minimum(starts + 1, m - 1)] | 32) == ord("x")
    )
    starts = starts + 2 * has_pre
    lens = np.clip(ends - starts, 0, 16)
    W = int(lens.max()) if n else 1
    for j in range(max(W, 1)):
        c = sub[np.clip(starts + j, 0, m - 1)].astype(np.int64)
        nib = np.where(c >= ord("a"), c - ord("a") + 10,
                       np.where(c >= ord("A"), c - ord("A") + 10, c - ord("0")))
        nib = np.clip(nib, 0, 15).astype(np.uint64)
        use = j < lens
        out = np.where(use, out * np.uint64(16) + nib, out)
    return out


def _extract_chunk(buf, start, end):
    """Vectorized parse of a newline-aligned byte range into push/pop columns.

    Returns ``(time float64, container S, value S, offset int64, meta_lines,
    mem)`` for the chunk. Value is ``b""`` for pop events (matches the pairing
    code's push detection); offset is the line's byte position in the file (a
    tiebreaker that preserves file order). ``mem`` holds the memory events
    (``20`` lines) as a dict of columns, or None when the chunk has none.
    """
    sub = buf[start:end]
    m = len(sub)
    empty = np.asarray([], dtype="S1")
    none = (np.empty(0), empty, empty, np.empty(0, np.int64))
    if m == 0:
        return (*none, [], None)

    nl = np.flatnonzero(sub == _NL)
    ls = np.empty(len(nl) + 1, np.int64)
    ls[0] = 0
    ls[1:] = nl + 1
    if ls[-1] >= m:
        ls = ls[:-1]
    L = len(ls)
    line_end = np.empty(L, np.int64)
    line_end[: len(nl)] = nl
    if L > len(nl):
        line_end[-1] = m  # last line without a trailing newline

    c0 = sub[ls]
    c1 = sub[np.minimum(ls + 1, m - 1)]
    c2 = sub[np.minimum(ls + 2, m - 1)]
    is_pp = (c0 == _C1) & ((c1 == _C2) | (c1 == _C3)) & (c2 == _TAB)
    is_mem = (c0 == _C2) & (c1 == _C0) & (c2 == _TAB)

    # the few non-event, non-comment lines (defines / creates / variables)
    meta_idx = np.flatnonzero((~is_pp) & (~is_mem) & (c0 != _PCT))
    meta = [sub[ls[k] : line_end[k]].tobytes() for k in meta_idx]

    tabs = np.flatnonzero(sub == _TAB)
    ntab = len(tabs)
    first_tab = np.searchsorted(tabs, ls)
    BIG = np.iinfo(np.int64).max

    mem = None
    msel = np.flatnonzero(is_mem)
    if len(msel):
        ft = first_tab[msel]
        end_s = line_end[msel]
        # 20 <t> M <container> <kind> <bytes> 0x<addr>: six tabs before the newline
        ok = (ft + 5 < ntab) & (tabs[np.minimum(ft + 5, ntab - 1)] < end_s)
        if not ok.all():
            keep = np.flatnonzero(ok)
            msel, ft, end_s = msel[keep], ft[keep], end_s[keep]
        if len(msel):
            t0, t1, t2 = tabs[ft], tabs[ft + 1], tabs[ft + 2]
            t3, t4, t5 = tabs[ft + 3], tabs[ft + 4], tabs[ft + 5]
            mem = {
                "t": _gather(sub, t0 + 1, t1).astype(np.float64),
                "cont": _gather(sub, t2 + 1, t3),
                "kind": _gather(sub, t3 + 1, t4),
                "size": _gather(sub, t4 + 1, t5).astype(np.float64),
                "addr": _parse_hex(sub, t5 + 1, end_s),
                "off": ls[msel] + start,
            }

    sel = np.flatnonzero(is_pp)
    if len(sel) == 0:
        return (*none, meta, mem)

    ft = first_tab[sel]  # index of each line's first tab
    end_s = line_end[sel]
    push = c1[sel] == _C2

    # drop incomplete trailing lines (need at least 3 tabs before the newline —
    # e.g. a byte-truncated last line); complete files keep every event
    ok = (ft + 2 < ntab) & (tabs[np.minimum(ft + 2, ntab - 1)] < end_s)
    if not ok.all():
        keep = np.flatnonzero(ok)
        sel, ft, end_s, push = sel[keep], ft[keep], end_s[keep], push[keep]
        if len(sel) == 0:
            return (*none, meta, mem)

    t0 = tabs[ft]
    t1 = tabs[ft + 1]
    t2 = tabs[ft + 2]
    t3 = tabs[np.minimum(ft + 3, ntab - 1)]
    # value's closing tab; when the line has no 5th tab the value runs to the
    # newline, so use a sentinel that min()s down to end_s
    t4 = np.where(ft + 4 < ntab, tabs[np.minimum(ft + 4, ntab - 1)], BIG)

    # time = field 1 (tab0..tab1); container = field 3 (tab2..tab3 for push, else
    # to newline); value = field 4 (push only, tab3..tab4-or-newline)
    tvals = _gather(sub, t0 + 1, t1).astype(np.float64)
    cont = _gather(sub, t2 + 1, np.where(push, t3, end_s))
    vstart = np.where(push, t3 + 1, end_s)
    vend = np.where(push, np.minimum(t4, end_s), end_s)
    val = _gather(sub, vstart, vend)
    return tvals, cont, val, ls[sel] + start, meta, mem


def _newline_bounds(data, n, chunks):
    """Byte offsets partitioning ``data`` into ``chunks`` newline-aligned pieces."""
    bounds = [0]
    for i in range(1, chunks):
        p = data.find(b"\n", n * i // chunks)
        bounds.append(p + 1 if p >= 0 else n)
    bounds.append(n)
    out = [0]
    for b in bounds[1:]:
        if b > out[-1]:
            out.append(b)
    return out


def _read_events(path, progress=None):
    """Read push/pop columns + meta lines with a vectorized numpy parser.

    The file is split into newline-aligned byte ranges parsed by
    :func:`_extract_chunk`; the numpy ops release the GIL, so a plain
    ``ThreadPoolExecutor`` parallelizes them — no worker processes, so nothing
    inherits the app's websocket/event-loop state (which broke fork on macOS).
    """
    if progress:
        progress(0.05, "reading file")
    with open(path, "rb") as f:
        data = f.read()
    buf = np.frombuffer(data, dtype=np.uint8)
    n = len(data)

    try:
        workers = min(os.cpu_count() or 1, 8)
    except Exception:
        workers = 1
    if n < 4 * 1024 * 1024 or sys.platform == "emscripten":  # no threads in pyodide
        workers = 1

    bounds = _newline_bounds(data, n, workers)
    total = len(bounds) - 1
    results = [None] * total

    if total == 1:
        results[0] = _extract_chunk(buf, bounds[0], bounds[1])
        if progress:
            progress(0.8, "parsing events")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_extract_chunk, buf, bounds[i], bounds[i + 1]): i
                for i in range(total)
            }
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
                done += 1
                if progress:
                    progress(0.1 + 0.7 * done / total,
                             f"parsing events · {done}/{total} chunks")

    ts, cs, vs, os_, meta, mems = [], [], [], [], [], []
    for tv, cont, val, off, m, mem in results:  # in file order
        if len(tv):
            ts.append(tv)
            cs.append(cont)
            vs.append(val)
            os_.append(off)
        meta.extend(m)
        if mem is not None:
            mems.append(mem)
    mem = _concat_mem(mems)
    if not ts:
        empty = np.asarray([], dtype="S1")
        return np.empty(0), empty, empty, np.empty(0, np.int64), meta, mem
    t = np.concatenate(ts)
    cont = _concat_s(cs)
    val = _concat_s(vs)
    off = np.concatenate(os_)
    return t, cont, val, off, meta, mem


def _concat_s(arrs):
    w = max(a.dtype.itemsize for a in arrs)
    return np.concatenate([a.astype(f"S{w}") for a in arrs])


def _concat_mem(mems):
    if not mems:
        return None
    out = {}
    for key in mems[0]:
        cols = [m[key] for m in mems]
        out[key] = _concat_s(cols) if cols[0].dtype.kind == "S" else np.concatenate(cols)
    return out


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

    t_events, cont_col, val_col, off_col, meta_lines, mem_raw = _read_events(path, progress)

    # The push/pop pairing below assumes events are in chronological order, but
    # some traces interleave events out of time order (e.g. multiple worker
    # threads flushing buffers). Stable-sort by time so the cumsum depth reflects
    # real nesting; a stable sort keeps file order among equal-timestamp events
    # (so a pop preceding a same-time push stays before it).
    if len(t_events) and not np.all(np.diff(t_events) >= 0):
        order = np.argsort(t_events, kind="stable")
        t_events, cont_col, val_col, off_col = (
            t_events[order], cont_col[order], val_col[order], off_col[order]
        )

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
            col = entity_colors.get(alias) or _auto_color(name)
        else:
            # inline (quoted) value with no DEFINE_ENTITY_VALUE — most GPU/state
            # events arrive this way; give each distinct name its own color
            name = _unquote(alias)
            col = _auto_color(name)
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
    has_mem = mem_raw is not None
    if has_mem:  # memory curves take the two top rows
        rows.append(Row(name="memory host", container="", kind="memory"))
        rows.append(Row(name="memory device", container="", kind="memory"))

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
    if has_mem and len(mem_raw["t"]):
        tmax = max(tmax, float(mem_raw["t"].max()))
        tmin = min(tmin, float(mem_raw["t"].min()))

    # timer-stack tree for memory attribution: node 0 = no timer active, every
    # other node is (parent node, timer value) interned via a flat integer key
    stack_key: dict[int, int] = {}
    stack_parent = [-1]
    stack_value = [-1]
    n_names = max(len(names), 1)
    if has_mem:
        m_t, m_off = mem_raw["t"], mem_raw["off"]
        w = max(cont_aliases.dtype.itemsize, mem_raw["cont"].dtype.itemsize)
        ca = cont_aliases.astype(f"S{w}")
        mc = mem_raw["cont"].astype(f"S{w}")
        m_ci = np.minimum(np.searchsorted(ca, mc), max(len(ca) - 1, 0))
        m_ci = np.where(ca[m_ci] == mc, m_ci, -1) if len(ca) else np.full(len(mc), -1)
        mem_sid = np.zeros(len(m_t), np.int64)
        # ngcore records timers in "Timer level N" containers (one nesting
        # level each) and only tasks on the thread containers; the stack of a
        # memory event is the timer levels' states followed by its thread's own
        timer_ci = []
        for ci, alias in enumerate(cont_aliases):
            c = containers.get(alias)
            if c is not None and c.name.startswith("Timer level "):
                try:
                    timer_ci.append((int(c.name[12:]), ci))
                except ValueError:
                    pass
        timer_ci = [ci for _, ci in sorted(timer_ci)]
        timer_state = {}  # ci -> (t, off, sid_after) for the timer level containers

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
        if has_mem:
            sid_after = np.zeros(len(sel), np.int64)  # stack id after each event
            prev_push = None
        for level in range(1, int(d.max()) + 1):
            push_i = np.flatnonzero((k == 1) & (d == level))
            pop_i = np.flatnonzero((k == -1) & (d == level - 1))
            if has_mem:
                if level == 1:
                    parent_sid = np.zeros(len(push_i), np.int64)
                else:
                    pp = prev_push[np.searchsorted(prev_push, push_i, side="right") - 1]
                    parent_sid = sid_after[pp]
                key = parent_sid * n_names + v_sel[push_i].astype(np.int64)
                uniq, inv = np.unique(key, return_inverse=True)
                ids = np.empty(len(uniq), np.int64)
                for j, kk in enumerate(uniq.tolist()):
                    sid = stack_key.get(kk)
                    if sid is None:
                        sid = len(stack_parent)
                        stack_key[kk] = sid
                        stack_parent.append(kk // n_names)
                        stack_value.append(kk % n_names)
                    ids[j] = sid
                sid_after[push_i] = ids[inv]
                # a pop back to this level continues the enclosing push's stack
                back_i = np.flatnonzero((k == -1) & (d == level))
                if len(back_i) and len(push_i):
                    sid_after[back_i] = sid_after[
                        push_i[np.searchsorted(push_i, back_i, side="right") - 1]
                    ]
                prev_push = push_i
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
            # depth is packed into 8 bits of the shader flags, so clamp at 255
            depths_l.append(np.full(n_pairs, min(level - 1, 255), dtype=np.uint8))
            values_l.append(v_sel[push_i])
            rows[r].max_depth = max(rows[r].max_depth, level)
        if has_mem:
            msel_c = np.flatnonzero(m_ci == ci)
            if len(msel_c):
                mem_sid[msel_c] = _stack_at(
                    t_sel, off_col[sel], sid_after, m_t[msel_c], m_off[msel_c]
                )
            if ci in timer_ci:
                timer_state[ci] = (t_sel, off_col[sel], sid_after)

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

    memory = None
    if has_mem:
        if progress:
            progress(0.9, "matching memory events")
        if timer_state:
            cols = []
            for ci in timer_ci:
                if ci in timer_state:
                    t_s, off_s, sid_s = timer_state[ci]
                    cols.append(_stack_at(t_s, off_s, sid_s, m_t, m_off))
            cols.append(mem_sid)
            mem_sid = _compose_stacks(stack_key, stack_parent, stack_value, n_names, cols)
        memory = _build_memory(mem_raw, mem_sid, stack_parent, stack_value)

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
        memory=memory,
    )


def _stack_at(t_s, off_s, sid_after, t_m, off_m):
    """Stack id active at each memory event: merge the container's state
    events and memory events by (time, file offset) and take the stack after
    the most recent state event."""
    n_s = len(t_s)
    order = np.lexsort((np.concatenate((off_s, off_m)), np.concatenate((t_s, t_m))))
    is_state = order < n_s
    last = np.maximum.accumulate(np.where(is_state, np.arange(len(order)), -1))
    mpos = np.flatnonzero(~is_state)
    ls = last[mpos]
    sid_merged = np.concatenate((sid_after, np.zeros(len(t_m), np.int64)))
    sid = np.where(ls >= 0, sid_merged[order[np.maximum(ls, 0)]], 0)
    out = np.empty(len(t_m), np.int64)
    out[order[mpos] - n_s] = sid
    return out


def _path_values(parent, value, sids):
    """(n, max depth) matrix of timer values along each stack, outermost
    first, padded with -1."""
    depth = np.zeros(len(parent), np.int64)
    for i in range(1, len(parent)):
        depth[i] = depth[parent[i]] + 1
    maxd = int(depth[sids].max()) if len(sids) else 0
    out = np.full((len(sids), maxd), -1, np.int64)
    cur = sids.copy()
    for j in range(maxd):
        pos = depth[sids] - 1 - j
        ok = pos >= 0
        out[np.flatnonzero(ok), pos[ok]] = value[cur[ok]]
        cur = np.where(ok, parent[np.maximum(cur, 0)], cur)
    return out


def _compose_stacks(stack_key, stack_parent, stack_value, n_names, cols):
    """Chain several per-container stacks into one path per event and intern
    it in the shared tree: the timer levels first, the thread's own states last."""
    parent = np.asarray(stack_parent, np.int64)
    value = np.asarray(stack_value, np.int64)
    values = [_path_values(parent, value, np.asarray(c, np.int64)) for c in cols]
    n = len(cols[0])
    sid = np.zeros(n, np.int64)
    for mat in values:
        for j in range(mat.shape[1]):
            v = mat[:, j]
            ok = np.flatnonzero(v >= 0)
            if len(ok) == 0:
                continue
            key = sid[ok] * n_names + v[ok]
            uniq, inv = np.unique(key, return_inverse=True)
            ids = np.empty(len(uniq), np.int64)
            for i, kk in enumerate(uniq.tolist()):
                s = stack_key.get(kk)
                if s is None:
                    s = len(stack_parent)
                    stack_key[kk] = s
                    stack_parent.append(kk // n_names)
                    stack_value.append(kk % n_names)
                ids[i] = s
            sid[ok] = ids[inv]
    return sid


def _match_kind(t, off, size, addr, sid, is_alloc, row):
    """Pair frees with allocs by address in time order and build the curve."""
    n = len(t)
    order = np.lexsort((off, t, addr))  # per address in time (then file) order
    a_o, al_o = addr[order], is_alloc[order]
    matched = np.zeros(n, dtype=bool)  # free directly preceded by an alloc of its address
    if n > 1:
        matched[1:] = (a_o[1:] == a_o[:-1]) & al_o[:-1] & ~al_o[1:]
    apos = np.flatnonzero(al_o)
    ai = order[apos]
    t_alloc, sz, st = t[ai], size[ai], sid[ai]
    t_free = np.full(len(ai), np.inf)
    rank = np.cumsum(al_o) - 1  # alloc ordinal at each sorted position
    fpos = np.flatnonzero(matched)
    t_free[rank[fpos - 1]] = t[order[fpos]]
    upos = np.flatnonzero(~al_o & ~matched)
    free_t, free_size = t[order[upos]], size[order[upos]]

    o = np.argsort(t_alloc, kind="stable")
    t_alloc, t_free, sz, st = t_alloc[o], t_free[o], sz[o], st[o]

    fin = np.isfinite(t_free)
    ct = np.concatenate((t_alloc, t_free[fin], free_t))
    dy = np.concatenate((sz, -sz[fin], -free_size))
    later = np.concatenate((np.zeros(len(t_alloc)), np.ones(fin.sum() + len(free_t))))
    o = np.lexsort((later, ct))  # allocs before frees at equal times
    return MemoryKind(
        t_alloc=t_alloc, t_free=t_free, size=sz, stack=st,
        free_t=free_t, free_size=free_size,
        curve_t=ct[o], curve_y=np.cumsum(dy[o]), row=row,
    )


def _build_memory(mem_raw, mem_sid, stack_parent, stack_value):
    kind = mem_raw["kind"]
    is_host = (kind == b"ha") | (kind == b"hf")
    is_alloc = (kind == b"ha") | (kind == b"da")
    kinds = []
    for row, mask in ((0, is_host), (1, ~is_host)):
        kinds.append(
            _match_kind(
                mem_raw["t"][mask], mem_raw["off"][mask], mem_raw["size"][mask],
                mem_raw["addr"][mask], mem_sid[mask], is_alloc[mask], row,
            )
        )
    parent = np.asarray(stack_parent, dtype=np.int64)
    depth = np.zeros(len(parent), np.int64)
    for i in range(1, len(parent)):  # parents precede children
        depth[i] = depth[parent[i]] + 1
    return MemoryData(
        stack_parent=parent,
        stack_value=np.asarray(stack_value, dtype=np.int64),
        stack_depth=depth,
        host=kinds[0],
        device=kinds[1],
        n_events=len(kind),
    )
