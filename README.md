# ngs-traceview

A ViTE-like viewer for Paje trace files written by NGSolve's `ngcore::PajeTrace`
(task manager / timer traces), built with [ngapp](https://github.com/CERBSim/ngapp)
and the CERBSim `webgpu` framework. Import package: `ngs_traceview`.

## Usage

```bash
pip install -e .
ngs-traceview mytrace.trace              # open a local file directly
ngs-traceview                            # open, then use the file picker
python -m ngs_traceview mytrace.trace    # equivalent (module form)
```

### Synthetic memory trace (for testing)

```bash
python tests/make_memtrace.py /tmp/mem.trace            # small, hand-checked scenario
python tests/make_memtrace.py /tmp/mem.trace --big 2000000
ngs-traceview /tmp/mem.trace
```

### In a Jupyter notebook

```python
from ngs_traceview import ShowTrace
ShowTrace("mytrace.trace")               # embeds the full viewer in the cell
```

### Controls

| input | action |
| --- | --- |
| left drag | draw a box → zoom to that time range |
| right click | step back to the previous view (undo a zoom/pan) |
| middle-drag / shift-drag / two-finger swipe | pan |
| mouse wheel | zoom time axis (at cursor) |
| ctrl + wheel | zoom rows (at cursor) |
| hover | tooltip with task name and duration |
| click | show the task's info in the bottom bar |
| double click | highlight that function, dim everything else |

### Memory rows

Traces that contain memory events (`PajeNewEvent 20`, kinds `ha`/`hf`/`da`/`df`)
get two extra rows at the top, *memory host* and *memory device*, showing the
allocated bytes over time relative to the trace start (the curve dips below zero
when memory allocated before the trace is freed). Each allocation is attributed
to the stack of timers active on its thread at that moment; frees are matched to
allocations by address.

| input (on a memory row) | action |
| --- | --- |
| hover | tooltip with time and allocated bytes; crosshair |
| click | open the memory panel: sunburst of the memory alive at that time, grouped by timer stack (root = all, ring 1 = outermost timer, ...); bytes allocated directly in a timer, outside its child timers, are a muted "in <timer> itself" arc after the children ("outside any timer" at the root) |
| left drag | sunburst of the growth in that range: allocations made inside the range that are still alive at its end (where memory grows per time step) |

| memory panel | action |
| --- | --- |
| hover a segment | timer name, bytes and share |
| click a segment / list row | zoom the sunburst into that stack (muted "itself" and grey "smaller stacks" arcs are not zoomable) |
| click the centre / back | one level up; *root* returns to the whole stack tree |
| go to peak | move the click time to the maximum of that row |
| close (or `s`) | back to the statistics panel |

### Keyboard

| key | action |
| --- | --- |
| space / f | zoom to the full trace |
| backspace | step back to the previous view |
| escape | clear highlight / search |
| s | toggle the statistics panel (from the memory panel: switch to statistics) |
| / or ctrl/⌘+f | focus the search box |
| + / − | zoom in / out (centred) |
| arrow keys | pan (←→ time, ↑↓ rows) |
