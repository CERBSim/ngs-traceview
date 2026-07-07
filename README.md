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

### Keyboard

| key | action |
| --- | --- |
| space / f | zoom to the full trace |
| backspace | step back to the previous view |
| escape | clear highlight / search |
| s | toggle the statistics panel |
| / or ctrl/⌘+f | focus the search box |
| + / − | zoom in / out (centred) |
| arrow keys | pan (←→ time, ↑↓ rows) |
