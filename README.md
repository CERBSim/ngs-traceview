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

### Controls

| input | action |
| --- | --- |
| mouse wheel | zoom time axis (at cursor) |
| ctrl + wheel | zoom rows (at cursor) |
| left drag | pan |
| hover | tooltip with task name and duration |
| click | pin task details + aggregate statistics in the bottom panel |
| double click / fit button | zoom to full trace |
