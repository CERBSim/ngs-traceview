from ngapp import AppConfig
from . import __version__, TraceViewer

_DESCRIPTION = (
    "ViTE-like viewer for Paje trace files (NGSolve task manager / timer "
    "traces) — WebGPU-rendered timeline that handles millions of intervals."
)

config = AppConfig(
    name="Trace Viewer",
    version = __version__,
    python_class=TraceViewer,
    frontend_pip_dependencies=[],
    description=_DESCRIPTION,
)
    
