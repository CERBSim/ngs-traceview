"""WebGPU renderer and 2D view state for the trace timeline."""

import ctypes as ct
from pathlib import Path

import numpy as np

from webgpu.renderer import Renderer, RenderOptions
from webgpu.uniforms import UniformBase
from webgpu.utils import (
    buffer_from_array,
    read_shader_file,
    register_shader_directory,
)
from webgpu.webgpu_api import (
    BufferUsage,
    PrimitiveTopology,
    VertexAttribute,
    VertexBufferLayout,
    VertexFormat,
    VertexStepMode,
)

from .paje import TraceData

register_shader_directory("ngs_traceview", str(Path(__file__).parent / "shaders"))

# row layout in "row units" (one row of lane index i spans [i, i+1))
ROW_HEIGHT = 0.9
ROW_PAD = 0.05
DEPTH_INSET = 0.12


NO_HIGHLIGHT = 0xFFFFFFFF


class TimelineUniforms(UniformBase):
    """Must match TimelineUniforms in shaders/timeline.wgsl."""

    _binding = 60
    _fields_ = [
        ("off_hi", ct.c_float),
        ("off_lo", ct.c_float),
        ("scale", ct.c_float),
        ("y_off", ct.c_float),
        ("y_scale", ct.c_float),
        ("min_w", ct.c_float),
        ("canvas_w", ct.c_float),
        ("canvas_h", ct.c_float),
        ("highlight", ct.c_uint32),
        ("_pad", ct.c_uint32 * 3),
    ]

    def __init__(self, **kwargs):
        super().__init__(highlight=NO_HIGHLIGHT, **kwargs)


_INSTANCE_DTYPE = np.dtype(
    [
        ("hi", "<f4"),
        ("lo", "<f4"),
        ("dur", "<f4"),
        ("y", "<f4"),
        ("h", "<f4"),
        ("color", "<u4"),
        ("flags", "<u4"),
    ]
)


class TimelineRenderer(Renderer):
    """All trace intervals as one instanced quad draw call."""

    n_vertices = 4
    topology = PrimitiveTopology.triangle_strip
    vertex_entry_point = "vertex_timeline"
    fragment_entry_point = "fragment_timeline"
    select_entry_point = "fragment_select_timeline"

    def __init__(self, trace: TraceData):
        super().__init__(label="timeline")
        self.uniforms = TimelineUniforms()
        self._instance_buffer = None
        self.set_trace(trace)

    def set_trace(self, trace: TraceData):
        self.trace = trace
        self.n_instances = trace.n_intervals
        self._data_dirty = True
        self.set_needs_update()

    def update(self, options: RenderOptions):
        # update() runs once per frame in the legacy render path — only
        # rebuild the (large) instance buffer when the trace data changed
        if not self._data_dirty:
            return
        trace = self.trace
        inst = np.empty(trace.n_intervals, dtype=_INSTANCE_DTYPE)
        hi = trace.start.astype(np.float32)
        inst["hi"] = hi
        inst["lo"] = (trace.start - hi.astype(np.float64)).astype(np.float32)
        inst["dur"] = (trace.end - trace.start).astype(np.float32)
        depth = trace.depth.astype(np.float32)
        inst["y"] = trace.row.astype(np.float32) + ROW_PAD + DEPTH_INSET * depth
        inst["h"] = ROW_HEIGHT - 2 * DEPTH_INSET * depth
        rgba = np.round(trace.colors * 255).astype(np.uint8)  # (n_names, 4)
        inst["color"] = rgba.copy().view("<u4").ravel()[trace.value]
        # flags: bits 0-7 nesting depth, bits 8-31 entity-value id (for highlight)
        inst["flags"] = trace.depth.astype(np.uint32) | (
            trace.value.astype(np.uint32) << 8
        )
        self._instance_buffer = buffer_from_array(
            inst,
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
            label="timeline instances",
            reuse=self._instance_buffer,
        )
        self.vertex_buffers = [self._instance_buffer]
        self.vertex_buffer_layouts = [
            VertexBufferLayout(
                arrayStride=_INSTANCE_DTYPE.itemsize,
                stepMode=VertexStepMode.instance,
                attributes=[
                    VertexAttribute(format=VertexFormat.float32x4, offset=0, shaderLocation=0),
                    VertexAttribute(format=VertexFormat.float32, offset=16, shaderLocation=1),
                    VertexAttribute(format=VertexFormat.unorm8x4, offset=20, shaderLocation=2),
                    VertexAttribute(format=VertexFormat.uint32, offset=24, shaderLocation=3),
                ],
            )
        ]
        self._data_dirty = False

    def get_bindings(self):
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("ngs_traceview/timeline.wgsl")

    def get_bounding_box(self):
        # the camera is unused (we bring our own view transform), but the
        # Scene needs some box to initialize it
        return ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])


class TimelineView:
    """Visible window (time + rows) with ViTE-like pan/zoom semantics.

    Times are float64 milliseconds; the shader gets them as hi/lo float32
    pairs. All positions arrive in device pixels (canvasX/canvasY).
    """

    MIN_SPAN = 1e-6  # ms, = 1 ns

    def __init__(self, renderer: TimelineRenderer):
        self.renderer = renderer
        self.scene = None
        self.on_change = []  # callbacks (e.g. axis/label overlay updates)
        self.t0 = 0.0
        self.t1 = 1.0
        self.y0 = 0.0
        self.rows_visible = 1.0
        self.fit()

    # -- geometry --

    @property
    def trace(self) -> TraceData:
        return self.renderer.trace

    @property
    def n_rows(self) -> int:
        return len(self.trace.rows)

    @property
    def span(self) -> float:
        return self.t1 - self.t0

    def _canvas_size(self):
        canvas = self.scene.canvas if self.scene else None
        if canvas is None or not canvas.width:
            return 800, 600
        return canvas.width, canvas.height

    def time_at(self, px: float) -> float:
        w, _ = self._canvas_size()
        return self.t0 + (px / max(w, 1)) * self.span

    def row_at(self, py: float) -> int | None:
        _, h = self._canvas_size()
        row = int(self.y0 + (py / max(h, 1)) * self.rows_visible)
        return row if 0 <= row < self.n_rows else None

    # -- view manipulation --

    def set_highlight(self, value):
        """Highlight one entity-value id (dim the rest); None clears."""
        self.renderer.uniforms.highlight = (
            NO_HIGHLIGHT if value is None else int(value)
        )
        self.apply()

    def fit(self):
        trace = self.trace
        pad = 0.01 * (trace.tmax - trace.tmin or 1.0)
        self.t0 = trace.tmin - pad
        self.t1 = trace.tmax + pad
        self.y0 = 0.0
        self.rows_visible = max(self.n_rows, 1)

    def zoom_time(self, px: float, factor: float):
        """Zoom the time axis by *factor*, keeping the time at pixel *px* fixed."""
        t_fix = self.time_at(px)
        trace = self.trace
        full = max(trace.tmax - trace.tmin, 1.0)
        new_span = self.span / factor
        new_span = min(max(new_span, self.MIN_SPAN), 4 * full)
        f = (t_fix - self.t0) / self.span
        self.t0 = t_fix - f * new_span
        self.t1 = self.t0 + new_span
        self._clamp()

    def zoom_rows(self, py: float, factor: float):
        """Zoom the row axis, keeping the row at pixel *py* fixed."""
        _, h = self._canvas_size()
        r_fix = self.y0 + (py / max(h, 1)) * self.rows_visible
        new_vis = self.rows_visible / factor
        new_vis = min(max(new_vis, 2.0), max(self.n_rows, 2))
        f = (r_fix - self.y0) / self.rows_visible
        self.y0 = r_fix - f * new_vis
        self.rows_visible = new_vis
        self._clamp()

    def pan_px(self, dx: float, dy: float):
        w, h = self._canvas_size()
        self.t0 -= dx / max(w, 1) * self.span
        self.t1 = self.t0 + self.span
        self.y0 -= dy / max(h, 1) * self.rows_visible
        self._clamp()

    def _clamp(self):
        trace = self.trace
        span = self.span
        full = trace.tmax - trace.tmin
        # keep at least a sliver of the trace on screen
        self.t0 = min(max(self.t0, trace.tmin - 2 * full - span), trace.tmax + full)
        self.t1 = self.t0 + span
        max_y0 = max(self.n_rows - self.rows_visible, 0.0)
        self.y0 = min(max(self.y0, 0.0), max_y0)

    # -- applying to the GPU --

    def attach(self, scene):
        self.scene = scene
        self.apply()

    def apply(self):
        u = self.renderer.uniforms
        w, h = self._canvas_size()
        off_hi = np.float32(self.t0)
        u.off_hi = off_hi
        u.off_lo = np.float32(self.t0 - np.float64(off_hi))
        u.scale = 2.0 / self.span
        u.y_off = self.y0
        u.y_scale = 2.0 / self.rows_visible
        u.min_w = 2.0 / w  # ~1 device pixel
        u.canvas_w = w
        u.canvas_h = h
        u.update_buffer()

        scene = self.scene
        if scene is not None and scene.canvas is not None:
            import os

            if os.environ.get("NGS_TRACEVIEW_DEBUG"):
                print(
                    f"view.apply: t0={self.t0:.6f} span={self.span:.6f} "
                    f"y0={self.y0:.2f} rows_vis={self.rows_visible:.2f} "
                    f"canvas={scene.canvas.width}x{scene.canvas.height}"
                )
            # view changed → GPU pick buffer content is stale
            scene._select_buffer_valid = False
            with scene._render_mutex:
                # fast path: uniform-only change, skip pipeline updates
                scene._render_highlight()
        for cb in self.on_change:
            cb()
