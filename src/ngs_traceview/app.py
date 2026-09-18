import math
import os
import re
import threading
import time

import numpy as np
from ngapp.app import App
from ngapp.components import (
    Component,
    Div,
    FileUpload,
    QBtn,
    QBtnToggle,
    QIcon,
    QInput,
    QLinearProgress,
    QSpace,
    QSpinnerGears,
    QSplitter,
    QTable,
    QTooltip,
    WebgpuComponent,
)

from . import memory, style
from .memory import format_bytes
from .style import AXIS_HEIGHT, LABEL_WIDTH

MAX_TICKS = 11
NAME_MAX = 90  # truncate long C++ symbols in the table
SUN_RINGS = 5  # rings drawn below the focused stack
SUN_LIST = 12  # children listed under the sunburst


def nice_ticks(t0: float, t1: float, max_ticks: int = MAX_TICKS):
    """1-2-5 tick positions covering [t0, t1] (times in ms)."""
    span = max(t1 - t0, 1e-12)
    raw = span / max_ticks
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * mag >= raw:
            step = m * mag
            break
    first = math.ceil(t0 / step) * step
    ticks = []
    t = first
    while t <= t1:
        ticks.append(t)
        t += step
    return ticks, step


def format_time(t_ms: float, step_ms: float) -> str:
    """Format an absolute trace time with a unit chosen from the tick step."""
    if step_ms >= 100.0:
        factor, unit = 1000.0, "s"
    elif step_ms >= 0.1:
        factor, unit = 1.0, "ms"
    elif step_ms >= 1e-4:
        factor, unit = 1e-3, "µs"
    else:
        factor, unit = 1e-6, "ns"
    decimals = max(0, -math.floor(math.log10(step_ms / factor) + 1e-9))
    return f"{t_ms / factor:.{decimals}f} {unit}"


def format_duration(d_ms: float) -> str:
    for factor, unit in ((1000.0, "s"), (1.0, "ms"), (1e-3, "µs")):
        if d_ms >= factor:
            return f"{d_ms / factor:.3g} {unit}"
    return f"{d_ms / 1e-6:.3g} ns"


def _hex(color) -> str:
    r, g, b = (int(round(c * 255)) for c in color[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _swatch(color):
    return Div(ui_class=str(style.swatch), ui_style=f"background:{_hex(color)};")


def _short(name: str, n: int = NAME_MAX) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _svg(tag: str, *children, **attrs):
    """SVG element as an ngapp component (attribute names as in SVG)."""
    c = Component(tag, *children)
    for k, v in attrs.items():
        c._props[k] = v
    return c


def _arc_path(cx, cy, r0, r1, a0, a1):
    """Ring sector between radii r0 < r1 and angles a0 < a1 (radians, clockwise
    from 12 o'clock)."""
    a1 = min(a1, a0 + 2 * math.pi - 1e-3)
    large = 1 if a1 - a0 > math.pi else 0
    x = lambda r, a: cx + r * math.sin(a)
    y = lambda r, a: cy - r * math.cos(a)
    return (
        f"M{x(r1, a0):.2f} {y(r1, a0):.2f} A{r1} {r1} 0 {large} 1 {x(r1, a1):.2f} {y(r1, a1):.2f} "
        f"L{x(r0, a1):.2f} {y(r0, a1):.2f} A{r0} {r0} 0 {large} 0 {x(r0, a0):.2f} {y(r0, a0):.2f} Z"
    )


class TraceViewer(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        style.install(self, default_theme="system")
        self._dark = style.resolved_theme() == "dark"

        self.trace = None
        self.renderer = None
        self.view = None

        self._build_ui()

        self._tick_pool = []
        self._label_pool = []
        self._drag_last = None
        self._drag_mode = None  # "pan" | "select" | "back"
        self._sel_start = 0
        self._sel_moved = False
        self._pan_pushed = False
        self._history = []  # view snapshots for right-click "back"
        self._last_wheel_push = 0.0
        self._hover_px = (0, 0)
        self._shown_pick = None  # interval idx currently in the tooltip
        self._click_highlight = None  # fn value highlighted via double-click
        self._hide_timer = None
        self._stats_open = False
        self._stats_width = 460  # remembered panel width (px)
        self._stats_mode = "all"  # "all" | "view"
        self._stats_timer = None
        self._loading_path = None
        self._side_mode = "stats"  # which panel fills the side pane: "stats" | "memory"
        self._hover_line_on = False
        self._mem_renderer = None
        self._mem_kind = None  # "host" | "device" while the memory panel is open
        self._mem_time = None  # click time (alive-at view)
        self._mem_range = None  # (t0, t1) for the growth view
        self._mem_focus = 0
        self._mem_hist = []  # focus history for "back"
        self._mem_incl = None
        self._mem_drag_kind = None

        self.canvas.on_mounted(self._on_canvas_mounted)
        self.on_mounted(self._apply_quasar_dark)
        self._setup_keybindings()

    # ---- keybindings ----

    def _setup_keybindings(self):
        # hotkeys-js: single/named keys; auto-ignored while typing in inputs
        keys = {
            "space": self._on_fit,                       # zoom to full trace
            "f": self._on_fit,
            "backspace": self._restore_previous,         # back (undo zoom/pan)
            "escape": self._clear_highlight,             # clear highlight/search
            "s": self._toggle_stats,                     # statistics panel
            "=": lambda: self._key_zoom(1.4),            # zoom in (centre)
            "+": lambda: self._key_zoom(1.4),
            "-": lambda: self._key_zoom(1 / 1.4),        # zoom out
            "left": lambda: self._key_pan(90, 0),        # pan earlier
            "right": lambda: self._key_pan(-90, 0),      # pan later
            "up": lambda: self._key_pan(0, 90),          # pan rows
            "down": lambda: self._key_pan(0, -90),
        }
        for key, fn in keys.items():
            self.add_keybinding(key, self._make_key_handler(fn))

    def _make_key_handler(self, fn):
        # the keybinding callback is invoked with the event; our actions take no
        # args. (preventDefault can't be done here — over the websocket link it
        # is async and fires too late; keys that need it are handled in JS via
        # _install_search_keys.)
        def handler(ev=None):
            fn()
        return handler

    def _blur_search(self):
        try:
            self.search_input._js_call_method("blur")
        except Exception:
            pass

    _SEARCH_KEYS_JS = r"""(function(){
      if (window.__tvSearchKeys) return;
      window.__tvSearchKeys = true;
      function editable(el){ return el && (el.tagName === 'INPUT' ||
        el.tagName === 'TEXTAREA' || el.isContentEditable); }
      document.addEventListener('keydown', function(e){
        var f = document.querySelector('.tv-search input');
        if (!f) return;
        var find = (e.ctrlKey || e.metaKey) && (e.key === 'f' || e.key === 'F');
        var slash = e.key === '/' && !editable(document.activeElement);
        if (find || slash) { e.preventDefault(); f.focus(); f.select(); }
        else if (e.key === 'Escape' && document.activeElement === f) { f.blur(); }
      });
    })();"""

    def _install_search_keys(self):
        # Focus the search box on '/', Ctrl/Cmd+F (preventing the browser's find
        # bar) and blur it on Escape — all in JS, because preventDefault must be
        # synchronous and can't be driven from Python over the websocket link.
        def _run(js):
            try:
                js.eval(self._SEARCH_KEYS_JS)
            except Exception:
                pass

        try:
            self.call_js(_run)
        except Exception:
            pass

    def _key_zoom(self, factor):
        if self.view is None:
            return
        self._push_history()
        w = self.view.scene.canvas.width if self.view.scene else 800
        self.view.zoom_time(w / 2, factor)
        self.view.apply()

    def _key_pan(self, dx, dy):
        if self.view is None:
            return
        self.view.pan_px(dx, dy)
        self.view.apply()

    # ---- UI construction ----

    def _build_ui(self):
        self.file_input = FileUpload(
            id="file_upload",
            ui_label="Open .trace file",
            ui_dense=True,
            ui_outlined=True,
            ui_accept=".trace",
            ui_style="width: 240px;",
        )
        self.file_input.on_upload_complete(self._on_upload)

        self.btn_fit = QBtn(
            QTooltip("Zoom to full trace"),
            ui_icon="fit_screen", ui_flat=True, ui_dense=True, ui_round=True,
        )
        self.btn_fit.on_click(self._on_fit)
        self.btn_clear = QBtn(
            QTooltip("Clear highlight & search"),
            ui_icon="layers_clear", ui_flat=True, ui_dense=True, ui_round=True,
        )
        self.btn_clear.on_click(lambda *_: self._clear_highlight())
        self.btn_stats = QBtn(
            QTooltip("Toggle statistics"),
            ui_icon="bar_chart", ui_flat=True, ui_dense=True, ui_round=True,
        )
        self.btn_stats.on_click(self._toggle_stats)

        # regex search: highlights every function whose name matches
        self.search_input = QInput(
            QTooltip("Highlight functions matching a regex (case-insensitive)"),
            ui_model_value="",
            ui_dense=True,
            ui_outlined=True,
            ui_clearable=True,
            ui_debounce=200,
            ui_hide_bottom_space=True,
            ui_slots={"prepend": [QIcon(ui_name="search", ui_size="18px")]},
            ui_class="tv-search",
            ui_style="width: 240px;",
        )
        # `placeholder` is a native <input> attribute, not a generated ui_ prop
        self.search_input._props["placeholder"] = "highlight regex…"
        self.search_input.on("update:model-value", self._on_search)
        self.search_input.on_mounted(self._install_search_keys)

        self.info_label = Div("no trace loaded", ui_class=str(style.info))

        bar = Div(
            Div(
                Div("Trace Viewer", ui_class=str(style.brand_name)),
                Div("Paje / ngcore timeline", ui_class=str(style.brand_sub)),
                ui_class=str(style.brand),
            ),
            Div(ui_class=str(style.sep)),
            self.file_input,
            self.btn_fit,
            self.btn_clear,
            self.search_input,
            QSpace(),
            self.info_label,
            Div(ui_class=str(style.sep)),
            self.btn_stats,
            ui_class=str(style.bar),
        )

        # timeline: axis strip + (labels | canvas)
        self._axis = Div(ui_class=str(style.axis))
        self._labels = Div(ui_class=str(style.labels))
        self.canvas = WebgpuComponent(width="100%", height="100%", id="timeline_canvas")
        self.tooltip = self._build_tooltip()
        self.sel_box = Div(ui_class=str(style.sel_box))
        self.sel_box.ui_style = "display:none;"
        self.hover_line = Div(ui_class=str(style.hover_line), ui_style="display:none;")
        self.mem_mark = Div(ui_class=str(style.mem_mark), ui_style="display:none;")
        canvas_wrap = Div(
            self.canvas, self.sel_box, self.mem_mark, self.hover_line, self.tooltip,
            ui_class=str(style.canvas_wrap),
        )
        timeline_col = Div(
            self._axis,
            Div(self._labels, canvas_wrap, ui_class=str(style.body)),
            ui_class=str(style.timeline_col),
        )

        self.stats_panel = self._build_stats_panel()
        self.mem_panel = self._build_mem_panel()
        self.mem_panel.ui_hidden = True
        side = Div(self.stats_panel, self.mem_panel,
                   ui_style="display:flex; width:100%; height:100%; min-height:0;")

        self.loading = self._build_loading()
        self.loading.ui_hidden = True

        # QSplitter makes the stats panel width draggable; reverse=True measures
        # the right ("after") pane in px, and 0 collapses it (panel closed)
        self.splitter = QSplitter(
            ui_model_value=0,
            ui_unit="px",
            ui_reverse=True,
            ui_limits=[0, 1000],
            ui_emit_immediately=True,
            ui_slots={"before": [timeline_col], "after": [side]},
            ui_style="height:100%; width:100%;",
        )
        self.splitter.on("update:model-value", self._on_splitter)
        mid = Div(self.splitter, self.loading, ui_class=str(style.mid))

        # detail bar is always present (fixed height) so clicking never shifts
        # the layout; empty state shows a faint hint
        self.detail = Div(ui_class=str(style.detail))
        self.status = Div("open a .trace file to begin", ui_class=str(style.status))

        self.component = Div(bar, mid, self.detail, self.status, ui_class=str(style.page))

    def _build_loading(self):
        self.loading_title = Div("Loading trace", ui_class=str(style.load_title))
        self.loading_msg = Div("", ui_class=str(style.load_msg))
        self.loading_bar = QLinearProgress(
            ui_value=0.0, ui_indeterminate=False, ui_rounded=True,
            ui_color="primary", ui_track_color="grey-8" if self._dark else "grey-3",
            ui_size="6px", ui_class=str(style.load_bar),
        )
        card = Div(
            QSpinnerGears(ui_size="46px", ui_color="primary"),
            self.loading_title,
            self.loading_msg,
            self.loading_bar,
            ui_class=str(style.load_card),
        )
        return Div(card, ui_class=str(style.overlay))

    def _set_loading(self, active, title=None, msg=None, frac=None):
        self.loading.ui_hidden = not active
        if not active:
            return
        if title is not None:
            self.loading_title.ui_children = [title]
        if msg is not None:
            self.loading_msg.ui_children = [msg]
        if frac is None:
            self.loading_bar.ui_indeterminate = True
        else:
            self.loading_bar.ui_indeterminate = False
            self.loading_bar.ui_value = max(0.0, min(1.0, float(frac)))

    def _build_tooltip(self):
        self._tt_title = Div(ui_class=str(style.tip_title))
        self._tt_sub = Div(ui_class=str(style.tip_sub))
        tip = Div(self._tt_title, self._tt_sub, ui_class=str(style.tooltip))
        tip.ui_style = "display:none;"
        return tip

    def _build_stats_panel(self):
        self.stats_mode_toggle = QBtnToggle(
            ui_model_value="all",
            ui_options=[
                {"label": "Whole trace", "value": "all"},
                {"label": "Current view", "value": "view"},
            ],
            ui_dense=True, ui_flat=True, ui_no_caps=True, ui_toggle_color="primary",
            ui_style="font-size:11px;",
        )
        self.stats_mode_toggle.on("update:model-value", self._on_stats_mode)

        columns = [
            {"name": "name", "label": "Function", "field": "name", "align": "left",
             "sortable": True, "classes": "tv-fn", "headerClasses": "tv-fn"},
            {"name": "count", "label": "Calls", "field": "count", "align": "right", "sortable": True},
            {"name": "total", "label": "Total s", "field": "total", "align": "right", "sortable": True},
            {"name": "pct", "label": "Busy %", "field": "pct", "align": "right", "sortable": True},
            {"name": "mean", "label": "Mean ms", "field": "mean", "align": "right", "sortable": True},
            {"name": "max", "label": "Max ms", "field": "max", "align": "right", "sortable": True},
        ]
        self.stats_table = QTable(
            ui_columns=columns,
            ui_rows=[],
            ui_row_key="id",
            ui_dense=True,
            ui_flat=True,
            ui_selection="multiple",
            ui_selected=[],
            ui_virtual_scroll=True,
            ui_pagination={"rowsPerPage": 0},
            ui_hide_bottom=True,
            ui_dark=self._dark,
            ui_style="height:100%;",
        )
        self.stats_table.on("update:selected", self._on_stats_select)

        head = Div(
            Div("Statistics", ui_class=str(style.stats_title)),
            QSpace(),
            self.stats_mode_toggle,
            ui_class=str(style.stats_head),
        )
        return Div(
            head,
            Div(self.stats_table, ui_class=str(style.stats_body)),
            ui_class=str(style.stats),
        )

    def _build_mem_panel(self):
        self.mem_title = Div("Memory", ui_class=str(style.stats_title))
        btn_peak = QBtn(
            QTooltip("Move the click time to the maximum of this row"),
            ui_label="go to peak", ui_icon="vertical_align_top", ui_flat=True,
            ui_dense=True, ui_no_caps=True, ui_size="sm",
        )
        btn_peak.on_click(self._mem_go_to_peak)
        btn_close = QBtn(
            QTooltip("Close (back to statistics)"),
            ui_icon="close", ui_flat=True, ui_dense=True, ui_round=True, ui_size="sm",
        )
        btn_close.on_click(lambda *_: self._close_memory())
        head = Div(self.mem_title, QSpace(), btn_peak, btn_close,
                   ui_class=str(style.stats_head))

        self.mem_mode = Div(ui_class=str(style.mem_mode))
        self.mem_hero_label = Div(ui_class=str(style.mem_tile_label))
        self.mem_hero = Div(ui_class=str(style.mem_hero))
        self.mem_peak_label = Div("peak", ui_class=str(style.mem_tile_label))
        self.mem_peak = Div(ui_class=str(style.mem_value))
        tiles = Div(
            Div(self.mem_hero_label, self.mem_hero),
            Div(self.mem_peak_label, self.mem_peak),
            ui_class=str(style.mem_tiles),
        )
        btn_back = QBtn(QTooltip("One level up"), ui_icon="arrow_back", ui_flat=True,
                        ui_dense=True, ui_round=True, ui_size="sm")
        btn_back.on_click(lambda *_: self._mem_back())
        btn_root = QBtn(QTooltip("Back to the whole stack tree"), ui_icon="home",
                        ui_flat=True, ui_dense=True, ui_round=True, ui_size="sm")
        btn_root.on_click(lambda *_: self._mem_zoom(0, push=True))
        self.mem_crumb = Div(ui_class=str(style.mem_crumb))
        nav = Div(btn_back, btn_root, self.mem_crumb, ui_class=str(style.mem_nav))

        self.mem_svg = _svg("svg", viewBox="0 0 400 400")
        self.mem_svg.ui_class = str(style.mem_sun)
        self.mem_center = Div(ui_class=str(style.mem_center))
        sun = Div(self.mem_svg, self.mem_center, ui_class=str(style.mem_sun_wrap))
        self.mem_list_head = Div(ui_class=str(style.mem_list_head))
        self.mem_list = Div(ui_class=str(style.mem_list))

        body = Div(self.mem_mode, tiles, nav, sun, self.mem_list_head, self.mem_list,
                   ui_class=str(style.mem_body))
        panel = Div(head, body, ui_class=str(style.stats))
        panel.on_mounted(self._install_sun_tip)
        return panel

    _SUN_TIP_JS = r"""(function(){
      if (window.__tvSunTip) return;
      window.__tvSunTip = true;
      var tip = document.createElement('div');
      tip.className = 'tv-suntip';
      tip.style.display = 'none';
      document.body.appendChild(tip);
      // a click rerenders the sunburst: the hovered segment is gone
      document.addEventListener('mousedown', function(){ tip.style.display = 'none'; }, true);
      document.addEventListener('mousemove', function(e){
        var el = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
        if (!el) { tip.style.display = 'none'; return; }
        tip.textContent = el.getAttribute('data-tip');
        tip.style.display = 'block';
        var w = window.innerWidth, h = window.innerHeight;
        tip.style.left = (e.clientX > w * 0.6 ? '' : (e.clientX + 14) + 'px');
        tip.style.right = (e.clientX > w * 0.6 ? (w - e.clientX + 14) + 'px' : '');
        tip.style.top = (e.clientY > h * 0.65 ? '' : (e.clientY + 14) + 'px');
        tip.style.bottom = (e.clientY > h * 0.65 ? (h - e.clientY + 14) + 'px' : '');
      });
    })();"""

    def _install_sun_tip(self):
        # cursor-following tooltip for the sunburst segments, entirely in the
        # browser (native events reach Python without cursor coordinates)
        def _run(js):
            try:
                js.eval(self._SUN_TIP_JS)
            except Exception:
                pass

        try:
            self.call_js(_run)
        except Exception:
            pass

    def _apply_quasar_dark(self):
        def _set(js):
            try:
                js.eval(f"window.$q && window.$q.dark && window.$q.dark.set({str(self._dark).lower()})")
            except Exception:
                pass
        try:
            self.call_js(_set)
        except Exception:
            pass

    # ---- loading ----

    def _on_canvas_mounted(self):
        path = os.environ.get("NGS_TRACEVIEW_FILE")
        if path and os.path.exists(path) and self.trace is None:
            self._load_async(path)

    def _on_upload(self):
        def work():
            with self.file_input.as_temporary_file as path:
                self._load(str(path))

        threading.Thread(target=work, daemon=True).start()

    def _load_async(self, path: str):
        threading.Thread(target=self._load, args=(path,), daemon=True).start()

    def _on_progress(self, frac, msg):
        base = os.path.basename(self._loading_path or "")
        self._set_loading(True, f"Loading {base}", msg, frac)

    def _load(self, path: str):
        from . import paje

        self._loading_path = path
        base = os.path.basename(path)
        self._set_loading(True, f"Loading {base}", "reading file", 0.0)
        self.status.ui_children = [f"parsing {base} …"]
        try:
            trace = paje.parse(path, progress=self._on_progress)
        except Exception as e:
            self._set_loading(False)
            self.status.ui_children = [f"failed to load {base}: {e}"]
            raise
        self.trace = trace
        mem_info = (
            f"  ·  {trace.memory.n_events:,} memory events" if trace.memory else ""
        )
        self.info_label.ui_children = [
            f"{base}  ·  {trace.n_intervals:,} intervals  ·  "
            f"{len(trace.rows)} rows{mem_info}  ·  {trace.parse_time:.1f}s"
        ]
        self._set_loading(True, f"Loading {base}", "uploading to GPU", None)
        self.status.ui_children = ["uploading to GPU …"]
        try:
            self._draw()
        except Exception:
            import traceback

            traceback.print_exc()
            self._set_loading(False)
            self.status.ui_children = ["draw failed — see console"]
            return
        self._set_loading(False)
        self._refresh_stats()
        self.status.ui_children = [
            "drag: zoom to range · right-click: back · "
            "middle/shift-drag or two-finger swipe: pan · wheel: zoom · "
            "click: info · double-click: highlight"
        ]

    # ---- rendering ----

    def _draw(self):
        from .timeline import MemoryRenderer, TimelineRenderer, TimelineView

        self._close_memory(refresh=False)
        self.renderer = TimelineRenderer(self.trace)
        renderers = [self.renderer]
        self._mem_renderer = None
        if self.trace.memory is not None:
            self._mem_renderer = MemoryRenderer(self.trace, dark=self._dark)
            renderers.append(self._mem_renderer)
        self.view = TimelineView(self.renderer, aux=[r for r in renderers[1:]])
        # legacy (Python-driven) render path: the JS engine's built-in 3D
        # camera would consume drag/wheel, which we need for 2D pan/zoom
        scene = self.canvas.draw(renderers, use_js_engine=False)
        self.view.attach(scene)
        self.view.on_change.append(self._update_overlays)
        self.view.on_change.append(self._on_view_changed)

        # replace the 3D camera gestures with timeline pan/zoom; the camera
        # would re-register on visibility changes, so disable it for good
        camera = scene.options.camera
        camera.unregister_callbacks(scene.input_handler)
        camera.register_callbacks = lambda *a, **k: None
        ih = scene.input_handler
        ih.on_mousedown(self._on_mousedown)
        ih.on_mouseup(self._on_mouseup)
        ih.on_drag(self._on_drag)
        ih.on_wheel(self._on_wheel)
        ih.on_dblclick(self._on_dblclick)
        ih.on_mousemove(self._on_hover)
        ih.on_mouseout(self._on_mouseout)
        ih.on_click(self._on_click)
        self.renderer.on_select(self._on_pick)
        scene.on_click_background(self._on_pick_background)
        self.canvas.canvas.on_resize(self._on_resize)

        self._build_overlay_pools()
        self.view.apply()
        self._show_detail_empty()

    def _on_resize(self, *args):
        if self.view is not None:
            self.view.apply()

    # ---- interaction ----

    # -- view history (right-click steps back) --

    def _push_history(self):
        v = self.view
        if v is None:
            return
        self._history.append((v.t0, v.t1, v.y0, v.rows_visible))
        if len(self._history) > 500:
            self._history.pop(0)

    def _restore_previous(self):
        if self.view is None or not self._history:
            return
        v = self.view
        v.t0, v.t1, v.y0, v.rows_visible = self._history.pop()
        v.apply()

    def _on_mousedown(self, ev):
        # interacting with the view drops focus from the search box (the canvas
        # preventDefaults the click, so the browser won't blur it for us) — so
        # keyboard shortcuts like space work again
        self._blur_search()
        button = ev.get("button", 0)
        if button == 2:  # right button: go back to the previous view
            self._drag_mode = "back"
            self._restore_previous()
            return
        x, y = ev["canvasX"], ev["canvasY"]
        self._drag_last = (x, y)
        # plain left-drag = rubber-band time zoom (ViTE); shift or middle = pan;
        # on a memory row a left-drag selects the range for the growth sunburst
        pan = ev.get("shiftKey") or button == 1 or ev.get("buttons") == 4
        self._drag_mode = "pan" if pan else "select"
        self._mem_drag_kind = None if pan else self._memory_kind_at(y)
        if self._mem_drag_kind is not None:
            self._drag_mode = "memsel"
        self._sel_start = x
        self._sel_moved = False
        self._pan_pushed = False

    def _on_mouseup(self, ev):
        if self.view is not None and self._sel_moved:
            a = self.view.time_at(min(self._sel_start, ev["canvasX"]))
            b = self.view.time_at(max(self._sel_start, ev["canvasX"]))
            if self._drag_mode == "select":
                self._push_history()
                self.view.set_time_range(a, b)
                self.view.apply()
            elif self._drag_mode == "memsel":
                self._open_memory(self._mem_drag_kind, t0=a, t1=b)
        self._hide_selection()
        self._drag_last = None
        self._drag_mode = None

    def _on_drag(self, ev):
        if self.view is None:
            return
        x, y = ev["canvasX"], ev["canvasY"]
        self._cancel_hide()
        self._hide_now()
        # middle button (buttons==4) or shift held → always pan, even if the
        # mousedown that set the mode was missed; this keeps middle-drag a pure
        # pan instead of ever falling into the box-zoom select.
        if ev.get("buttons") == 4 or ev.get("shiftKey"):
            if self._drag_mode != "pan":
                self._drag_mode = "pan"
                self._pan_pushed = False
                self._hide_selection()
                self._drag_last = (x, y)
        if self._drag_mode == "pan":
            if not self._pan_pushed:  # one history entry per pan gesture
                self._push_history()
                self._pan_pushed = True
            if self._drag_last is not None:
                self.view.pan_px(x - self._drag_last[0], y - self._drag_last[1])
                self.view.apply()
            self._drag_last = (x, y)
        elif self._drag_mode in ("select", "memsel"):
            if abs(x - self._sel_start) > 3:
                self._sel_moved = True
            self._show_selection(self._sel_start, x)

    def _show_selection(self, x0_dev, x1_dev):
        dpr = (self.view.scene.canvas.dpr if self.view.scene else 1) or 1
        lo = min(x0_dev, x1_dev) / dpr
        hi = max(x0_dev, x1_dev) / dpr
        self.sel_box.ui_style = f"display:block; left:{lo:.0f}px; width:{hi - lo:.0f}px;"

    def _hide_selection(self):
        self.sel_box.ui_style = "display:none;"

    def _on_wheel(self, ev):
        if self.view is None:
            return
        dx = ev.get("deltaX", 0) or 0
        dy = ev.get("deltaY", 0) or 0
        # two-finger trackpad swipe (has a horizontal component) or shift+scroll
        # → pan; a horizontal swipe pans time, a vertical one pans rows.
        if dx != 0 or ev.get("shiftKey"):
            if dx != 0:
                self.view.pan_px(-dx, -dy)  # trackpad: both axes
            else:
                self.view.pan_px(-dy, 0)  # shift + vertical wheel → pan time
            self.view.apply()
            return
        # otherwise zoom (mouse wheel / vertical two-finger / pinch)
        now = time.time()
        if now - self._last_wheel_push > 0.4:
            self._push_history()
        self._last_wheel_push = now
        factor = 2.0 ** (-dy / 240.0)
        if ev.get("ctrlKey"):
            self.view.zoom_rows(ev["canvasY"], factor)
        else:
            self.view.zoom_time(ev["canvasX"], factor)
        self.view.apply()

    def _on_dblclick(self, ev):
        """Double-click a block: highlight that function and dim everything else."""
        trace = self.trace
        idx = self._shown_pick
        if trace is None or idx is None:
            return
        value = int(trace.value[idx])
        self._click_highlight = value
        self._set_highlight(value)
        self._sync_stats_selection(value)
        self._show_detail(value, idx)

    def _on_fit(self, *_):
        if self.view is None:
            return
        self._push_history()
        self.view.fit()
        self.view.apply()

    def _set_highlight(self, value):
        if self.view is not None:
            self.view.set_highlight(value)

    def _clear_highlight(self, *_):
        self._click_highlight = None
        self.search_input.ui_model_value = ""
        self.search_input.ui_error = False
        self._set_highlight(None)
        self._sync_stats_selection(None)
        self._show_detail_empty()

    # ---- regex search highlight ----

    def _on_search(self, ev):
        text = ev.value if hasattr(ev, "value") else ev
        text = (text or "").strip()
        if self.trace is None or self.view is None:
            return
        if not text:
            self.search_input.ui_error = False
            self._set_highlight(None)
            self._sync_stats_selection(None)
            return
        try:
            rx = re.compile(text, re.IGNORECASE)
        except re.error:
            self.search_input.ui_error = True  # invalid regex: leave view as is
            return
        self.search_input.ui_error = False
        self._click_highlight = None  # search now drives the highlight
        matches = [i for i, name in enumerate(self.trace.names) if rx.search(name)]
        self._set_highlight(matches or None)
        if self._stats_open:
            self.stats_table.ui_selected = []  # search drives the highlight now
        self.status.ui_children = [
            f"/{text}/  ·  {len(matches)} of {len(self.trace.names)} functions highlighted"
        ]

    # ---- hover tooltip (GPU picking) ----

    def _on_hover(self, ev):
        if self.view is None:
            return
        # during a drag (button held) skip GPU picking — the "mousemove" event
        # fires alongside "drag", and a select round-trip per move would clog
        # the link and starve the drag/mouseup events
        if self._drag_mode in ("pan", "select", "memsel", "back") or ev.get("buttons"):
            return
        self._hover_px = (ev["canvasX"], ev["canvasY"])
        kind = self._memory_kind_at(ev["canvasY"])
        if kind is not None:
            self._hover_memory(kind, ev["canvasX"])
            return
        self._set_hover_line(None)
        self.canvas.select(ev["canvasX"], ev["canvasY"])

    def _on_mouseout(self, ev):
        self._cancel_hide()
        self._hide_now()
        self._set_hover_line(None)

    # ---- memory rows ----

    def _memory_kind_at(self, py) -> str | None:
        """'host' / 'device' if the canvas pixel row is a memory row."""
        if self.view is None or self.trace is None or self.trace.memory is None:
            return None
        row = self.view.row_at(py)
        return None if row is None else self.trace.memory.kind_of_row(row)

    def _set_hover_line(self, x_dev):
        if x_dev is None:
            if self._hover_line_on:
                self._hover_line_on = False
                self.hover_line.ui_style = "display:none;"
            return
        dpr = (self.view.scene.canvas.dpr if self.view.scene else 1) or 1
        self._hover_line_on = True
        self.hover_line.ui_style = f"display:block; left:{x_dev / dpr:.0f}px;"

    def _hover_memory(self, kind, x_dev):
        self._cancel_hide()
        self._shown_pick = None
        k = self.trace.memory.kind(kind)
        t = self.view.time_at(x_dev)
        value = k.value_at(t)
        _, step = nice_ticks(self.view.t0, self.view.t1)
        self._tt_title.ui_children = [format_bytes(value)]
        self._tt_sub.ui_children = [
            Div(f"memory {kind}  ·  @ {format_time(t, step / 100)}  ·  {value:,.0f} B")
        ]
        self.status.ui_children = [
            f"memory {kind}:  {format_bytes(value)} @ {format_time(t, step / 100)}  ·  "
            "click: sunburst of memory alive here · drag: growth in a range"
        ]
        self._set_hover_line(x_dev)
        self._place_tooltip()

    def _open_memory(self, kind, t=None, t0=None, t1=None):
        self._mem_kind = kind
        self._mem_time = t
        self._mem_range = None if t0 is None else (min(t0, t1), max(t0, t1))
        self._mem_focus = 0
        self._mem_hist = []
        self._show_side("memory")
        self._render_memory()
        self._update_overlays()

    def _close_memory(self, refresh=True):
        self._mem_kind = None
        self._mem_incl = None
        self.mem_mark.ui_style = "display:none;"
        if refresh:
            self._show_side("stats")

    def _mem_go_to_peak(self, *_):
        if self._mem_kind is None:
            return
        k = self.trace.memory.kind(self._mem_kind)
        t_peak, _ = k.peak()
        self._mem_time, self._mem_range = t_peak, None
        self._mem_focus, self._mem_hist = 0, []
        v = self.view
        if not (v.t0 <= t_peak <= v.t1):  # bring the peak into view
            self._push_history()
            v.set_time_range(t_peak - v.span / 2, t_peak + v.span / 2)
            v.apply()
        self._render_memory()
        self._update_overlays()

    def _mem_zoom(self, sid, push=True):
        if self._mem_kind is None or sid == self._mem_focus:
            return
        if push:
            self._mem_hist.append(self._mem_focus)
        self._mem_focus = int(sid)
        self._render_memory(recompute=False)

    def _mem_back(self):
        if self._mem_hist:
            self._mem_focus = self._mem_hist.pop()
        elif self._mem_focus:
            self._mem_focus = int(self.trace.memory.stack_parent[self._mem_focus])
        else:
            return
        self._render_memory(recompute=False)

    def _stack_name(self, sid):
        m = self.trace.memory
        return "all memory" if sid == 0 else self.trace.names[m.stack_value[sid]]

    def _own_label(self, sid):
        return "outside any timer" if sid == 0 else f"in {_short(self._stack_name(sid), 60)} itself"

    def _stack_color(self, sid):
        m = self.trace.memory
        return self.trace.colors[m.stack_value[sid]] if sid else (0.55, 0.55, 0.55)

    def _render_memory(self, recompute=True):
        trace = self.trace
        if trace is None or trace.memory is None or self._mem_kind is None:
            return
        m = trace.memory
        k = m.kind(self._mem_kind)
        _, step = nice_ticks(self.view.t0, self.view.t1)
        fmt = lambda t: format_time(t, step / 100)
        if recompute:
            if self._mem_range is None:
                mask = memory.alive_mask(k, self._mem_time)
            else:
                mask = memory.growth_mask(k, *self._mem_range)
            self._mem_incl = memory.inclusive_bytes(m, memory.self_bytes(m, k, mask))
        incl = self._mem_incl
        total = float(incl[0])

        self.mem_title.ui_children = [f"Memory {self._mem_kind}"]
        t_peak, v_peak = k.peak()
        self.mem_peak.ui_children = [f"{format_bytes(v_peak)} @ {fmt(t_peak)}"]
        if self._mem_range is None:
            t = self._mem_time
            self.mem_mode.ui_children = [f"memory alive at {fmt(t)}"]
            self.mem_hero_label.ui_children = ["allocated at click time, relative to trace start"]
            self.mem_hero.ui_children = [format_bytes(k.value_at(t))]
        else:
            t0, t1 = self._mem_range
            self.mem_mode.ui_children = [f"memory growth in [{fmt(t0)}, {fmt(t1)}]"]
            self.mem_hero_label.ui_children = [
                "allocated inside the range and still alive at its end"
            ]
            self.mem_hero.ui_children = [format_bytes(total)]

        focus = self._mem_focus
        path = m.path(focus)
        crumb = " › ".join(_short(self._stack_name(s), 40) for s in path) or "all stacks"
        self.mem_crumb.ui_children = [crumb]
        self.mem_crumb._props["title"] = " › ".join(self._stack_name(s) for s in path)

        self.mem_svg.ui_children = self._sunburst_children(m, incl, focus, total)
        f_val = float(incl[focus])
        self.mem_center.ui_children = [
            Div(format_bytes(f_val), ui_style="font-weight:600; color:var(--fg);"),
            Div(f"{100 * f_val / total:.1f}%" if total > 0 else "–"),
        ]
        self._render_mem_list(m, incl, focus, total)

    def _sunburst_children(self, m, incl, focus, total):
        cx = cy = 200.0
        r_in, r_out = 46.0, 196.0
        ring_w = (r_out - r_in) / SUN_RINGS
        ref = float(incl[focus])
        items = []
        f_col = _hex(self._stack_color(focus)) if focus else "var(--border-strong)"
        centre = _svg(
            "circle", cx=cx, cy=cy, r=r_in - 3,
            style=f"fill:{f_col}; fill-opacity:{0.35 if focus else 0.5}; stroke:var(--panel); stroke-width:2;",
        )
        centre._props["data-tip"] = (
            f"{self._stack_name(focus)}\n{format_bytes(ref)}"
            + (" · click: one level up" if focus else "")
        )
        centre.on("click", lambda ev: self._mem_back())
        items.append(centre)
        if ref <= 0:
            return items
        for a in memory.sunburst(m, incl, focus, rings=SUN_RINGS):
            r0 = r_in + (a.level - 1) * ring_w
            d = _arc_path(cx, cy, r0, r0 + ring_w, a.a0, a.a1)
            share = f"{format_bytes(a.value)} · {100 * a.value / ref:.1f}%"
            fill, opacity = _hex(self._stack_color(a.sid)), 1.0
            if a.own:
                label = self._own_label(a.sid)
                fill, opacity = ("var(--fg-subtle)", 0.3) if a.sid == 0 else (fill, 0.3)
                tip = f"{label}\n{share}"
            elif a.other:
                fill = "var(--border-strong)"
                tip = f"{a.other} smaller stacks\n{share}"
            else:
                tip = f"{self._stack_name(a.sid)}\n{share}\nclick: zoom in"
            p = _svg("path", d=d, style=(
                f"fill:{fill}; fill-opacity:{opacity}; stroke:var(--panel); stroke-width:2;"
            ))
            p._props["data-tip"] = tip
            if a.own or a.other:
                p.ui_class = str(style.mem_sun_static)
            else:
                p.on("click", lambda ev, sid=a.sid: self._mem_zoom(sid))
            items.append(p)
        return items

    def _render_mem_list(self, m, incl, focus, total):
        order, start = memory.children_lists(m)
        kids = order[start[focus] : start[focus + 1]]
        kids = kids[incl[kids] > 0]
        kids = kids[np.argsort(-incl[kids], kind="stable")]
        ref = float(incl[focus])
        rows = []
        own = ref - float(incl[kids].sum())
        if own > 0.5 and ref > 0:
            label = self._own_label(focus)
            rows.append(Div(
                _swatch((0.55, 0.55, 0.55)),
                Div(label, ui_class=str(style.mem_list_name), ui_style="color:var(--fg-muted);"),
                Div(f"{format_bytes(own)} · {100 * own / ref:.1f}%", ui_class=str(style.mem_list_val)),
                ui_class=str(style.mem_list_row), ui_style="cursor:default;",
            ))
        for sid in kids[:SUN_LIST]:
            v = float(incl[sid])
            name = self._stack_name(int(sid))
            row = Div(
                _swatch(self._stack_color(int(sid))),
                Div(name, ui_class=str(style.mem_list_name)),
                Div(f"{format_bytes(v)} · {100 * v / ref:.1f}%", ui_class=str(style.mem_list_val)),
                ui_class=str(style.mem_list_row),
            )
            row._props["title"] = name
            row.on("click", lambda ev, sid=int(sid): self._mem_zoom(sid))
            rows.append(row)
        rest = len(kids) - SUN_LIST
        if rest > 0:
            rows.append(Div(f"… {rest} more stacks", ui_class=str(style.mem_list_val),
                            ui_style="padding:3px 6px;"))
        if ref <= 0:
            rows = [Div("no allocations", ui_class=str(style.mem_list_val),
                        ui_style="padding:3px 6px;")]
        self.mem_list_head.ui_children = [
            f"stacks below {'root' if focus == 0 else 'focus'}  ·  {len(kids)}"
        ]
        self.mem_list.ui_children = rows

    def _cancel_hide(self):
        t = self._hide_timer
        self._hide_timer = None
        if t is not None:
            t.cancel()

    def _schedule_hide(self):
        self._cancel_hide()
        t = threading.Timer(0.09, self._hide_now)
        t.daemon = True
        self._hide_timer = t
        t.start()

    def _hide_now(self):
        self._shown_pick = None
        self.tooltip.ui_style = "display:none;"

    def _on_pick(self, sel_ev):
        trace = self.trace
        idx = int(sel_ev.uint32[0])
        if trace is None or idx >= trace.n_intervals or self.view is None:
            self._schedule_hide()
            return
        self._cancel_hide()
        # only rebuild content when we move onto a *different* interval — this
        # is what keeps the tooltip from flickering as the mouse moves within
        # one block (each move otherwise tore down and rebuilt the DOM)
        if idx != self._shown_pick:
            self._shown_pick = idx
            value = trace.value[idx]
            name = trace.names[value]
            start, end = trace.start[idx], trace.end[idx]
            row_name = trace.rows[trace.row[idx]].name
            _, step = nice_ticks(self.view.t0, self.view.t1)
            self._tt_title.ui_children = [name]
            self._tt_sub.ui_children = [
                _swatch(trace.colors[value]),
                Div(
                    f"{format_duration(end - start)}  ·  {row_name}  ·  "
                    f"@ {format_time(start, step / 100)}"
                ),
            ]
            self.status.ui_children = [
                f"{row_name}:  {format_duration(end - start)}  —  {name[:140]}"
            ]
        self._place_tooltip()

    def _place_tooltip(self):
        # reposition every move (cheap single-node style update, no rebuild).
        # Flip above / left of the cursor near the bottom / right edges so the
        # tooltip is never clipped by the canvas area (e.g. on the last rows).
        canvas = self.view.scene.canvas
        dpr = canvas.dpr or 1
        w, h = canvas.width / dpr, canvas.height / dpr
        x, y = self._hover_px[0] / dpr, self._hover_px[1] / dpr
        if x > w * 0.62:
            left, tx = x - 14, "-100%"
        else:
            left, tx = x + 14, "0"
        if y > h * 0.6:
            top, ty = y - 14, "-100%"
        else:
            top, ty = y + 14, "0"
        self.tooltip.ui_style = (
            f"display:block; left:{left:.0f}px; top:{top:.0f}px; "
            f"transform: translate({tx}, {ty});"
        )

    def _on_pick_background(self, sel_ev):
        self._schedule_hide()

    def _on_click(self, ev):
        """Left-click a block: show its info. If a double-click highlight is
        active, clicking again removes it (toggle off)."""
        if ev.get("button", 0) != 0:  # right-click is handled as "go back"
            return
        kind = self._memory_kind_at(ev.get("canvasY", -1))
        if kind is not None:
            self._open_memory(kind, t=self.view.time_at(ev["canvasX"]))
            return
        if self._click_highlight is not None:
            self._click_highlight = None
            self._set_highlight(None)
            self._sync_stats_selection(None)
        trace = self.trace
        idx = self._shown_pick
        if trace is None or idx is None:
            self._show_detail_empty()
            return
        self._show_detail(int(trace.value[idx]), idx)

    def _show_detail(self, value, idx):
        import numpy as np

        trace = self.trace
        name = trace.names[value]
        start, end = trace.start[idx], trace.end[idx]
        mask = trace.value == value
        count = int(mask.sum())
        total = float((trace.end - trace.start)[mask].sum())
        span = max(trace.tmax - trace.tmin, 1e-12)
        # "busy": union of this function's intervals (parallel/nested-safe)
        sm, em = trace.start[mask], trace.end[mask]
        order = np.argsort(sm)
        s, e = sm[order], em[order]
        run = np.maximum.accumulate(e)
        prev = np.empty_like(run)
        prev[0] = -np.inf
        prev[1:] = run[:-1]
        busy = float(np.maximum(0.0, e - np.maximum(s, prev)).sum())
        _, step = nice_ticks(self.view.t0, self.view.t1)
        close = QBtn(
            QTooltip("Clear highlight"),
            ui_icon="close", ui_flat=True, ui_dense=True, ui_round=True, ui_size="sm",
        )
        close.on_click(self._clear_highlight)
        self.detail.ui_children = [
            _swatch(trace.colors[value]),
            Div(name, ui_class=str(style.detail_name)),
            Div(
                f"this call {format_duration(end - start)} @ {format_time(start, step / 100)}"
                f"   ·   {count:,} calls · {format_duration(total)} total · "
                f"busy {100 * busy / span:.1f}% of the trace",
                ui_class=str(style.detail_meta),
            ),
            close,
        ]

    def _show_detail_empty(self):
        if self.trace is None:
            self.detail.ui_children = []
            return
        self.detail.ui_children = [
            Div("Click a task to inspect it · double-click to highlight it",
                ui_class=str(style.detail_hint))
        ]

    # ---- statistics panel ----

    def _toggle_stats(self, *_):
        # drive the splitter: 0 = closed, remembered width = open; from the
        # memory panel the key switches back to the statistics instead
        if self._stats_open and self._side_mode == "memory":
            self._close_memory()
            return
        opening = not self._stats_open
        self.splitter.ui_model_value = self._stats_width if opening else 0
        self._stats_open = opening
        if opening:
            self._refresh_stats()

    def _show_side(self, mode):
        self._side_mode = mode
        self.stats_panel.ui_hidden = mode != "stats"
        self.mem_panel.ui_hidden = mode != "memory"
        if not self._stats_open:
            self.splitter.ui_model_value = self._stats_width
            self._stats_open = True
        if mode == "stats":
            self._refresh_stats()

    def _on_splitter(self, ev):
        val = ev.value if hasattr(ev, "value") else ev
        try:
            val = float(val)
        except (TypeError, ValueError):
            return
        was_open = self._stats_open
        self._stats_open = val > 24
        if self._stats_open:
            self._stats_width = val
        if self._stats_open and not was_open:
            self._refresh_stats()

    def _on_stats_mode(self, ev):
        self._stats_mode = ev.value if hasattr(ev, "value") else ev
        self._refresh_stats()

    def _on_view_changed(self):
        if self._stats_open and self._stats_mode == "view":
            self._schedule_stats_refresh()

    def _schedule_stats_refresh(self):
        t = self._stats_timer
        if t is not None:
            t.cancel()
        t = threading.Timer(0.25, self._refresh_stats)
        t.daemon = True
        self._stats_timer = t
        t.start()

    def _refresh_stats(self):
        if self.trace is None or not self._stats_open or self._side_mode != "stats":
            return
        from . import stats

        if self._stats_mode == "view" and self.view is not None:
            data = stats.compute(self.trace, self.view.t0, self.view.t1)
        else:
            data = stats.compute(self.trace)
        rows = []
        for s in data:
            name = s.name if len(s.name) <= NAME_MAX else s.name[: NAME_MAX - 1] + "…"
            rows.append(
                {
                    "id": str(s.value),
                    "value": s.value,
                    "name": name,
                    "count": s.count,
                    "total": round(s.total / 1000.0, 4),
                    "pct": round(s.percent, 2),
                    "mean": round(s.mean, 4),
                    "max": round(s.max, 4),
                }
            )
        self.stats_table.ui_rows = rows

    def _on_stats_select(self, ev):
        selected = ev.value if hasattr(ev, "value") else ev
        selected = selected or []
        self.stats_table.ui_selected = selected
        self._click_highlight = None  # table selection drives the highlight
        values = [int(r["value"]) for r in selected]
        self._set_highlight(values if values else None)

    def _sync_stats_selection(self, value):
        """Reflect a timeline highlight in the stats table selection."""
        if not self._stats_open:
            return
        if value is None:
            self.stats_table.ui_selected = []
        else:
            for row in self.stats_table.ui_rows:
                if row.get("value") == value:
                    self.stats_table.ui_selected = [row]
                    return
            self.stats_table.ui_selected = []

    # ---- overlays (time axis + row labels) ----

    def _row_label(self, r):
        row = self.trace.rows[r]
        if row.kind == "memory" and self.trace.memory is not None:
            k = self.trace.memory.kind_of_row(r)
            _, v_peak = self.trace.memory.kind(k).peak()
            return f"{row.name}  ▲ {format_bytes(v_peak)}"
        return row.name

    def _update_mem_mark(self):
        view = self.view
        if self._mem_kind is None or view is None:
            self.mem_mark.ui_style = "display:none;"
            return
        if self._mem_range is None:
            f = (self._mem_time - view.t0) / view.span
            self.mem_mark.ui_style = (
                f"display:block; left:{f * 100:.4f}%; width:0; border-right:none;"
            )
        else:
            t0, t1 = self._mem_range
            f0 = (t0 - view.t0) / view.span
            f1 = (t1 - view.t0) / view.span
            self.mem_mark.ui_style = (
                f"display:block; left:{f0 * 100:.4f}%; width:{(f1 - f0) * 100:.4f}%;"
            )

    def _build_overlay_pools(self):
        self._tick_pool = [Div(ui_style="display:none;") for _ in range(MAX_TICKS + 2)]
        self._axis.ui_children = list(self._tick_pool)
        self._label_pool = [Div(ui_style="display:none;") for _ in self.trace.rows]
        self._labels.ui_children = list(self._label_pool)

    def _update_overlays(self):
        view = self.view
        if view is None or view.scene is None or view.scene.canvas is None:
            return
        # overlay divs are positioned in CSS pixels as fractions of the canvas;
        # the device-pixel-ratio cancels out for relative positions
        ticks, step = nice_ticks(view.t0, view.t1)
        base = str(style.tick)
        for i, div in enumerate(self._tick_pool):
            if i < len(ticks):
                frac = (ticks[i] - view.t0) / view.span
                div.ui_children = [format_time(ticks[i], step)]
                div.ui_class = base
                div.ui_style = f"left:{frac * 100:.4f}%;"
            else:
                div.ui_style = "display:none;"

        self._update_mem_mark()

        row_h = 1.0 / view.rows_visible
        base_lbl = str(style.label_row)
        for r, div in enumerate(self._label_pool):
            top = (r - view.y0) * row_h
            if -row_h < top < 1.0:
                div.ui_children = [self._row_label(r)]
                div.ui_class = base_lbl
                div.ui_style = f"top:{top * 100:.4f}%; height:{row_h * 100:.4f}%;"
            else:
                div.ui_style = "display:none;"
