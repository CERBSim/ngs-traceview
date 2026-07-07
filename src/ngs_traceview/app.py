import math
import os
import re
import threading
import time

from ngapp.app import App
from ngapp.components import (
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

from . import style
from .style import AXIS_HEIGHT, LABEL_WIDTH

MAX_TICKS = 11
NAME_MAX = 90  # truncate long C++ symbols in the table


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
        canvas_wrap = Div(
            self.canvas, self.sel_box, self.tooltip, ui_class=str(style.canvas_wrap)
        )
        timeline_col = Div(
            self._axis,
            Div(self._labels, canvas_wrap, ui_class=str(style.body)),
            ui_class=str(style.timeline_col),
        )

        self.stats_panel = self._build_stats_panel()

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
            ui_slots={"before": [timeline_col], "after": [self.stats_panel]},
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
        self.info_label.ui_children = [
            f"{base}  ·  {trace.n_intervals:,} intervals  ·  "
            f"{len(trace.rows)} rows  ·  {trace.parse_time:.1f}s"
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
        from .timeline import TimelineRenderer, TimelineView

        self.renderer = TimelineRenderer(self.trace)
        self.view = TimelineView(self.renderer)
        # legacy (Python-driven) render path: the JS engine's built-in 3D
        # camera would consume drag/wheel, which we need for 2D pan/zoom
        scene = self.canvas.draw([self.renderer], use_js_engine=False)
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
        # plain left-drag = rubber-band time zoom (ViTE); shift or middle = pan
        pan = ev.get("shiftKey") or button == 1 or ev.get("buttons") == 4
        self._drag_mode = "pan" if pan else "select"
        self._sel_start = x
        self._sel_moved = False
        self._pan_pushed = False

    def _on_mouseup(self, ev):
        if (
            self.view is not None
            and self._drag_mode == "select"
            and self._sel_moved
        ):
            self._push_history()
            a = self.view.time_at(min(self._sel_start, ev["canvasX"]))
            b = self.view.time_at(max(self._sel_start, ev["canvasX"]))
            self.view.set_time_range(a, b)
            self.view.apply()
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
        elif self._drag_mode == "select":
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
        if self._drag_mode in ("pan", "select", "back") or ev.get("buttons"):
            return
        self._hover_px = (ev["canvasX"], ev["canvasY"])
        self.canvas.select(ev["canvasX"], ev["canvasY"])

    def _on_mouseout(self, ev):
        self._cancel_hide()
        self._hide_now()

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
        # drive the splitter: 0 = closed, remembered width = open
        opening = not self._stats_open
        self.splitter.ui_model_value = self._stats_width if opening else 0
        self._stats_open = opening
        if opening:
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
        if self.trace is None or not self._stats_open:
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

        row_h = 1.0 / view.rows_visible
        base_lbl = str(style.label_row)
        for r, div in enumerate(self._label_pool):
            top = (r - view.y0) * row_h
            if -row_h < top < 1.0:
                div.ui_children = [self.trace.rows[r].name]
                div.ui_class = base_lbl
                div.ui_style = f"top:{top * 100:.4f}%; height:{row_h * 100:.4f}%;"
            else:
                div.ui_style = "display:none;"
