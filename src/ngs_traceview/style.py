"""Trace-viewer styling.

Reuses the CerbSIM design system from :mod:`ngsolve_gui.cerbsim_style` (design
tokens as CSS custom properties, IBM Plex fonts, Quasar brand-colour mapping,
light/dark themes) and adds only the handful of classes the trace viewer needs
that have no Quasar equivalent: the positioned time-axis / row-label overlays,
the hover tooltip, and the statistics side panel. Interactive chrome (toolbar,
buttons, the statistics table) uses Quasar components directly.
"""

from ngapp.style import Style, StyleSheet
from ngsolve_gui import cerbsim_style as cb

# geometry shared with app.py
LABEL_WIDTH = 176  # px, row-label column
AXIS_HEIGHT = 26  # px, time-axis strip

css = StyleSheet(prefix="tv")


def _cls(name, **props):
    return css.add(Style(**props), name=name)


def install(app, default_theme="system"):
    """Install the CerbSIM theme + trace-viewer sheet. Call once after super().__init__."""
    cb.install(app, default_theme=default_theme)
    css.inject(app)


def resolved_theme(default="system"):
    """Concrete 'dark'/'light' for the current OS (falls back to 'light')."""
    return cb._resolve(default) if default == "system" else default


# ── shell ────────────────────────────────────────────────────────────────────
page = _cls(
    "tv-page", height="100vh", width="100vw", overflow="hidden",
    display="flex", flex_direction="column",
    background="var(--bg)", color="var(--fg)", font_family="var(--font-sans)",
)
mid = _cls("tv-mid", position="relative", display="flex", flex="1",
           min_height="0", overflow="hidden")
timeline_col = _cls(
    "tv-tcol", display="flex", flex_direction="column", flex="1",
    min_width="0", min_height="0",
)
# QSplitter panes default to overflow:auto (a stray scrollbar) and don't force
# their child to fill the height — make them flex containers that clip, so the
# canvas / stats table fill the pane exactly and manage their own scrolling.
css.add_rule(".tv-mid .q-splitter, .tv-mid .q-splitter__panel", Style(min_height="0"))
css.add_rule(".tv-mid .q-splitter__panel", Style(overflow="hidden", display="flex"))

# ── top bar ──────────────────────────────────────────────────────────────────
bar = _cls(
    "tv-bar", display="flex", align_items="center", gap="10px", height="52px",
    padding="0 12px", flex="none",
    background="var(--panel-header)", border_bottom="1px solid var(--border)",
)
brand = _cls("tv-brand", display="flex", flex_direction="column", line_height="1.15",
             padding_right="6px")
brand_name = _cls("tv-brand-name", font_size="15px", font_weight="700",
                  letter_spacing="var(--ls-snug)", color="var(--fg)")
brand_sub = _cls("tv-brand-sub", font_size="9px", letter_spacing="0.08em",
                 text_transform="uppercase", color="var(--fg-subtle)")
sep = _cls("tv-sep", width="1px", height="26px", background="var(--border)", margin="0 2px")
info = _cls("tv-info", font_family="var(--font-mono)", font_size="11.5px",
            color="var(--fg-subtle)", white_space="nowrap")

# ── time axis ────────────────────────────────────────────────────────────────
axis = _cls(
    "tv-axis", position="relative", height=f"{AXIS_HEIGHT}px", flex="none",
    overflow="hidden", margin_left=f"{LABEL_WIDTH}px",
    background="var(--panel-header)", border_bottom="1px solid var(--border)",
)
tick = _cls(
    "tv-tick", position="absolute", bottom="4px", transform="translateX(-50%)",
    white_space="nowrap", padding_left="5px",
    border_left="1px solid var(--border-strong)",
    font_family="var(--font-mono)", font_size="10px", color="var(--fg-muted)",
)

# ── body: labels + canvas ────────────────────────────────────────────────────
body = _cls("tv-body", display="flex", flex="1", min_height="0", overflow="hidden")
labels = _cls(
    "tv-labels", position="relative", width=f"{LABEL_WIDTH}px", flex="none",
    overflow="hidden", background="var(--panel)",
    border_right="1px solid var(--border)",
)
label_row = _cls(
    "tv-label", position="absolute", right="8px", display="flex",
    align_items="center", justify_content="flex-end", white_space="nowrap",
    overflow="hidden", text_overflow="ellipsis",
    font_size="11.5px", color="var(--fg-muted)",
)
label_row.rule(".tv-swatch", margin_left="6px")
canvas_wrap = _cls(
    "tv-canvas", position="relative", flex="1", min_width="0",
    overflow="hidden", background="var(--viewport)", cursor="crosshair",
)

# rubber-band time selection box (drag to zoom to a time range)
sel_box = _cls(
    "tv-sel", position="absolute", top="0", bottom="0", z_index="40",
    pointer_events="none",
    background="color-mix(in srgb, var(--accent) 20%, transparent)",
    border_left="1px solid var(--accent)", border_right="1px solid var(--accent)",
)

# ── hover tooltip ────────────────────────────────────────────────────────────
tooltip = _cls(
    "tv-tip", position="absolute", z_index="60", pointer_events="none",
    max_width="460px", padding="7px 10px",
    background="var(--surface)", color="var(--fg)",
    border="1px solid var(--border-strong)", border_radius="var(--r-md)",
    box_shadow="var(--shadow-pop)",
)
tip_title = _cls("tv-tip-title", font_family="var(--font-mono)", font_size="11.5px",
                 font_weight="600", color="var(--fg)", overflow_wrap="anywhere")
tip_sub = _cls("tv-tip-sub", margin_top="3px", font_size="11.5px", color="var(--fg-muted)",
               display="flex", align_items="center", gap="7px")

# ── swatch (color chip) ──────────────────────────────────────────────────────
swatch = _cls(
    "tv-swatch", display="inline-block", width="11px", height="11px",
    flex="none", border_radius="var(--r-xs)", border="1px solid rgba(0,0,0,.25)",
)

# ── detail banner ────────────────────────────────────────────────────────────
detail = _cls(
    "tv-detail", display="flex", align_items="center", gap="12px", flex="none",
    padding="8px 12px", background="var(--panel)",
    border_top="1px solid var(--border)",
    font_size="12px", color="var(--fg)",
)
detail_name = _cls("tv-detail-name", font_family="var(--font-mono)", font_size="11px",
                   color="var(--fg)", overflow_wrap="anywhere", flex="1", min_width="0")
detail_meta = _cls("tv-detail-meta", font_family="var(--font-mono)", font_size="11.5px",
                   color="var(--fg-muted)", white_space="nowrap")

# ── status bar ───────────────────────────────────────────────────────────────
status = _cls(
    "tv-status", display="flex", align_items="center", gap="8px", flex="none",
    height="24px", padding="0 12px",
    background="var(--panel-header)", border_top="1px solid var(--border)",
    font_family="var(--font-mono)", font_size="11px", color="var(--fg-subtle)",
    white_space="nowrap", overflow="hidden",
)

# ── loading overlay ──────────────────────────────────────────────────────────
overlay = _cls(
    "tv-overlay", position="absolute", top="0", left="0", right="0", bottom="0",
    z_index="80", display="flex", align_items="center", justify_content="center",
    background="var(--overlay)", backdrop_filter="blur(1.5px)",
)
load_card = _cls(
    "tv-card", display="flex", flex_direction="column", align_items="center",
    gap="16px", min_width="320px", padding="26px 32px",
    background="var(--surface)", color="var(--fg)",
    border="1px solid var(--border)", border_radius="var(--r-lg)",
    box_shadow="var(--shadow-pop)",
)
load_msg = _cls("tv-load-msg", font_size="13px", color="var(--fg-muted)",
                font_family="var(--font-mono)")
load_title = _cls("tv-load-title", font_size="15px", font_weight="600", color="var(--fg)")
load_bar = _cls("tv-load-bar", width="260px")

# ── statistics panel ─────────────────────────────────────────────────────────
# fills the QSplitter "after" pane (the splitter controls the width)
stats = _cls(
    "tv-stats", width="100%", height="100%", display="flex",
    flex_direction="column", min_height="0", overflow="hidden",
    background="var(--panel)", border_left="1px solid var(--border)",
)
stats_head = _cls(
    "tv-stats-head", display="flex", align_items="center", gap="8px", flex="none",
    height="40px", padding="0 6px 0 12px",
    background="var(--panel-header)", border_bottom="1px solid var(--border)",
)
stats_title = _cls(
    "tv-stats-title", font_size="11px", font_weight="700", letter_spacing="0.07em",
    text_transform="uppercase", color="var(--fg-muted)",
)
stats_body = _cls("tv-stats-body", position="relative", flex="1", min_height="0")
# QTable inside the panel: fill height, compact, themed
css.add_rule(".tv-stats-body .q-table__container", Style(height="100%"))
css.add_rule(".tv-stats-body .q-table th",
             Style(font_weight="600", color="var(--fg-muted)", white_space="nowrap"))
css.add_rule(".tv-stats-body .q-table tbody td",
             Style(font_family="var(--font-mono)", font_size="11.5px"))
css.add_rule(".tv-stats-body .q-table tbody tr",
             Style(cursor="pointer"))
# clamp the (long C++ symbol) Function column so the numeric columns stay visible
css.add_rule(".tv-stats-body .tv-fn",
             Style(max_width="180px", overflow="hidden", text_overflow="ellipsis",
                   white_space="nowrap"))
css.add_rule(".tv-stats-body .q-table td:first-child, .tv-stats-body .q-table th:first-child",
             Style(padding_left="6px", padding_right="4px"))
css.add_rule(".tv-stats-body .q-table td, .tv-stats-body .q-table th",
             Style(padding_left="6px", padding_right="6px"))
