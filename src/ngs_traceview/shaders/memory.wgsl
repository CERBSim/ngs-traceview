// Step-curve renderer for the memory rows: one instance per curve step,
// drawn as three quads (area wash to the baseline, horizontal 2px line,
// vertical connector to the next level). Time uses the same hi/lo float32
// split as timeline.wgsl so both renderers share one view transform.

struct MemoryUniforms {
  off_hi: f32,
  off_lo: f32,
  scale: f32,
  y_off: f32,
  y_scale: f32,
  min_w: f32,
  canvas_w: f32,
  canvas_h: f32,
  bg_r: f32,      // canvas clear color (the wash is pre-blended onto it)
  bg_g: f32,
  bg_b: f32,
  line_px: f32,   // line half-width in device pixels
  base0: f32,     // normalized position of 0 bytes per kind (0 = row bottom)
  base1: f32,
  row_pad: f32,
  _p0: f32,
  col0_r: f32,    // host curve
  col0_g: f32,
  col0_b: f32,
  _p1: f32,
  col1_r: f32,    // device curve
  col1_g: f32,
  col1_b: f32,
  _p2: f32,
  grid_r: f32,    // zero baseline
  grid_g: f32,
  grid_b: f32,
  _p3: f32,
};

@group(0) @binding(62) var<uniform> u_mem : MemoryUniforms;

struct VertexIn {
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
  @location(0) step: vec4f,   // t_hi, t_lo, duration [ms], level v0 (0..1 in row)
  @location(1) v1: f32,       // level after the step (0..1)
  @location(2) flags: u32,    // bits 0-7 row, bit 8 kind, bit 9 baseline
};

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
};

fn row_y(row: f32, v: f32) -> f32 {
  let inner = u_mem.row_pad + (1.0 - v) * (1.0 - 2.0 * u_mem.row_pad);
  return 1.0 - (row + inner - u_mem.y_off) * u_mem.y_scale;
}

@vertex
fn vertex_memory(in: VertexIn) -> VertexOut {
  let quad = in.vi / 6u;
  let idx = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[in.vi % 6u];
  let cx = f32(idx & 1u);
  let cy = f32(idx >> 1u);

  let row = f32(in.flags & 0xffu);
  let kind = (in.flags >> 8u) & 1u;
  let is_base = ((in.flags >> 9u) & 1u) == 1u;

  let t_rel = (in.step.x - u_mem.off_hi) + (in.step.y - u_mem.off_lo);
  let x0 = t_rel * u_mem.scale - 1.0;
  let x1 = x0 + max(in.step.z * u_mem.scale, u_mem.min_w);
  let base = select(u_mem.base0, u_mem.base1, kind == 1u);
  let yb = row_y(row, base);
  let ya = row_y(row, in.step.w);
  let yc = row_y(row, in.v1);
  let hp = u_mem.line_px * 2.0 / u_mem.canvas_h;
  let wp = u_mem.line_px * 2.0 / u_mem.canvas_w;

  var col = vec3f(u_mem.col0_r, u_mem.col0_g, u_mem.col0_b);
  if (kind == 1u) {
    col = vec3f(u_mem.col1_r, u_mem.col1_g, u_mem.col1_b);
  }
  let bg = vec3f(u_mem.bg_r, u_mem.bg_g, u_mem.bg_b);

  var x = x0;
  var y = ya;
  var z = 0.5;
  var rgb = col;
  if (quad == 0u) {          // area wash between the curve and the baseline
    x = mix(x0, x1, cx);
    y = mix(ya, yb, cy);
    z = 0.8;
    rgb = mix(bg, col, 0.14);
    if (is_base) { y = yb; }
  } else if (quad == 1u) {   // horizontal segment
    x = mix(x0, x1, cx);
    y = ya + (cy * 2.0 - 1.0) * hp;
    if (is_base) {
      rgb = vec3f(u_mem.grid_r, u_mem.grid_g, u_mem.grid_b);
      y = ya + (cy * 2.0 - 1.0) * hp * 0.5;
      z = 0.7;
    }
  } else {                   // vertical connector to the next level
    x = x1 + (cx * 2.0 - 1.0) * wp;
    y = mix(ya, yc, cy);
    if (is_base) { y = ya; }
  }

  var out: VertexOut;
  out.pos = vec4f(x, y, z, 1.0);
  out.color = vec4f(rgb, 1.0);
  return out;
}

@fragment
fn fragment_memory(in: VertexOut) -> @location(0) vec4f {
  return in.color;
}
