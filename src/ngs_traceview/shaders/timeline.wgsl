// Instanced rectangle renderer for trace timeline intervals.
//
// Times are in milliseconds, split into hi/lo float32 pairs on the CPU
// (t = hi + lo with |lo| <= ulp(hi)/2). The view offset is split the same
// way, so (hi - off_hi) + (lo - off_lo) recovers the time relative to the
// left view edge with ~picosecond precision even at microsecond zoom
// levels over a multi-minute trace.

struct TimelineUniforms {
  off_hi: f32,   // view left edge [ms], hi part
  off_lo: f32,   // view left edge [ms], lo part
  scale: f32,    // NDC units per ms (2.0 / visible ms)
  y_off: f32,    // vertical scroll offset in row units
  y_scale: f32,  // NDC units per row unit
  min_w: f32,    // minimum rectangle width in NDC units (~1 px)
  canvas_w: f32, // canvas size in device pixels
  canvas_h: f32,
  highlight: u32, // entity-value id to highlight (0xffffffff = none)
  // three scalar u32 pads (NOT vec3<u32> — that has 16-byte alignment and
  // would round the struct up to 64 bytes, mismatching the 48-byte host buffer)
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(60) var<uniform> u_view : TimelineUniforms;

struct VertexIn {
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
  @location(0) rect: vec4f,   // t_hi, t_lo, duration [ms], y_top [row units]
  @location(1) h: f32,        // height [row units]
  @location(2) color: vec4f,  // unorm8x4
  @location(3) flags: u32,    // bits 0-7: nesting depth; bits 8-31: value id
};

struct VertexOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
  @location(1) @interpolate(flat) instance: u32,
  @location(2) uv: vec2f,                          // 0..1 inside the rect
  @location(3) @interpolate(flat) size_px: vec2f,  // rect size in pixels
  @location(4) @interpolate(flat) value: u32,      // entity-value id
};

@vertex
fn vertex_timeline(in: VertexIn) -> VertexOut {
  let t_rel = (in.rect.x - u_view.off_hi) + (in.rect.y - u_view.off_lo);
  let w = max(in.rect.z * u_view.scale, u_view.min_w);
  let hn = in.h * u_view.y_scale;
  let x0 = t_rel * u_view.scale - 1.0;
  let y0 = 1.0 - (in.rect.w - u_view.y_off) * u_view.y_scale;

  let corner = vec2f(f32(in.vi & 1u), f32(in.vi >> 1u));
  let x = x0 + corner.x * w;
  let y = y0 - corner.y * hn;

  // nested states get a smaller z so they always paint on top
  let depth = f32(in.flags & 0xffu);
  let z = 0.9 - 0.1 * depth;

  var out: VertexOut;
  out.pos = vec4f(x, y, z, 1.0);
  out.color = in.color;
  if(in.vi % 2 == 1u) {
    out.color = vec4f(0.55 * in.color.xyz, in.color.w);
  }
  out.instance = in.ii;
  out.uv = corner;
  out.size_px = vec2f(w * u_view.canvas_w * 0.5, hn * u_view.canvas_h * 0.5);
  out.value = in.flags >> 8u;
  return out;
}

@fragment
fn fragment_timeline(in: VertexOut) -> @location(0) vec4f {
  var rgb = in.color.rgb;
  // darken a 1px border when the rectangle is large enough to afford one
  if (in.size_px.x > 5.0 && in.size_px.y > 5.0) {
    let px = in.uv * in.size_px;
    let d = min(min(px.x, in.size_px.x - px.x), min(px.y, in.size_px.y - px.y));
    if (d < 1.0) {
      rgb *= 0.55;
    }
  }
  // when a function is highlighted, fade everything else toward neutral grey
  if (u_view.highlight != 0xffffffffu && in.value != u_view.highlight) {
    let lum = dot(rgb, vec3f(0.299, 0.587, 0.114));
    rgb = mix(vec3f(lum), rgb, 0.15);
    rgb = mix(rgb, vec3f(0.6), 0.6);
  }
  return vec4f(rgb, 1.0);
}

#ifdef SELECT_PIPELINE
@fragment
fn fragment_select_timeline(in: VertexOut) -> @location(0) vec4<u32> {
  return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(in.pos.z), in.instance, 0u);
}
#endif SELECT_PIPELINE
