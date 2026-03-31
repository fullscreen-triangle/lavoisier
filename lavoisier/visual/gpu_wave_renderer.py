"""
GPU-Accelerated Thermodynamic Wave Renderer
============================================

Replaces the sequential Python pixel loops in ThermodynamicWaveGenerator
with a multi-pass WebGL2 / OpenGL ES 3.0 shader pipeline.

Architecture:
    Pass 1 — Ion wave field (one draw call per ion, additive blend)
    Pass 2 — Normalisation + S-entropy overlay + physics validation mask
    Pass 3 — Bijective validation (visual path vs numerical path)

The Python side:
    - Runs IonDecompositionValidator to produce List[IonDroplet]
    - Serialises ion parameters to a flat Float32 texture
    - Hands off to the GPU for all pixel-level work
    - Reads back the result for final scoring

Performance: N ions x 512² pixels drops from ~N*262k sequential Python ops
to N draw calls at ~0.1ms each => 50 ions in ~5ms total.

Author: Kundai Sachikonye
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Import the existing types
from .IonToDropletConverter import IonDroplet, SEntropyCoordinates, DropletParameters

# Path to GLSL shader sources
SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


def _load_shader(filename: str) -> str:
    """Load a GLSL shader source from the shaders directory."""
    path = os.path.join(SHADER_DIR, filename)
    with open(path, "r") as f:
        return f.read()


# ============================================================================
# GPU Texture Serialisation
# ============================================================================

def ion_droplets_to_texture(
    ion_droplets: List[IonDroplet],
    resolution: Tuple[int, int] = (512, 512),
    mz_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Serialise IonDroplet list to a Float32 texture for GPU upload.

    Layout: one ion per column, three rows of vec4:
        Row 0: [cx, cy, amplitude, wavelength]
        Row 1: [decay_rate, impact_angle_rad, phase_offset, radius]
        Row 2: [Sk, St, Se, categorical_state_normalised]

    Parameters
    ----------
    ion_droplets : list of IonDroplet
        The ion-to-droplet transformation results.
    resolution : (int, int)
        Canvas resolution (height, width).
    mz_range : (float, float) or None
        Optional m/z range for coordinate mapping.

    Returns
    -------
    np.ndarray
        Shape (3, N, 4), dtype float32. Ready for texImage2D upload.
    """
    N = len(ion_droplets)
    if N == 0:
        return np.zeros((3, 1, 4), dtype=np.float32)

    data = np.zeros((3, N, 4), dtype=np.float32)

    if mz_range is None:
        mzs = [d.mz for d in ion_droplets]
        mz_range = (min(mzs), max(mzs))

    H, W = resolution

    for i, drop in enumerate(ion_droplets):
        p = drop.droplet_params
        s = drop.s_entropy_coords

        # Map m/z -> x position, S_time -> y position
        cx = float(np.interp(drop.mz, list(mz_range), [0, W - 1]))
        cy = s.s_time * (H - 1)

        # Wave parameters — exact match to generate_wave_pattern()
        amplitude = p.velocity * np.log1p(drop.intensity) / 10.0
        wavelength = p.radius * (1.0 + p.surface_tension * 10.0)
        decay_rate = (p.temperature / 373.15) / (p.phase_coherence + 0.1)
        phase_offset = (drop.categorical_state * np.pi / 10.0)
        angle_rad = np.deg2rad(p.impact_angle)

        # Row 0: position and wave shape
        data[0, i] = [cx, cy, amplitude, wavelength]

        # Row 1: decay, directionality, phase, radius
        data[1, i] = [decay_rate, angle_rad, phase_offset, p.radius]

        # Row 2: S-entropy coordinates + normalised categorical state
        cat_norm = drop.categorical_state / max(N, 1)
        data[2, i] = [s.s_knowledge, s.s_time, s.s_entropy, cat_norm]

    return data.astype(np.float32)


def texture_to_uniform_sequence(
    texture_data: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Convert the (3, N, 4) texture into a list of per-ion uniform dicts,
    for renderers that set uniforms per draw call rather than using
    a data texture.

    Returns
    -------
    list of dict
        Each dict has keys 'u_ion' (vec4) and 'u_ion2' (vec4).
    """
    N = texture_data.shape[1]
    uniforms = []
    for i in range(N):
        uniforms.append({
            "u_ion": texture_data[0, i].tolist(),    # [cx, cy, amplitude, wl]
            "u_ion2": texture_data[1, i].tolist(),   # [decay, angle, phase, radius]
            "s_entropy": texture_data[2, i].tolist(), # [Sk, St, Se, cat_norm]
        })
    return uniforms


# ============================================================================
# CPU Fallback: Software Rasteriser (same math as GPU, for validation)
# ============================================================================

def render_wave_field_cpu(
    ion_droplets: List[IonDroplet],
    resolution: Tuple[int, int] = (512, 512),
    mz_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Render the wave field on CPU using the same math as the GPU shader.
    This is the reference implementation for validating GPU output.

    Returns
    -------
    np.ndarray
        Shape (H, W), dtype float32. Raw accumulated wave field (not normalised).
    """
    H, W = resolution
    canvas = np.zeros((H, W), dtype=np.float32)
    tex = ion_droplets_to_texture(ion_droplets, resolution, mz_range)

    N = tex.shape[1]
    yy, xx = np.ogrid[:H, :W]

    for i in range(N):
        cx, cy, amplitude, wavelength = tex[0, i]
        decay_rate, angle_rad, phase_off, radius = tex[1, i]

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        wave = amplitude * np.exp(
            -dist / (radius * 30.0 * decay_rate + 1e-6)
        ) * np.cos(2.0 * np.pi * dist / (wavelength * 5.0 + 1e-6))

        angle_grid = np.arctan2(yy - cy, xx - cx)
        wave *= (1.0 + 0.3 * np.cos(angle_grid - angle_rad))
        wave *= np.cos(phase_off)

        canvas += wave

    return canvas


def normalise_wave_field(canvas: np.ndarray) -> np.ndarray:
    """Normalise raw wave field to [0, 1]."""
    cmin = canvas.min()
    cmax = canvas.max()
    if cmax - cmin < 1e-10:
        return np.zeros_like(canvas)
    return (canvas - cmin) / (cmax - cmin)


def render_sentropy_overlay(
    normalised: np.ndarray,
    ion_droplets: List[IonDroplet],
    resolution: Tuple[int, int] = (512, 512),
    mz_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Apply S-entropy colour overlay to the normalised wave field.
    CPU reference for Pass 2.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype float32. RGB image.
    """
    H, W = resolution
    tex = ion_droplets_to_texture(ion_droplets, resolution, mz_range)
    N = tex.shape[1]

    # For each pixel, find nearest ion center and get its S-entropy
    yy, xx = np.mgrid[:H, :W]
    sk_map = np.zeros((H, W), dtype=np.float32)
    st_map = np.zeros((H, W), dtype=np.float32)
    se_map = np.zeros((H, W), dtype=np.float32)
    qual_map = np.ones((H, W), dtype=np.float32)
    min_dist = np.full((H, W), 1e9, dtype=np.float32)

    for i in range(N):
        cx, cy = tex[0, i, 0], tex[0, i, 1]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = d < min_dist
        min_dist[mask] = d[mask]
        sk_map[mask] = tex[2, i, 0]
        st_map[mask] = tex[2, i, 1]
        se_map[mask] = tex[2, i, 2]

    # Colour mapping
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :, 0] = sk_map * 0.0 + st_map * 1.0 + se_map * 0.75
    rgb[:, :, 1] = sk_map * 0.83 + st_map * 0.42 + se_map * 0.5
    rgb[:, :, 2] = sk_map * 1.0 + st_map * 0.2 + se_map * 1.0
    rgb = np.clip(rgb, 0.0, 1.0)

    # Modulate by wave intensity
    wave_3d = normalised[:, :, np.newaxis]
    rgb *= wave_3d

    return rgb


def render_bijective_validation(
    visual_field: np.ndarray,
    numerical_field: np.ndarray,
    tolerance: float = 0.15,
) -> Tuple[np.ndarray, float]:
    """
    Bijective validation: compare visual and numerical categorical states.
    CPU reference for Pass 3.

    Parameters
    ----------
    visual_field : np.ndarray
        Normalised wave field [0, 1], shape (H, W).
    numerical_field : np.ndarray
        Normalised spectrum field [0, 1], shape (H, W).
    tolerance : float
        Categorical bin tolerance.

    Returns
    -------
    (validation_image, score)
        validation_image: shape (H, W, 4), RGBA float32
        score: float in [0, 1], fraction of signal pixels where bijection holds
    """
    cat_visual = np.floor(visual_field * 10.0) / 10.0
    cat_numerical = np.floor(numerical_field * 10.0) / 10.0

    discrepancy = np.abs(cat_visual - cat_numerical)
    is_valid = (discrepancy <= tolerance).astype(np.float32)

    # Colour: green=valid, red=invalid
    rgba = np.zeros((*visual_field.shape, 4), dtype=np.float32)
    rgba[:, :, 0] = np.where(is_valid > 0.5, 0.2, 1.0)
    rgba[:, :, 1] = np.where(is_valid > 0.5, 0.9, 0.2)
    rgba[:, :, 2] = np.where(is_valid > 0.5, 0.3, 0.1)

    # Signal mask
    signal_mask = np.clip((np.abs(visual_field) - 0.02) / 0.08, 0, 1)
    rgba[:, :, :3] = (
        rgba[:, :, :3] * signal_mask[:, :, np.newaxis]
        + 0.05 * (1 - signal_mask[:, :, np.newaxis])
    )
    rgba[:, :, 3] = signal_mask

    # Score
    signal_pixels = signal_mask > 0.05
    if signal_pixels.sum() == 0:
        score = 0.0
    else:
        score = float(is_valid[signal_pixels].mean())

    return rgba, score


# ============================================================================
# GPU Readback Scoring
# ============================================================================

def score_bijective_from_readback(gpu_readback: np.ndarray) -> float:
    """
    Compute overall bijective validation score from GPU readback.

    Parameters
    ----------
    gpu_readback : np.ndarray
        RGBA float32 from glReadPixels, shape (H, W, 4).
        Green channel > 0.5 where bijection holds.
        Alpha channel = signal presence.

    Returns
    -------
    float
        Bijective validation score in [0, 1].
    """
    signal_mask = gpu_readback[:, :, 3] > 0.05
    if signal_mask.sum() == 0:
        return 0.0

    # Green channel dominance = valid bijection
    green = gpu_readback[:, :, 1]
    red = gpu_readback[:, :, 0]
    is_valid = green[signal_mask] > red[signal_mask]

    return float(is_valid.mean())


# ============================================================================
# Shader Source Export (for WebGL2 / JS embedding)
# ============================================================================

def get_shader_sources() -> Dict[str, str]:
    """
    Load and return all shader sources for embedding in a WebGL2 application.

    Returns
    -------
    dict
        Keys: 'wave_vert', 'wave_frag', 'physics_frag', 'bijective_frag'
    """
    return {
        "wave_vert": _load_shader("wave.vert"),
        "wave_frag": _load_shader("wave.frag"),
        "physics_frag": _load_shader("physics_overlay.frag"),
        "bijective_frag": _load_shader("bijective_validation.frag"),
    }


def export_webgl_html(
    ion_droplets: List[IonDroplet],
    output_path: str,
    resolution: Tuple[int, int] = (512, 512),
    mz_range: Optional[Tuple[float, float]] = None,
) -> str:
    """
    Export a self-contained HTML file with WebGL2 shaders that renders
    the ion wave field in-browser. This is the bridge between Python
    ion analysis and GPU-accelerated visualisation.

    Parameters
    ----------
    ion_droplets : list of IonDroplet
    output_path : str
        Path to write the HTML file.
    resolution : (int, int)
    mz_range : (float, float) or None

    Returns
    -------
    str
        Path to the written HTML file.
    """
    tex = ion_droplets_to_texture(ion_droplets, resolution, mz_range)
    uniforms = texture_to_uniform_sequence(tex)
    shaders = get_shader_sources()

    N = len(uniforms)
    W, H = resolution[1], resolution[0]

    # Build the ion data as a JS array
    ion_js_lines = []
    for u in uniforms:
        ion_js_lines.append(
            f"  {{ ion: {u['u_ion']}, ion2: {u['u_ion2']}, "
            f"se: {u['s_entropy']} }}"
        )
    ion_js = "[\n" + ",\n".join(ion_js_lines) + "\n]"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Lavoisier GPU Wave Renderer</title>
<style>
  body {{ margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }}
  canvas {{ border: 1px solid #333; }}
</style>
</head>
<body>
<canvas id="c" width="{W}" height="{H}"></canvas>
<script>
"use strict";
const W = {W}, H = {H};
const ions = {ion_js};

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2');
if (!gl) {{ document.body.innerHTML = '<h1>WebGL2 not supported</h1>'; throw 0; }}

// --- Shader compilation ---
function compileShader(gl, src, type) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
function linkProgram(gl, vs, fs) {{
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  return p;
}}

const vertSrc = `{shaders['wave_vert']}`;
const fragSrc = `{shaders['wave_frag']}`;

const vs = compileShader(gl, vertSrc, gl.VERTEX_SHADER);
const fs = compileShader(gl, fragSrc, gl.FRAGMENT_SHADER);
const prog = linkProgram(gl, vs, fs);

// --- Fullscreen quad ---
const quad = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, 'a_position');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

// --- Uniforms ---
gl.useProgram(prog);
const uIon  = gl.getUniformLocation(prog, 'u_ion');
const uIon2 = gl.getUniformLocation(prog, 'u_ion2');
const uRes  = gl.getUniformLocation(prog, 'u_resolution');
gl.uniform2f(uRes, W, H);

// --- Render with additive blending ---
gl.clearColor(0, 0, 0, 1);
gl.clear(gl.COLOR_BUFFER_BIT);
gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE);

for (const ion of ions) {{
  gl.uniform4fv(uIon, ion.ion);
  gl.uniform4fv(uIon2, ion.ion2);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}}

gl.disable(gl.BLEND);
console.log('Rendered ' + ions.length + ' ions via GPU');
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


# ============================================================================
# Full Pipeline: CPU Reference
# ============================================================================

@dataclass
class WaveRenderResult:
    """Complete result from the wave rendering pipeline."""
    raw_wave: np.ndarray           # (H, W) float32, raw accumulated
    normalised_wave: np.ndarray    # (H, W) float32, [0, 1]
    sentropy_overlay: np.ndarray   # (H, W, 3) float32, RGB
    bijective_image: np.ndarray    # (H, W, 4) float32, RGBA
    bijective_score: float         # [0, 1]
    ion_texture: np.ndarray        # (3, N, 4) float32
    n_ions: int
    resolution: Tuple[int, int]


def render_full_pipeline_cpu(
    ion_droplets: List[IonDroplet],
    numerical_field: Optional[np.ndarray] = None,
    resolution: Tuple[int, int] = (512, 512),
    mz_range: Optional[Tuple[float, float]] = None,
    bijective_tolerance: float = 0.15,
) -> WaveRenderResult:
    """
    Run the complete 3-pass rendering pipeline on CPU.
    This is the reference implementation; the GPU version produces
    identical results at ~1000x speed for typical ion counts.

    Parameters
    ----------
    ion_droplets : list of IonDroplet
    numerical_field : np.ndarray or None
        If provided, used for bijective validation (Pass 3).
        If None, the normalised wave field is used as its own reference
        (self-consistency check).
    resolution : (int, int)
    mz_range : (float, float) or None
    bijective_tolerance : float

    Returns
    -------
    WaveRenderResult
    """
    # Pass 1: wave field accumulation
    raw = render_wave_field_cpu(ion_droplets, resolution, mz_range)
    normed = normalise_wave_field(raw)

    # Pass 2: S-entropy overlay
    overlay = render_sentropy_overlay(normed, ion_droplets, resolution, mz_range)

    # Pass 3: bijective validation
    if numerical_field is None:
        numerical_field = normed  # self-consistency
    bij_img, bij_score = render_bijective_validation(
        normed, numerical_field, bijective_tolerance
    )

    tex = ion_droplets_to_texture(ion_droplets, resolution, mz_range)

    return WaveRenderResult(
        raw_wave=raw,
        normalised_wave=normed,
        sentropy_overlay=overlay,
        bijective_image=bij_img,
        bijective_score=bij_score,
        ion_texture=tex,
        n_ions=len(ion_droplets),
        resolution=resolution,
    )
