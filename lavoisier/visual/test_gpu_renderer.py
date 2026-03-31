#!/usr/bin/env python3
"""
Test and validate the GPU wave renderer pipeline.
Runs the CPU reference implementation and exports a WebGL2 HTML.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lavoisier.visual.IonToDropletConverter import (
    IonDroplet, SEntropyCoordinates, DropletParameters
)
from lavoisier.visual.gpu_wave_renderer import (
    ion_droplets_to_texture,
    texture_to_uniform_sequence,
    render_wave_field_cpu,
    normalise_wave_field,
    render_sentropy_overlay,
    render_bijective_validation,
    render_full_pipeline_cpu,
    export_webgl_html,
    get_shader_sources,
)


def make_test_ions(n=20):
    """Create test IonDroplet list simulating an MS2 spectrum."""
    rng = np.random.RandomState(42)
    ions = []
    for i in range(n):
        mz = 100 + rng.uniform(0, 900)
        intensity = rng.uniform(100, 10000)
        sk = rng.uniform(0.3, 1.0)
        st = rng.uniform(0.1, 0.9)
        se = rng.uniform(0.0, 1.0)
        velocity = rng.uniform(0.5, 3.0)
        radius = rng.uniform(0.5, 5.0)
        surface_tension = rng.uniform(0.01, 0.1)
        angle = rng.uniform(0, 180)
        temp = rng.uniform(280, 400)
        coherence = rng.uniform(0.3, 1.0)

        ions.append(IonDroplet(
            mz=mz,
            intensity=intensity,
            s_entropy_coords=SEntropyCoordinates(sk, st, se),
            droplet_params=DropletParameters(
                velocity=velocity,
                radius=radius,
                surface_tension=surface_tension,
                impact_angle=angle,
                temperature=temp,
                phase_coherence=coherence,
            ),
            categorical_state=i,
            physics_quality=rng.uniform(0.5, 1.0),
        ))
    return ions


def main():
    print("=" * 60)
    print("GPU Wave Renderer — Validation Suite")
    print("=" * 60)

    # 1. Create test ions
    ions = make_test_ions(50)
    print(f"\n1. Created {len(ions)} test ions")

    # 2. Texture serialisation
    tex = ion_droplets_to_texture(ions)
    print(f"2. Ion texture shape: {tex.shape} (3 rows x {tex.shape[1]} ions x 4 channels)")
    assert tex.shape == (3, 50, 4), f"Expected (3, 50, 4), got {tex.shape}"
    print("   Texture serialisation: PASS")

    # 3. Uniform sequence
    uniforms = texture_to_uniform_sequence(tex)
    assert len(uniforms) == 50
    assert len(uniforms[0]["u_ion"]) == 4
    print(f"3. Uniform sequence: {len(uniforms)} entries, PASS")

    # 4. CPU wave field rendering
    t0 = time.perf_counter()
    raw = render_wave_field_cpu(ions)
    t_render = time.perf_counter() - t0
    print(f"4. CPU wave field: shape={raw.shape}, range=[{raw.min():.4f}, {raw.max():.4f}]")
    print(f"   Render time: {t_render * 1000:.1f}ms for {len(ions)} ions")

    # 5. Normalisation
    normed = normalise_wave_field(raw)
    assert normed.min() >= 0.0 and normed.max() <= 1.0
    print(f"5. Normalised field: range=[{normed.min():.4f}, {normed.max():.4f}] PASS")

    # 6. S-entropy overlay
    overlay = render_sentropy_overlay(normed, ions)
    assert overlay.shape == (512, 512, 3)
    print(f"6. S-entropy overlay: shape={overlay.shape}, PASS")

    # 7. Bijective validation (self-consistency)
    bij_img, bij_score = render_bijective_validation(normed, normed)
    print(f"7. Bijective validation (self): score={bij_score:.4f}")
    assert bij_score > 0.99, f"Self-consistency should be ~1.0, got {bij_score}"
    print("   Self-consistency: PASS")

    # 8. Bijective validation (with noise)
    noisy = normed + np.random.RandomState(0).uniform(-0.2, 0.2, normed.shape).astype(np.float32)
    noisy = np.clip(noisy, 0, 1)
    bij_img2, bij_score2 = render_bijective_validation(normed, noisy)
    print(f"8. Bijective validation (noisy): score={bij_score2:.4f}")
    assert bij_score2 < bij_score, "Noisy should score lower"
    print("   Noise degradation: PASS")

    # 9. Full pipeline
    t0 = time.perf_counter()
    result = render_full_pipeline_cpu(ions)
    t_full = time.perf_counter() - t0
    print(f"9. Full pipeline: {t_full * 1000:.1f}ms, bij_score={result.bijective_score:.4f}")
    print(f"   n_ions={result.n_ions}, resolution={result.resolution}")

    # 10. Shader sources
    shaders = get_shader_sources()
    for name, src in shaders.items():
        print(f"10. Shader '{name}': {len(src)} chars, {src.count(chr(10))} lines")

    # 11. WebGL HTML export
    html_path = os.path.join(os.path.dirname(__file__), "test_wave_render.html")
    export_webgl_html(ions, html_path)
    print(f"11. WebGL HTML exported: {html_path}")
    assert os.path.exists(html_path)
    fsize = os.path.getsize(html_path)
    print(f"    File size: {fsize} bytes")

    # 12. Performance estimate
    gpu_est_ms = len(ions) * 0.1  # ~0.1ms per draw call
    speedup = t_render * 1000 / gpu_est_ms if gpu_est_ms > 0 else 0
    print(f"\n12. Performance comparison:")
    print(f"    CPU render: {t_render * 1000:.1f}ms")
    print(f"    GPU estimate: {gpu_est_ms:.1f}ms ({len(ions)} draw calls x 0.1ms)")
    print(f"    Estimated speedup: {speedup:.0f}x")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
