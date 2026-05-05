#version 300 es
precision highp float;

// --------------------------------------------------------------------------
// Pass 3 — Bijective Validation Shader
//
// Encodes the bijective validation theorem directly: for each pixel, the
// visual path (wave field) and numerical path (spectrum data) should map
// to the same categorical state. Disagreement is coloured red.
//
// This replaces the pure-Python bijective_validated flag with a spatial
// quality map computed in parallel across all pixels.
// --------------------------------------------------------------------------

uniform sampler2D u_waveField;       // visual graph (normalised wave canvas)
uniform sampler2D u_numericalField;  // numerical graph (spectrum as texture)
uniform float     u_tolerance;       // categorical bin tolerance (default 0.15)

in vec2 v_uv;
out vec4 fragColor;

void main() {
    // Visual path: categorical state from wave field
    float waveVal = texture(u_waveField, v_uv).r;
    float catVisual = floor(waveVal * 10.0) / 10.0;

    // Numerical path: categorical state from spectrum
    float specVal = texture(u_numericalField, v_uv).r;
    float catNumerical = floor(specVal * 10.0) / 10.0;

    // Bijective check: they should be in the same categorical bin
    float discrepancy = abs(catVisual - catNumerical);
    float isValid = step(discrepancy, u_tolerance);

    // Colourmap: green = bijection holds, red = violation
    vec3 valid_col   = vec3(0.2, 0.9, 0.3);
    vec3 invalid_col = vec3(1.0, 0.2, 0.1);
    vec3 col = mix(invalid_col, valid_col, isValid);

    // Only visualise regions with meaningful signal
    float signalMask = smoothstep(0.02, 0.1, abs(waveVal));
    col = mix(vec3(0.05), col, signalMask);

    // Alpha encodes signal presence for GPU readback scoring
    fragColor = vec4(col, signalMask);
}
