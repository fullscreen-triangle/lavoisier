#version 300 es
precision highp float;

// --------------------------------------------------------------------------
// Pass 6 — Physical Quality Metrics (single-pass extraction)
//
// For each cell, this shader writes a 4-channel output:
//   R = partition sharpness (gradient magnitude, local 3×3)
//   G = noise indicator    (Laplacian magnitude — high-frequency content)
//   B = phase coherence    (cos of phase difference with right/down neighbours)
//   A = signal mask        (1 where the field has significant amplitude)
//
// Aggregation across the full texture is done on the CPU after
// readback (a trivial reduction), OR can be done with a second
// mipmap-style reduction pass. For now the per-cell output is
// sufficient for averaging.
//
// These metrics are the GPU physical observables — objective,
// deterministic, free. They become the training signal for the
// compiled probe without human labels.
// --------------------------------------------------------------------------

uniform sampler2D u_waveField;   // accumulated or normalised wave field
uniform vec2      u_resolution;  // pixels
uniform float     u_signalThreshold;

in  vec2 v_uv;
out vec4 fragColor;

// Sample the field at a pixel offset
float tap(vec2 uv, vec2 offset) {
    return texture(u_waveField, uv + offset / u_resolution).r;
}

void main() {
    vec2 uv = v_uv;
    float c = tap(uv, vec2(0.0, 0.0));

    // Gradient magnitude — Sobel 3×3
    float tl = tap(uv, vec2(-1.0, -1.0));
    float t  = tap(uv, vec2( 0.0, -1.0));
    float tr = tap(uv, vec2( 1.0, -1.0));
    float l  = tap(uv, vec2(-1.0,  0.0));
    float r  = tap(uv, vec2( 1.0,  0.0));
    float bl = tap(uv, vec2(-1.0,  1.0));
    float b  = tap(uv, vec2( 0.0,  1.0));
    float br = tap(uv, vec2( 1.0,  1.0));

    float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    float sharpness = sqrt(gx * gx + gy * gy);

    // Laplacian — proxy for high-frequency noise
    float laplacian = abs((t + b + l + r) - 4.0 * c);

    // Phase coherence — cos of phase differences with right and down
    // neighbours. Use sign as proxy for phase (cos stays 1 when signs agree).
    float phaseR = sign(c) * sign(r);
    float phaseD = sign(c) * sign(b);
    float coherence = 0.5 * (phaseR + phaseD);  // [-1, 1]
    coherence = 0.5 * (coherence + 1.0);         // [0, 1]

    // Signal mask
    float signal = abs(c);
    float mask = step(u_signalThreshold, signal);

    fragColor = vec4(sharpness, laplacian, coherence, mask);
}
