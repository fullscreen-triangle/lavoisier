#version 300 es
precision highp float;

// --------------------------------------------------------------------------
// Pass 4 — Resonance Comparison (Interference Observation)
//
// Reads a query observation (Texture 1: the accumulated wave field of the
// query ions) and a candidate observation (Texture 2: a generated candidate
// spectrum rendered through the same shader). Computes per-cell
// interference magnitude.
//
// Low interference anywhere signal exists = oscillatory resonance
// (shared partition structure). This IS the comparison — no algorithmic
// similarity function is evaluated.
// --------------------------------------------------------------------------

uniform sampler2D u_queryField;       // Pass 1 output for the query
uniform sampler2D u_candidateField;   // Pass 1 output for a candidate
uniform vec2  u_resolution;
uniform float u_queryMax;              // normalisation of query
uniform float u_candidateMax;          // normalisation of candidate
uniform float u_signalThreshold;       // below this, treat as no signal

in  vec2 v_uv;
out vec4 fragColor;

void main() {
    float q = texture(u_queryField, v_uv).r;
    float c = texture(u_candidateField, v_uv).r;

    // Normalise so both live in comparable [-1, 1]
    float qn = u_queryMax > 0.0 ? q / u_queryMax : 0.0;
    float cn = u_candidateMax > 0.0 ? c / u_candidateMax : 0.0;

    float interference = abs(qn - cn);

    // Has signal at this cell? (either field nontrivially populated)
    float signal = max(abs(qn), abs(cn));
    float mask = step(u_signalThreshold, signal);

    // Resonance score: 1 = perfect resonance, 0 = maximum disagreement
    float resonance = mask * (1.0 - clamp(interference, 0.0, 1.0));

    // Colour output for visual inspection:
    //   green  = high resonance (small interference where both have signal)
    //   red    = low resonance (large interference where one or both have signal)
    //   black  = no signal at this cell
    vec3 high = vec3(0.2, 0.9, 0.35);
    vec3 low  = vec3(1.0, 0.25, 0.25);
    vec3 col  = mix(low, high, resonance);
    col *= mask;

    // r = resonance score, g = interference magnitude, b = signal strength, a = mask
    fragColor = vec4(resonance, interference, signal, mask);
}
