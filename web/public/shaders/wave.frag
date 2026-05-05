#version 300 es
precision highp float;

// --------------------------------------------------------------------------
// Pass 1 — Ion Wave Field Rendering
//
// Encodes the exact wave equation from ThermodynamicWaveGenerator:
//   wave = amplitude * exp(-d / (radius * 30 * decay)) * cos(2pi * d / (wl * 5))
//        * (1 + 0.3 * cos(angle_grid - impact_angle))
//        * cos(categorical_state * pi / 10)
//
// One draw call per ion, accumulated via additive blending.
// --------------------------------------------------------------------------

const float TWO_PI = 6.283185307;
const float PI     = 3.141592654;

// Ion parameters — set per draw call
uniform vec4  u_ion;   // [cx, cy, amplitude, wavelength]
uniform vec4  u_ion2;  // [decay_rate, impact_angle_rad, phase_offset, radius]
uniform vec2  u_resolution;

out vec4 fragColor;

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 center    = u_ion.xy;
    float amplitude  = u_ion.z;
    float wavelength = u_ion.w;
    float decay_rate = u_ion2.x;
    float angle      = u_ion2.y;
    float phase_off  = u_ion2.z;
    float radius     = u_ion2.w;

    float dist = length(fragCoord - center);

    // Wave equation — verbatim translation of Python:
    // wave = amplitude * exp(-distance / (radius * 30 * decay_rate))
    //       * cos(2pi * distance / (wavelength * 5))
    float wave = amplitude
               * exp(-dist / (radius * 30.0 * decay_rate + 1e-6))
               * cos(TWO_PI * dist / (wavelength * 5.0 + 1e-6));

    // Directional bias: 1 + 0.3 * cos(angle_grid - impact_angle)
    float angle_grid = atan(fragCoord.y - center.y, fragCoord.x - center.x);
    wave *= (1.0 + 0.3 * cos(angle_grid - angle));

    // Categorical state phase offset
    wave *= cos(phase_off);

    // Output — additive blending accumulates all ions
    fragColor = vec4(wave, wave, wave, 1.0);
}
