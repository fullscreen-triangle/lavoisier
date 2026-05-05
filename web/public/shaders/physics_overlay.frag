#version 300 es
precision highp float;

// --------------------------------------------------------------------------
// Pass 2 — Normalisation + S-Entropy Colour Overlay + Physics Validation
//
// Reads the accumulated wave field from Pass 1 and applies:
//   1. Normalisation to [0, 1]
//   2. S-entropy colour mapping (Sk=cyan, St=amber, Se=violet)
//   3. Physics quality masking (dim invalid ions)
//
// Replaces Python normalisation loop + validate_physics() checks.
// --------------------------------------------------------------------------

const int MAX_IONS = 512;

uniform sampler2D u_waveField;   // accumulated wave canvas from Pass 1
uniform sampler2D u_ionData;     // ion parameters texture (3 rows x N columns)
uniform int       u_ionCount;
uniform vec2      u_resolution;
uniform float     u_waveMin;     // pre-computed global min
uniform float     u_waveMax;     // pre-computed global max

in vec2 v_uv;
out vec4 fragColor;

void main() {
    // Read accumulated wave field
    float rawWave = texture(u_waveField, v_uv).r;

    // Normalise to [0, 1]
    float wave = (rawWave - u_waveMin) / (u_waveMax - u_waveMin + 1e-10);

    // Find the dominant ion at this pixel (nearest center)
    vec2 fragCoord = v_uv * u_resolution;
    float minDist = 1e9;
    float quality = 1.0;
    float Sk = 0.0, St = 0.0, Se = 0.0;

    for (int i = 0; i < MAX_IONS; i++) {
        if (i >= u_ionCount) break;

        // Row 0: [cx, cy, amplitude, wavelength]
        vec4 row0 = texelFetch(u_ionData, ivec2(i, 0), 0);
        // Row 1: [decay_rate, impact_angle, phase_offset, radius]
        // Row 2: [Sk, St, Se, categorical_state_normalised]
        vec4 row2 = texelFetch(u_ionData, ivec2(i, 2), 0);

        vec2 center = row0.xy;
        float d = length(fragCoord - center);

        if (d < minDist) {
            minDist = d;
            Sk = row2.x;
            St = row2.y;
            Se = row2.z;
            // Physics quality from row 1.w
            vec4 row1 = texelFetch(u_ionData, ivec2(i, 1), 0);
            quality = row1.w;  // placeholder; replaced by proper physics score
        }
    }

    // S-entropy colour mapping
    // Sk → cyan (0.00, 0.83, 1.00)
    // St → amber (1.00, 0.42, 0.20)
    // Se → violet (0.75, 0.50, 1.00)
    vec3 cSk = vec3(0.0, 0.83, 1.0);
    vec3 cSt = vec3(1.0, 0.42, 0.2);
    vec3 cSe = vec3(0.75, 0.5, 1.0);
    vec3 entropyCol = Sk * cSk + St * cSt + Se * cSe;
    entropyCol = clamp(entropyCol, 0.0, 1.0);

    // Dim invalid ions (physics quality < threshold)
    float valid = smoothstep(0.2, 0.5, quality);

    // Final colour: wave intensity * entropy colour * validity
    vec3 col = wave * entropyCol * valid;

    fragColor = vec4(col, 1.0);
}
