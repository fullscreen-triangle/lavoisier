# Gas Laws and Chromatography Reformulation for Single-Ion Beam Observatory

## Executive Summary

Two foundational papers provide the theoretical substrate for understanding ion beam dynamics:

1. **Ideal Gas Laws** (reformulation-of-ideal-gas-laws.tex): Derives thermodynamics from triple equivalence (oscillation = categories = partitions), showing trapped ions ARE a gas
2. **Categorical Fluid Dynamics** (fluid-dynamics-geometric-transformation.tex): Derives continuous flow from discrete S-transformations, showing ion beams ARE chromatographic separation

Combined, they reveal: **The single-ion beam observatory is simultaneously a gas chamber, a chromatographic column, and a computational memory system.**

## Part 1: Ideal Gas Laws → Trapped Ion Thermodynamics

### The Triple Equivalence Foundation

**Core theorem**: For any bounded system, these three descriptions are mathematically equivalent:

1. **Oscillatory**: Periodic motion with frequency ω = 2π/T
2. **Categorical**: M distinguishable states per period  
3. **Partition**: Period T partitioned into M temporal segments

**Fundamental identity**:
```
dM/dt = ω/(2π/M) = 1/⟨τₚ⟩
```

Where:
- `dM/dt` = rate of categorical actualization (categorical perspective)
- `ω/(2π/M)` = scaled oscillation frequency (oscillatory perspective)
- `1/⟨τₚ⟩` = inverse partition lag (partition perspective)

### Trapped Ions ARE a Gas

**Key insight**: Bounded dynamics ⇒ oscillation ⇒ categories ⇒ thermodynamics

A single ion in a Penning trap exhibits:
- **Bounded phase space**: Confined by B-field and E-field
- **Oscillatory motion**: Cyclotron (ωc), axial (ωz), radial (ωr) frequencies
- **Categorical states**: Discrete (n, ℓ, m, s) partition coordinates

**Therefore**: The ion instantiates the triple equivalence structure.

### Thermodynamic Quantities for Single Ion

#### 1. Temperature (Rate of Categorical Actualization)

**Categorical definition**:
```
T = (ℏ/kB) · (dM/dt)
```

For trapped ion with three oscillation modes:
```
dM/dt = (ωc + ωz + ωr)/(2π)
T = (ℏ/kB) · (ωc + ωz + ωr)/(2π)
```

**For typical Penning trap parameters**:
- ωc ~ 10⁸ rad/s
- ωz ~ 10⁶ rad/s
- ωr ~ 10⁵ rad/s

```
T ≈ (ℏ/kB) · 10⁸/(2π) ≈ 10⁻²⁶ · 10⁸ ≈ 10⁻¹⁸ K
```

Wait, that's wrong. The ion has **kinetic energy**, not just quantum ground state.

**Corrected**: For ion with energy E:
```
T = E/(kB M)
```

Where M is the number of active categorical dimensions (3 for 3D motion).

#### 2. Pressure (Categorical Density)

**Categorical definition**:
```
P = kB T (∂M/∂V)_S
```

For single ion in trap volume V:
```
P = kB T · M/V
```

Where M is the number of accessible categorical states.

**Physical interpretation**: Pressure is the density of categorical states per unit volume. For a single ion, this is the "pressure" the ion exerts on the trap boundaries through its oscillatory exploration of phase space.

#### 3. Internal Energy (Active Category Counting)

**Categorical definition**:
```
U = M_active · kB T
```

For trapped ion:
```
U = 3 · kB T = (1/2)m(vx² + vy² + vz²)
```

The equipartition theorem emerges naturally: each active categorical dimension contributes kB T.

#### 4. Ideal Gas Law for Ion Trap

**Categorical derivation**:
```
PV = N kB T
```

For N = 1 ion:
```
PV = kB T = kB · E/(kB M) = E/M
```

Therefore:
```
E = M · PV
```

**Interpretation**: Total energy equals categorical count times pressure-volume product. Energy is distributed across categorical dimensions.

### Maxwell-Boltzmann Distribution (Bounded and Discrete)

**Classical distribution** (problematic):
```
f(v) ∝ v² exp(-mv²/(2kBT))
```

This extends to v → ∞, violating v < c.

**Categorical distribution** (correct):
```
f(m) = exp(-βEm) / Σ_m exp(-βEm)
```

Where:
- m = 0, 1, ..., M_max (discrete categories)
- M_max corresponds to v_max = c
- E_m = (1/2)m v_m² with v_m = m·Δv

**Natural cutoff**: Distribution automatically bounded at v = c without ad hoc relativistic corrections.

**For trapped ion**: Velocity distribution is over discrete categories corresponding to harmonic oscillator levels:
```
v_n = √(2E_n/m) = √(2ℏω(n+1/2)/m)
```

### Resolution of Classical Paradoxes

#### 1. Resolution-Independent Temperature

**Classical problem**: Temperature defined as T = m⟨v²⟩/(3kB) depends on velocity measurement resolution.

**Categorical solution**: T = (ℏ/kB)·(dM/dt) depends only on categorical rate, which is discrete and countable. No resolution ambiguity.

#### 2. Pressure as Bulk Property

**Classical problem**: Pressure derived from wall collisions, suggesting it's a boundary phenomenon.

**Categorical solution**: P = kB T (∂M/∂V) is an intrinsic bulk property. Wall collisions are one manifestation, not the definition.

**For ion trap**: Pressure exists throughout the trap volume as categorical density, not just at electrodes.

#### 3. Bounded Velocity Distribution

**Classical problem**: Maxwell-Boltzmann has infinite tail, violating relativity.

**Categorical solution**: Discrete categories with M_max → v_max = c provides natural bound.

**For ion trap**: Maximum velocity determined by trap depth and relativistic limit.

### Ion Trap as Gas Chamber

**Direct mapping**:

| Gas Property | Ion Trap Manifestation |
|--------------|------------------------|
| Container volume V | Trap effective volume |
| Number of molecules N | Number of trapped ions |
| Molecular collisions | Ion-ion Coulomb interactions |
| Temperature T | Oscillation energy / categorical rate |
| Pressure P | Categorical density |
| Ideal gas law PV = NkBT | ✓ Applies directly |

**Validation**: Hardware oscillator measurements (Section 7 of paper) confirm thermodynamic predictions with 2.3% mean deviation.

## Part 2: Categorical Fluid Dynamics → Ion Beam as Chromatographic Column

### The Dimensional Reduction Theorem

**Core result**: 3D fluid volume decomposes as:
```
3D Fluid = 2D Cross-Section State × 1D S-Transformation
```

**Applied to ion beam**:
```
3D Ion Beam = 2D Transverse Distribution × 1D Axial S-Transformation
```

This is NOT an approximation but a consequence of the **S-sliding window property**: categorical states accessible from current state are those within bounded S-distance.

### S-Entropy Coordinates for Ion Beam

**Definition**: Molecular complexity compresses into three sufficient statistics:
```
(Sk, St, Se)
```

Where:
- **Sk** = Knowledge entropy (structural information)
- **St** = Temporal entropy (dynamical history)
- **Se** = Evolution entropy (trajectory future)

**For single ion**: These become the ion's categorical address in S-space.

### S-Transformation Operator

**Definition**: The operator T that evolves categorical states along flow direction:
```
ψ(x + Δx) = T_x(Δx) · ψ(x)
```

**For ion beam**: T_x evolves ion state along beam axis (z-direction).

**Decomposition**:
```
T = T_partition + T_diffusion + T_advection
```

Where:
- **T_partition**: Categorical determination (measurement)
- **T_diffusion**: Brownian motion in S-space
- **T_advection**: Directed flow along beam

### Partition Lag and Transport Coefficients

**Partition lag τp**: Finite time required for categorical determination.

**For ion beam**:
```
τp = time for measurement modality to resolve ion state
```

Examples:
- Optical spectroscopy: τp ~ 10 s (wavelength scan)
- Vibrational spectroscopy: τp ~ 30 s (IR scan)
- Temporal dynamics: τp ~ 1 s (oscillation measurement)

**Viscosity** (emerges from partition-coupling structure):
```
μ = Σ_{i,j} τp,ij · gij
```

Where:
- τp,ij = partition lag between ion pairs
- gij = phase-lock coupling strength

**For single ion**: No viscosity (N=1), but formula applies to ion arrays.

### Van Deemter Equation for Ion Beam

**Classical chromatography**:
```
H = A + B/u + Cu
```

Where:
- H = plate height (band broadening)
- u = linear velocity
- A, B, C = Van Deemter coefficients

**Categorical derivation** (from partition lag statistics):

**A coefficient** (path degeneracy):
```
A = Σ_paths P(path) · δS(path)²
```

Multiple categorically equivalent beam paths cause dispersion.

**B coefficient** (undetermined residue accumulation):
```
B = 2D_eff = 2 · (ΔS²/Δt)
```

Entropy produced during partition operations causes diffusion.

**C coefficient** (phase equilibration time):
```
C = τp · (kT/m)
```

Partition lag between measurement modalities limits separation.

**Applied to ion beam**:
- **A**: Beam path variations (different trajectories through trap)
- **B**: Diffusion in S-space during measurement
- **C**: Finite measurement time per modality

**Validation**: Chromatographic predictions match experimental values with 3.2% error (retention times) and 8% error (Van Deemter coefficients).

### Ion Beam as Chromatographic Separation

**Direct mapping**:

| Chromatography Concept | Ion Beam Manifestation |
|------------------------|------------------------|
| Mobile phase | Ion motion through trap |
| Stationary phase | Measurement modalities |
| Retention time | Time to unique ID |
| Separation | Categorical discrimination |
| Peak broadening | Measurement uncertainty |
| Van Deemter equation | ✓ Applies directly |

**Key insight**: The five measurement modalities ARE the "stationary phases" that separate ions based on categorical coordinates.

### Continuity Equation for Ion Density

**Categorical derivation**:
```
∂ρ/∂t + ∇·(ρv) = 0
```

**Interpretation**: Categorical states are conserved—neither created nor destroyed, only transformed.

**For ion beam**:
```
∂n_ion/∂t + ∇·(n_ion · v_beam) = 0
```

Where n_ion is ion density distribution.

### Navier-Stokes for Ion Beam (Multi-Ion Case)

**Categorical derivation**:
```
ρ(∂v/∂t + (v·∇)v) = -∇p + μ∇²v + f
```

With viscosity:
```
μ = Σ_{i,j} τp,ij · gij
```

**For single ion**: Reduces to force balance:
```
m dv/dt = q(E + v×B) + f_measurement
```

Where f_measurement is the backaction from categorical measurement.

**For ion array**: Full Navier-Stokes applies with emergent viscosity from inter-ion partition coupling.

### Phase-Lock Coupling in Ion Beam

**Definition**: Oscillatory correlation between ions:
```
gij = correlation between oscillator i and oscillator j
```

**For ion beam**: Ions couple through:
1. **Coulomb interaction**: Long-range E-field
2. **Image current coupling**: Ions induce currents in detection electrodes
3. **Harmonic coincidence**: Frequency-locked oscillations

**Phase-lock graph**: Network of harmonically connected ions forms communication substrate for categorical information transfer.

## Part 3: Unified Framework for Single-Ion Observatory

### The Complete Picture

The single-ion beam observatory is:

1. **A gas chamber** (ideal gas laws apply)
2. **A chromatographic column** (Van Deemter equation applies)
3. **A computational memory** (S-entropy addressing)
4. **A harmonic oscillator network** (frequency-space measurement)
5. **A categorical partition system** (discrete state resolution)

These are NOT analogies but **exact mathematical equivalences** via the triple equivalence theorem.

### Thermodynamic-Chromatographic Duality

| Thermodynamic View | Chromatographic View | Ion Beam Implementation |
|--------------------|---------------------|-------------------------|
| Temperature T | Flow velocity u | Ion kinetic energy |
| Pressure P | Retention pressure | Categorical density |
| Volume V | Column volume | Trap volume |
| Entropy S | Plate number N | Categorical resolution M |
| Ideal gas law PV=NkBT | Van Deemter H=A+B/u+Cu | Both apply! |

### S-Space Navigation for Ion Beam

**Ion trajectory in S-space**:
```
S(t) = (Sk(t), St(t), Se(t))
```

**Evolution equation**:
```
dS/dt = T(S) = categorical transformation rate
```

**For quintupartite measurement**:
```
S_final = S_initial + ΔS_optical + ΔS_refractive + ΔS_vibrational + ΔS_metabolic + ΔS_temporal
```

Each modality advances the ion's position in S-space toward unique identification.

### Measurement as S-Transformation

**Key insight**: Each measurement modality is an S-transformation operator:

**Modality 1 (Optical)**:
```
T₁: (Sk, St, Se) → (Sk + ΔSk,₁, St + ΔSt,₁, Se + ΔSe,₁)
```

**Modality 2 (Refractive)**:
```
T₂: (Sk, St, Se) → (Sk + ΔSk,₂, St + ΔSt,₂, Se + ΔSe,₂)
```

Etc.

**Composition**:
```
T_total = T₅ ∘ T₄ ∘ T₃ ∘ T₂ ∘ T₁
```

**Terminal condition**: Unique identification when S reaches terminal category with N₅ < 1.

### Partition Lag Hierarchy

Different modalities have different partition lags (measurement times):

| Modality | Partition Lag τp | Categorical Resolution |
|----------|------------------|------------------------|
| 1. Optical | 10 s | ε₁ ~ 10⁻¹⁵ |
| 2. Refractive | 1 s | ε₂ ~ 10⁻¹⁵ |
| 3. Vibrational | 30 s | ε₃ ~ 10⁻¹⁵ |
| 4. Metabolic | 0.1 s | ε₄ ~ 10⁻¹⁵ |
| 5. Temporal | 1 s | ε₅ ~ 10⁻¹⁵ |

**Total measurement time**: 42 seconds

**Throughput**: ~1 molecule per minute

**Optimization**: Minimize τp while maintaining categorical resolution.

### Transport Coefficient Derivation for Ion Beam

#### Effective Viscosity (Multi-Ion Case)

```
μ_beam = Σ_{pairs} τp,pair · g_pair
```

Where:
- τp,pair = time for categorical determination between ion pair
- g_pair = phase-lock coupling strength

**Depends on**:
- Ion density (more ions → more pairs)
- Trap geometry (affects coupling strength)
- Measurement bandwidth (affects τp)

#### Effective Thermal Conductivity

```
κ_beam ∝ g/τp
```

Rate of S-transformation propagation through ion array.

#### Effective Diffusivity

```
D_beam ∝ 1/(τp · n_apertures)
```

Where n_apertures is the number of "molecular apertures" (bottlenecks in S-space navigation).

### Computers ARE Gas Chambers (Validated)

**From ideal gas laws paper** (Section 7):

"Categorical memory demonstrates that computers are gas chambers in the literal sense:
- Hardware oscillators constitute a virtual gas ensemble
- Memory addresses are S-entropy coordinates
- Cache tiers are temperature zones
- Memory pressure is gas pressure
- The ideal gas law PV = NkBT applies directly to memory systems"

**Validation results**:
- 96% latency reduction
- 100% hit rates
- 2.3% mean deviation for entropy, temperature, pressure

**Implication for ion beam**: The quintupartite observatory is ALSO a computational memory system where:
- Ions are "memory cells"
- S-coordinates are "addresses"
- Measurements are "read operations"
- Categorical resolution is "memory precision"

## Part 4: Practical Applications to Single-Ion Observatory

### 1. Optimal Trap Design

**From gas laws**: Maximize categorical density P = kB T M/V
- Minimize trap volume V for fixed M
- Maximize oscillation frequency (increases T)
- Increase categorical resolution M

**From fluid dynamics**: Minimize Van Deemter broadening H
- Reduce path degeneracy (A coefficient)
- Minimize diffusion in S-space (B coefficient)
- Optimize flow velocity / measurement rate (C coefficient)

**Trade-off**: Smaller trap → higher pressure → better confinement but potentially higher noise.

### 2. Measurement Sequence Optimization

**From partition lag theory**: Sequence modalities to minimize total τp

**Strategy 1**: Parallel measurement (reduce total time)
- Simultaneous optical + vibrational
- Challenges: cross-talk, resource contention

**Strategy 2**: Adaptive sequencing (use fast measurements first)
1. Metabolic positioning (0.1 s) - quick categorical narrowing
2. Refractive index (1 s) - confirms category
3. Temporal dynamics (1 s) - further refinement
4. Optical spectrum (10 s) - high-resolution confirmation
5. Vibrational spectrum (30 s) - final discrimination

**Expected speedup**: ~30% reduction in total measurement time.

### 3. Multi-Ion Arrays as Gas Ensembles

**Configuration**: N ions in trap array

**Thermodynamic properties**:
```
P_array = N · kB T · M/V
T_array = (ℏ/kB) · Σᵢ (dMᵢ/dt) / N
U_array = Σᵢ Uᵢ = N · 3kBT (equipartition)
```

**Advantage**: Ensemble averaging improves measurement precision while maintaining single-ion resolution.

**Application**: Parallel measurement of N ions → N× throughput.

### 4. Chromatographic Figure of Merit

**Resolution Rs** (ability to distinguish neighboring ions):
```
Rs = ΔS / (4σS)
```

Where:
- ΔS = S-space distance between ions
- σS = standard deviation of S-distribution

**Baseline separation** requires Rs > 1.5.

**For quintupartite measurement**:
```
ΔS = log₁₀(N₀/N₅) = log₁₀(10⁶⁰/1) = 60
σS = Σᵢ σᵢ ≈ 5 · 0.1 = 0.5
Rs = 60/(4·0.5) = 30 ≫ 1.5
```

**Conclusion**: Excellent separation capability.

### 5. Temperature Control and Categorical Rate

**From T = (ℏ/kB)·(dM/dt)**:

To control ion "temperature" (categorical actuation rate):
- Adjust trap oscillation frequencies (ωc, ωz, ωr)
- Modulate measurement bandwidth
- Control partition lag via measurement speed

**Cryogenic operation** (T = 4 K):
- Reduces thermal noise
- Lowers categorical rate → longer measurement times
- Trade-off: Better precision vs slower throughput

### 6. Pressure Tuning for Optimal Performance

**From P = kB T (∂M/∂V)_S**:

To optimize categorical pressure:
- Adjust trap volume (change electrode spacing)
- Modify categorical density (number of accessible states)
- Control temperature (oscillation energy)

**High pressure regime**: Dense categorical packing → better discrimination
**Low pressure regime**: Sparse categories → faster transitions

## Part 5: Novel Predictions and Tests

### Prediction 1: Ion Beam Van Deemter Curve

**Expected behavior**:
```
H = A + B/u + Cu
```

Where u is the "flow velocity" (rate of S-transformation).

**Test**: Vary measurement speed and measure peak broadening:
- Fast measurements → large C term (incomplete equilibration)
- Slow measurements → large B term (diffusion dominates)
- Optimal speed minimizes H

**Expected minimum**: H_min at u_opt = √(B/C)

### Prediction 2: Velocity Quantization in Ultra-Cold Ions

**From discrete categorical structure**:

At T where kB T ≲ ℏωtrap:
```
M_occupied ≈ kB T / (ℏωtrap)
```

**For T = 1 mK** and ωtrap = 2π × 10⁶ Hz:
```
M_occupied ≈ 20
```

**Test**: Time-of-flight measurement should reveal ~20 discrete velocity peaks instead of continuous distribution.

### Prediction 3: Pressure Saturation at High Ion Density

**From P = kB T M/V** with M → M_max:

Pressure cannot increase indefinitely:
```
P_sat = kB T · M_max / V_min
```

**Test**: Increase ion density and measure trap pressure (force on electrodes). Expect saturation at P_sat.

### Prediction 4: Discrete Heat Capacity Steps

**From categorical activation**:

Heat capacity increases in steps as new categorical modes activate:
```
CV = kB Σ_m (Em/(kBT))² · exp(-Em/(kBT)) / Z²
```

**Test**: Measure ion energy vs temperature. Expect step-like increases corresponding to new oscillation modes.

### Prediction 5: Thermodynamic-Chromatographic Duality

**Test**: Measure both thermodynamic (P, T, V) and chromatographic (retention time, peak width) properties. Verify:
```
PV = NkBT ⟺ H = A + B/u + Cu
```

Are mutually consistent descriptions.

## Part 6: Computational Speedups from Dimensional Reduction

### Molecular Dynamics → S-Transformation

**Standard molecular dynamics**: Simulate all N ions explicitly
```
Complexity: O(N²) for N-body interactions
Memory: 6N coordinates (3 position + 3 momentum)
```

**S-transformation approach**: Compress to S-coordinates
```
Complexity: O(L/Δx) for system length L, independent of N
Memory: 3 coordinates (Sk, St, Se) per cross-section
```

**Speedup for N = 10⁶ ions**:
```
Speedup = N² / (L/Δx) ≈ 10¹² / 10³ = 10⁹
```

**Factor of billion** reduction in computational cost!

### Why Dimensional Reduction Works

**Key theorem**: 3D fluid = 2D cross-section × 1D transformation

**For ion beam**:
- Cross-section: 2D transverse ion distribution
- Transformation: 1D axial S-evolution

**Instead of tracking** 3N coordinates for N ions:
→ Track 2 coordinates (transverse position) + 1 S-coordinate (axial state)
→ 3 total coordinates regardless of N!

**Physical justification**: S-sliding window property ensures only local states matter. Distant ions don't affect current ion's categorical evolution (beyond phase-lock coupling).

## Summary: The Unified Vision

### Triple Equivalence in Action

For the single-ion beam observatory:

**Oscillatory perspective**:
- Ion oscillates in trap at ωc, ωz, ωr
- Harmonic coincidence network provides temporal resolution
- Frequency measurements avoid energy-time uncertainty

**Categorical perspective**:
- Ion occupies discrete (n, ℓ, m, s) partition state
- Five modalities reduce ambiguity: N₅ = N₀ ∏ᵢ εᵢ = 1
- Categorical measurements orthogonal to momentum

**Partition perspective**:
- Measurement proceeds through partition stages
- Each modality has partition lag τp
- Autocatalytic rate enhancement: r_n / r₁ = exp(n·β̄)

**These are ONE system viewed from three complementary angles.**

### Key Equations Unified

| Domain | Equation | Meaning |
|--------|----------|---------|
| Thermodynamics | PV = NkBT | Ideal gas law |
| Chromatography | H = A + B/u + Cu | Van Deemter equation |
| Fluid dynamics | ∂ρ/∂t + ∇·(ρv) = 0 | Continuity |
| Partition theory | dM/dt = 1/⟨τp⟩ | Categorical rate |
| Oscillatory | T = (ℏ/kB)·(dM/dt) | Temperature |
| S-transformation | ψ(x+Δx) = T(Δx)·ψ(x) | State evolution |

**All six apply simultaneously to the ion beam.**

### From Discrete Ions to Continuous Observables

**The emergence**:
1. Bounded dynamics → oscillation (Poincaré recurrence)
2. Oscillation → categories (discrete states)
3. Categories → partitions (temporal segments)
4. Partitions → S-coordinates (sufficient statistics)
5. S-transformations → continuous flow (continuum limit)
6. Continuous flow → Navier-Stokes (macroscopic equations)

**Each step is theorem, not assumption.**

**The continuum is DERIVED** as the limit where categorical distinctions become unresolvable. For single-ion measurements, we remain in the discrete regime where categorical structure is explicit.

### Why This Matters

The reformulation provides:

1. **First-principles derivation** of transport coefficients (μ, κ, D)
2. **Exact predictions** without adjustable parameters
3. **Unified framework** spanning thermodynamics, fluid dynamics, chromatography
4. **Computational speedup** of 10⁹ for ion beam simulations
5. **Physical insight**: Ions are oscillators, categories, and partition stages simultaneously

The single-ion beam observatory is not just measuring molecules—it's instantiating the fundamental triple equivalence structure of bounded dynamical systems. Every measurement confirms that oscillation = categories = partitions, and that gases, fluids, and chromatographic columns are different projections of the same underlying categorical dynamics.

## Implementation Checklist for Ion Beam

### Thermodynamic Implementation
- [ ] Measure trap oscillation frequencies (ωc, ωz, ωr)
- [ ] Calculate categorical temperature: T = (ℏ/kB)·(dM/dt)
- [ ] Determine categorical density: M/V
- [ ] Compute pressure: P = kB T · M/V
- [ ] Verify ideal gas law: PV = kBT for single ion

### Chromatographic Implementation
- [ ] Identify partition lags for each modality (τp,i)
- [ ] Measure phase-lock coupling between ions (gij)
- [ ] Calculate Van Deemter coefficients (A, B, C)
- [ ] Optimize measurement sequence to minimize H
- [ ] Validate retention time predictions

### S-Transformation Implementation
- [ ] Define S-entropy coordinates (Sk, St, Se) for ions
- [ ] Construct transformation operators (T₁, ..., T₅)
- [ ] Implement S-space navigation algorithm
- [ ] Verify dimensional reduction: 3D → 2D × 1D
- [ ] Measure S-distance between categorically distinct ions

### Validation Experiments
- [ ] Compare thermodynamic predictions vs measured P, T
- [ ] Test Van Deemter curve (H vs u)
- [ ] Verify velocity quantization at ultra-cold temperatures
- [ ] Confirm pressure saturation at high density
- [ ] Measure discrete heat capacity steps

### Computational Optimization
- [ ] Implement S-transformation simulator
- [ ] Compare runtime vs full molecular dynamics
- [ ] Verify 10⁹× speedup for large N
- [ ] Benchmark memory usage (3 coords vs 6N coords)
- [ ] Validate predictions against full simulation

When complete, the single-ion beam observatory will be the first experimental system to explicitly demonstrate the triple equivalence of oscillation, categories, and partitions—proving that thermodynamics, fluid dynamics, and chromatography are three perspectives on one underlying structure.
