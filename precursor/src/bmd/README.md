# Biological Maxwell Demon (BMD) Module

**Hardware-Constrained Categorical Completion for Mass Spectrometry Analysis**

This module implements Biological Maxwell Demon operations that ground spectral processing in physical hardware reality through dual filtering and categorical completion.

## Overview

### What are Biological Maxwell Demons?

BMDs are **information catalysts** that perform dual filtering operations:

1. **Input Filter (ℑ_input)**: Select signal from noise based on phase-lock coherence
   - `Y_↓^(in) → Y_↑^(in)`
   - Hardware phase structure determines which peaks are signal vs noise
   - Peaks phase-locked with hardware oscillations = signal

2. **Output Filter (ℑ_output)**: Target physically-grounded interpretations
   - `Z_↓^(fin) → Z_↑^(fin)`
   - Only interpretations coherent with hardware stream are physical
   - Stream divergence filters unphysical interpretations

### Key Concepts

- **Categorical State**: Equivalence class of configurations with identical phase relationships
- **Oscillatory Hole**: Physical absence in oscillatory cascade requiring completion
- **Phase-Lock**: Mutual phase coherence (|φ_i - φ_j| < π/4) between oscillatory modes
- **Categorical Richness**: R(β) = |hole| × Π_k N_k(Φ) - number of completion pathways
- **Hardware BMD Stream**: Unified reality reference from all hardware devices

## Architecture

```
Hardware Oscillations (8 sources)
    ├─ Clock Drift (AC power line 50-60 Hz)
    ├─ Memory Access Patterns
    ├─ Network Latency Jitter
    ├─ USB Timing Phase
    ├─ GPU Thermal Cycles
    ├─ Disk Access Patterns
    └─ LED Blink Phase
         ↓
    Phase-Lock Composition (β_1 ⊛ β_2 ⊛ ... ⊛ β_n)
         ↓
    Unified BMD Stream β^(stream)_hardware
         ↓
    Reality Grounding (R = ⋂_devices C_device)
         ↓
Theatre/Stage/Process Filtering
```

## Usage

### 1. Initialize Hardware BMD Reference

```python
from precursor.bmd import BiologicalMaxwellDemonReference

# Create BMD reference with all hardware harvesters
bmd_ref = BiologicalMaxwellDemonReference(enable_all_harvesters=True)

# Or enable specific devices
bmd_ref = BiologicalMaxwellDemonReference(
    enable_all_harvesters=False,
    enable_specific=['clock', 'memory', 'network']
)
```

### 2. Measure Hardware BMD Stream

```python
# Measure current hardware state
hardware_stream = bmd_ref.measure_stream()

# Check stream quality
if hardware_stream.is_coherent(threshold=0.7):
    print(f"Stream is coherent with quality {hardware_stream.phase_lock_quality}")

# Get unified BMD
unified_bmd = hardware_stream.unified_bmd
print(f"Categorical richness: {unified_bmd.categorical_richness}")
```

### 3. Use BMD for Filtering

```python
from precursor.bmd import compute_ambiguity, compute_stream_divergence

# Input filtering (signal vs noise)
filtered_peaks = bmd.input_filter(
    candidates=detected_peaks,
    criterion='phase_lock'
)

# Compute ambiguity for region
ambiguity = compute_ambiguity(network_bmd, spectrum_region)

# Check stream coherence
stream_div = compute_stream_divergence(network_bmd, hardware_stream.unified_bmd)

# Output filtering (physical interpretations only)
physical_interpretations = bmd.output_filter(
    interpretations=all_interpretations,
    hardware_bmd=hardware_stream.unified_bmd
)
```

### 4. BMD Comparison and Generation

```python
from precursor.bmd import (
    compare_bmd_with_region,
    generate_bmd_from_comparison,
    integrate_hierarchical
)

# Compare BMD with spectrum region
ambiguity, stream_div = compare_bmd_with_region(
    bmd=network_bmd,
    region_data=spectrum_slice,
    hardware_bmd=hardware_stream.unified_bmd,
    lambda_coupling=0.1
)

# Generate new BMD through categorical completion
new_bmd = generate_bmd_from_comparison(
    bmd=current_bmd,
    target=spectrum_region,
    hardware_bmd=hardware_stream.unified_bmd
)

# Hierarchically integrate into network BMD
network_bmd = integrate_hierarchical(
    network_bmd=network_bmd,
    new_bmd=new_bmd,
    processing_sequence=['peak_detection', 's_entropy', 'annotation']
)
```

### 5. Categorical States

```python
from precursor.bmd import CategoricalState, CategoricalStateSpace

# Create categorical state from S-Entropy coordinates
state = CategoricalState(
    state_id="spectrum_001_peak_250.5",
    s_entropy_coords=np.array([...]),  # 14D coordinates
    phase_relationships={'clock': 0.5, 'memory': 1.2},
    categorical_richness=1000
)

# Create state space
space = CategoricalStateSpace()
space.add_state(state)

# Find phase-locked ensemble
ensemble = space.find_phase_locked_ensemble(
    reference_state=state,
    phase_threshold=np.pi/4
)
print(f"Found ensemble of {len(ensemble)} phase-locked states")
```

## Core Classes

### BMDState

```python
class BMDState:
    bmd_id: str
    current_categorical_state: Optional[CategoricalState]
    oscillatory_hole: Optional[OscillatoryHole]
    phase_structure: PhaseStructure
    categorical_richness: int

    def input_filter(self, candidates: List[Any]) -> List[Any]
    def output_filter(self, interpretations: List[Any]) -> List[Any]
    def hierarchical_merge(self, other: BMDState) -> BMDState
```

### CategoricalState

```python
class CategoricalState:
    state_id: str
    s_entropy_coords: np.ndarray  # 14D from S-Entropy
    phase_relationships: Dict[str, float]
    categorical_richness: int

    def is_phase_locked_with(self, other: CategoricalState) -> bool
    def compute_entropy(self, completion_sequences: List[List[str]]) -> float
```

### PhaseStructure

```python
class PhaseStructure:
    modes: Dict[str, float]  # mode_name -> phase (rad)
    frequencies: Dict[str, float]
    coherence_times: Dict[str, float]

    def evolve(self, dt: float) -> PhaseStructure
    def is_phase_locked(self, other: PhaseStructure) -> bool
    def merge(self, other: PhaseStructure) -> PhaseStructure
```

### HardwareBMDStream

```python
class HardwareBMDStream:
    unified_bmd: BMDState
    device_bmds: Dict[str, BMDState]
    phase_lock_quality: float  # [0, 1]

    def get_categorical_richness() -> int
    def is_coherent(threshold: float) -> bool
```

## BMD Algebra Operations

### Comparison

```python
# Ambiguity measure
A(β, R) = Σ_{c ∈ C(R)} P(c|R) · D_KL(P_complete(c|β) || P_image(c|R))
```

High ambiguity = many incompatible completion pathways
Low ambiguity = strong alignment

### Generation

```python
# New BMD through categorical completion
β' = Generate(β, R) = ⟨c_new, H(c_new), Φ'⟩

# Selection criterion
c_new = argmin_c [E_fill(c_current → c) + λ · A(β_c, R)]
```

Completes oscillatory hole, creates new hole for continued cascade.

### Stream Divergence

```python
# Measure drift from hardware reality
D_stream(β^network, β^hardware) = Σ_device D_KL(P_phase^network || P_phase^hardware,device)
```

High divergence = network BMD has drifted from physical constraints.

### Hierarchical Integration

```python
# Update network BMD
β^(network)_{i+1} = IntegrateHierarchical(β^(network)_i, β_{i+1}, σ)
```

Creates O(2^n) compound BMDs through hierarchical composition.

## Theoretical Foundation

### Information Catalysis

BMDs are **information catalysts** (iCat) that increase probability of specific transformations:

- Without catalyst: `Y_↓ → Z_↑` with probability p₀ ≈ 0
- With iCat: `Y_↓ →^(iCat) Z_↑` with p_iCat ≫ p₀

Unlike chemical catalysts (enhance rates), BMDs enhance **probabilities**.

### Order Creation

```python
Ω^POT = {[Y_↓^(in,r) → Z_↑^(fin,q)], (r,q) ∈ ℕ × ℕ}  # Potential transformations
                    ↓
              Φ = {iCat}  # Set of BMDs
                    ↓
Ω^ACT ⊂ Ω^POT  # Actual transformations (much smaller)
```

BMDs reduce vast potential transformation space to small actual space.

### Phase-Locked Ensembles

Gas molecules form ~10³-10⁴ molecule ensembles with:

- Coherence length ξ ≈ 10-20 nm
- Coherence lifetime τ_coh ∼ 10⁻¹³ s
- Phase criterion |φ_i - φ_j| < π/4

Molecular O₂ has 25,110 categorical states (paramagnetic ground state).

### Energy Dissipation (Landauer Bound)

```python
E_fill(c_i → c_{i+1}) ≥ k_B T log N(c_i)
```

Filling oscillatory hole requires energy dissipation proportional to entropy reduction.

## Integration with Precursor

### Theatre (Transcendent Observer)

```python
from precursor.pipeline import Theatre
from precursor.bmd import BiologicalMaxwellDemonReference

theatre = Theatre(...)
bmd_ref = BiologicalMaxwellDemonReference()

# Theatre uses BMD for grounding
results = theatre.observe_all_stages_with_bmd_grounding(
    spectrum_data,
    bmd_reference=bmd_ref
)
```

### Stage (Finite Observer)

```python
from precursor.pipeline import StageObserver
from precursor.bmd import compute_ambiguity

stage = StageObserver(...)

# Stage implements dual filtering
stage_result = stage.observe_with_bmd_grounding(
    spectrum_data,
    hardware_bmd=hardware_stream.unified_bmd,
    network_bmd=network_bmd
)
```

### S-Entropy as Categorical States

```python
from precursor.core import SEntropyTransformer
from precursor.bmd import CategoricalState

transformer = SEntropyTransformer()

# Transform spectrum → categorical state
coords, features = transformer.transform_and_extract(mz, intensity, rt=rt)

categorical_state = CategoricalState(
    state_id=f"spectrum_{rt}",
    s_entropy_coords=features.to_array(),  # 14D
    categorical_richness=int(features.shannon_entropy * 1000)
)
```

## References

1. **Mizraji, E. (2021).** *The biological Maxwell's demons: exploring ideas about the information processing in biological systems.* Theory in Biosciences, 140, 307–318.
   - Original BMD theory: dual filtering as information catalysis
   - iCat framework: Ω^POT → Ω^ACT reduction

2. **Sachikonye, K. F. (2025).** *Phase-Locked Molecular Ensembles as Information-Encoding Structures in Gas Systems.*
   - Phase-lock theory: ~10⁴ molecules with ξ ≈ 10-20 nm
   - O₂ categorical states: 25,110 from paramagnetic ground state

3. **Sachikonye, K. F. (2025).** *Hardware-Constrained Categorical Completion: A Physical Foundation for Image Understanding Through Biological Maxwell Demon Dynamics.*
   - Hardware BMD stream: unified reality reference
   - Dual objective: maximize ambiguity while maintaining stream coherence
   - Iterative algorithm with finite convergence guarantee

## Examples

See:
- `examples/bmd_filtering_example.py` - BMD filtering demonstration
- `examples/hardware_bmd_stream_example.py` - Hardware stream measurement
- `examples/categorical_completion_example.py` - Full categorical completion workflow

## Performance

- **Hardware stream measurement**: ~1-10 ms
- **BMD filtering**: ~0.1-1 ms per spectrum
- **Categorical richness**: Stream intersection provides ~10³-10⁶ grounding
- **Phase-lock quality**: Typically 0.6-0.9 for coherent hardware

## License

MIT License - Part of Lavoisier Precursor project
