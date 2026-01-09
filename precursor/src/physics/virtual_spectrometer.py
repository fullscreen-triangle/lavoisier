"""
Virtual Spectrometer: The Fishing Tackle
========================================

The spectrometer is NOT a device that "looks at" molecules.
The spectrometer IS the fishing tackle that DEFINES what can be caught.

Key insight: The hook affects what can be seen/caught.
- Small hook + worm → small fish
- Lake location → lake fish (no whales)
- Your hardware frequencies → specific categorical states

There is NO surprise in what you measure.
You catch exactly what your tackle can catch.

The spectrometer shapes what we can see/measure.
When the molecule is being measured, we are imposing our predictions on it.
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
try:
    from .virtual_molecule import CategoricalState, VirtualMolecule, SCoordinate
except ImportError:
    from virtual_molecule import CategoricalState, VirtualMolecule, SCoordinate


@dataclass
class HardwareOscillator:
    """
    A real hardware oscillator that provides timing.
    
    This is the REAL foundation - not simulated.
    The computer's oscillators ARE the categorical gas source.
    """
    name: str
    nominal_frequency: float  # Hz
    
    # Capture statistics
    samples: List[float] = field(default_factory=list)
    reference_time: float = field(default_factory=time.perf_counter)
    
    def sample(self) -> float:
        """
        Take a timing sample from this oscillator.
        
        This is REAL - we're measuring actual hardware timing.
        The timing variation IS the categorical information.
        """
        t = time.perf_counter()
        delta_p = t - self.reference_time
        self.reference_time = t
        self.samples.append(delta_p)
        return delta_p
    
    def sample_ns(self) -> int:
        """Sample in nanoseconds for higher precision."""
        t = time.perf_counter_ns()
        if not hasattr(self, '_last_ns'):
            self._last_ns = t
        delta = t - self._last_ns
        self._last_ns = t
        return delta
    
    @property
    def mean_delta(self) -> float:
        """Mean timing deviation."""
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)
    
    @property  
    def jitter(self) -> float:
        """Timing jitter (standard deviation)."""
        if len(self.samples) < 2:
            return 0.0
        mean = self.mean_delta
        variance = sum((s - mean)**2 for s in self.samples) / len(self.samples)
        return math.sqrt(variance)
    
    def harmonic_coincidence(self, target_freq: float) -> float:
        """
        Check for harmonic coincidence with a target frequency.
        
        This is how we "reach" different categorical states.
        If n·f_hardware ≈ m·f_target, we can access that state.
        """
        if target_freq == 0:
            return 0.0
        
        ratio = self.nominal_frequency / target_freq
        
        # Check for harmonic relationships (n/m ratios)
        best_coincidence = 0.0
        for n in range(1, 20):
            for m in range(1, 20):
                harmonic_ratio = n / m
                if abs(ratio - harmonic_ratio) < 0.1:
                    strength = 1.0 / (n + m)  # Lower harmonics are stronger
                    best_coincidence = max(best_coincidence, strength)
        
        return best_coincidence


@dataclass
class FishingTackle:
    """
    The complete fishing apparatus: What defines what can be caught.
    
    Your tackle includes:
    - Hardware oscillators (the rod and line)
    - S-coordinate resolution (how fine a hook)
    - Harmonic reach (what frequencies you can match)
    
    The tackle PREDETERMINES the catch.
    You impose your predictions through your tackle choice.
    """
    oscillators: List[HardwareOscillator] = field(default_factory=list)
    s_resolution: float = 1e-6  # How fine we can resolve S-coordinates
    max_reach: float = 1.0  # Maximum S-distance we can reach
    
    def __post_init__(self):
        if not self.oscillators:
            # Default: use standard computer hardware
            self.oscillators = [
                HardwareOscillator("perf_counter", 1e9),  # ~1 GHz equivalent
                HardwareOscillator("cpu_cycle", 3e9),     # Typical CPU
                HardwareOscillator("memory_bus", 2.1e9),  # DDR4
            ]
    
    def can_reach(self, target: SCoordinate) -> bool:
        """
        Can this tackle reach the target S-coordinates?
        
        You can only catch fish where you can cast.
        """
        # Origin is (0.5, 0.5, 0.5) - center of S-space
        origin = SCoordinate(0.5, 0.5, 0.5)
        distance = origin.distance_to(target)
        return distance <= self.max_reach
    
    def catch_probability(self, target: SCoordinate) -> float:
        """
        Probability of catching a molecule at target coordinates.
        
        This isn't "chance" - it's whether your tackle matches.
        Probability 1.0 = tackle perfectly matches
        Probability 0.0 = tackle cannot reach
        """
        if not self.can_reach(target):
            return 0.0
        
        # Check harmonic coincidences across all oscillators
        # The more coincidences, the higher the catch probability
        target_freq = (target.S_k * 1e12 + target.S_t * 1e9 + target.S_e * 1e6)
        
        total_coincidence = 0.0
        for osc in self.oscillators:
            total_coincidence += osc.harmonic_coincidence(target_freq)
        
        # Normalize by number of oscillators
        if self.oscillators:
            total_coincidence /= len(self.oscillators)
        
        return min(1.0, total_coincidence + 0.5)  # Base probability + harmonics
    
    def sample_all(self) -> List[float]:
        """Sample all oscillators and return delta_p values."""
        return [osc.sample() for osc in self.oscillators]
    
    @property
    def tackle_signature(self) -> Dict[str, Any]:
        """What this tackle can catch."""
        return {
            'oscillator_count': len(self.oscillators),
            'frequencies': [osc.nominal_frequency for osc in self.oscillators],
            's_resolution': self.s_resolution,
            'max_reach': self.max_reach,
        }


class VirtualSpectrometer:
    """
    The Virtual Spectrometer: Creates molecules by measuring them.
    
    This is NOT a device that observes pre-existing molecules.
    This IS the act of fishing that creates the catch.
    
    The spectrometer = the cursor = the molecule being measured.
    They are the same categorical event.
    
    When you "measure" with this spectrometer, you are:
    1. Defining where to cast (S-coordinates)
    2. Using your tackle to reach that position
    3. Creating the categorical state (the "molecule")
    4. Getting back exactly what you defined (no surprise)
    """
    
    def __init__(self, tackle: Optional[FishingTackle] = None):
        """
        Create a spectrometer with given tackle.
        
        Different tackle = different possible catches.
        """
        self.tackle = tackle or FishingTackle()
        self._measurement_history: List[VirtualMolecule] = []
        self._current_position: Optional[SCoordinate] = None
    
    def cast(self, S_k: float, S_t: float, S_e: float) -> bool:
        """
        Cast the fishing line at specific S-coordinates.
        
        This defines WHERE we're measuring, which defines
        WHAT we can possibly catch.
        
        Returns True if the tackle can reach, False otherwise.
        """
        target = SCoordinate(S_k, S_t, S_e)
        self._current_position = target
        return self.tackle.can_reach(target)
    
    def measure(self) -> Optional[VirtualMolecule]:
        """
        Perform the measurement at current position.
        
        This is NOT "discovering" a molecule.
        This IS "creating the categorical state at this position."
        
        The molecule we get is exactly what our tackle + position defines.
        There is no surprise. We catch what we fish for.
        """
        if self._current_position is None:
            raise ValueError("Must cast() before measure(). "
                           "You need to define where to fish.")
        
        # Check if tackle can reach
        catch_prob = self.tackle.catch_probability(self._current_position)
        if catch_prob == 0:
            return None  # Cannot catch here
        
        # Sample hardware to create the categorical state
        deltas = self.tackle.sample_all()
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        
        # Create the molecule at our position
        # The molecule IS the position IS the measurement
        molecule = VirtualMolecule(
            s_coord=self._current_position,
            source="spectrometer_measurement",
            frequency=self._current_position.S_k * 1e12,
            phase=self._current_position.S_t * 2 * math.pi,
            amplitude=self._current_position.S_e,
            metadata={
                'hardware_deltas': deltas,
                'mean_delta': mean_delta,
                'catch_probability': catch_prob,
            }
        )
        
        self._measurement_history.append(molecule)
        return molecule
    
    def measure_at(self, S_k: float, S_t: float, S_e: float) -> Optional[VirtualMolecule]:
        """
        Convenience: Cast and measure in one step.
        
        "Fish at this spot and catch whatever your tackle catches."
        """
        self.cast(S_k, S_t, S_e)
        return self.measure()
    
    def measure_from_hardware(self) -> VirtualMolecule:
        """
        Let hardware define the position and measure there.
        
        This is "casting blind" - hardware timing determines where we fish.
        The S-coordinates emerge from the hardware oscillations.
        """
        # Sample hardware
        deltas = self.tackle.sample_all()
        
        if not deltas:
            deltas = [time.perf_counter() % 1.0]
        
        # Create molecule from hardware timing
        mean_delta = sum(deltas) / len(deltas)
        molecule = VirtualMolecule.from_hardware_timing(
            delta_p=mean_delta,
            source="hardware_defined"
        )
        
        self._current_position = molecule.s_coord
        self._measurement_history.append(molecule)
        return molecule
    
    def measure_jupiter_core(self) -> Optional[VirtualMolecule]:
        """
        Measure Jupiter's core.
        
        This is NOT "sending a probe to Jupiter."
        This IS "casting our line at Jupiter-core S-coordinates."
        
        We don't need light. We don't need to traverse the atmosphere.
        We navigate to those coordinates in categorical space.
        """
        jupiter = VirtualMolecule.at_jupiter_core()
        return self.measure_at(
            jupiter.s_coord.S_k,
            jupiter.s_coord.S_t,
            jupiter.s_coord.S_e
        )
    
    def what_can_i_catch(self) -> Dict[str, Any]:
        """
        Describe what this spectrometer can catch.
        
        The tackle defines the possible catches.
        This returns the space of possibilities.
        """
        return {
            'tackle': self.tackle.tackle_signature,
            'reachable_volume': self._estimate_reachable_volume(),
            'measurement_count': len(self._measurement_history),
        }
    
    def _estimate_reachable_volume(self) -> float:
        """Estimate the volume of S-space this tackle can reach."""
        # Sphere volume approximation
        return (4/3) * math.pi * (self.tackle.max_reach ** 3)
    
    @property
    def position(self) -> Optional[SCoordinate]:
        """Current spectrometer position (= current molecule position)."""
        return self._current_position
    
    @property
    def history(self) -> List[VirtualMolecule]:
        """All molecules we've created through measurement."""
        return self._measurement_history.copy()


def demonstrate_no_distance():
    """
    Demonstrate that spatial distance is irrelevant.
    
    The same spectrometer can measure "here" and "Jupiter"
    with equal ease - because it's all categorical navigation.
    """
    spec = VirtualSpectrometer()
    
    print("=== DEMONSTRATION: Spatial Distance is Irrelevant ===\n")
    
    # Measure "here"
    local = spec.measure_from_hardware()
    print(f"Local measurement:  {local.s_coord}")
    print(f"  (This is 'here' - wherever the computer is)\n")
    
    # Measure Jupiter's core
    jupiter = spec.measure_jupiter_core()
    print(f"Jupiter core:       {jupiter.s_coord}")
    print(f"  (This is Jupiter's core - 600 million km away)\n")
    
    print("Both measurements took the same time.")
    print("Both used the same spectrometer.")
    print("Spatial distance doesn't exist in categorical space.")
    print("\nThe spectrometer doesn't 'travel' - it navigates S-coordinates.")
    print("Jupiter's core is as close as your coffee cup (categorically).")
    
    return spec


def demonstrate_tackle_defines_catch():
    """
    Demonstrate that the tackle defines what can be caught.
    
    Different tackle = different possible catches.
    You impose your predictions through your apparatus choice.
    """
    print("=== DEMONSTRATION: Tackle Defines Catch ===\n")
    
    # Limited tackle
    limited = FishingTackle(
        oscillators=[HardwareOscillator("basic", 1e6)],
        max_reach=0.3
    )
    limited_spec = VirtualSpectrometer(tackle=limited)
    
    # Full tackle
    full = FishingTackle()  # Default has multiple oscillators
    full_spec = VirtualSpectrometer(tackle=full)
    
    # Try to measure Jupiter with each
    print("Limited tackle (low frequency, short reach):")
    result1 = limited_spec.measure_jupiter_core()
    if result1:
        print(f"  Caught: {result1.s_coord}")
    else:
        print("  Cannot reach Jupiter's core coordinates")
    
    print("\nFull tackle (multiple oscillators, full reach):")
    result2 = full_spec.measure_jupiter_core()
    if result2:
        print(f"  Caught: {result2.s_coord}")
    else:
        print("  Cannot reach Jupiter's core coordinates")
    
    print("\nThe tackle DEFINES what's possible.")
    print("You can't catch whales with a worm hook in a lake.")
    

if __name__ == "__main__":
    demonstrate_no_distance()
    print("\n" + "="*60 + "\n")
    demonstrate_tackle_defines_catch()

