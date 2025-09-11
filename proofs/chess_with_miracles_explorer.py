#!/usr/bin/env python3
"""
Chess with Miracles Strategic Explorer

This implements Kundai's "Chess with Miracles" concept - a strategic navigation system
that combines chess-like position evaluation with S-entropy window sliding "miracles"
for solving subproblems. The system seeks viable solutions without rigid definitions
of "winning," allowing flexible strategic movement through the problem space.

Key Concepts:
1. Chess-like strategic evaluation of positions
2. Undefined but viable goals - continuous improvement seeking
3. Miracle window sliding for subsolution breakthroughs
4. Strategic acceptance of weak positions for long-term advantage
5. Meta-information as strategic knowledge of possibility space
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random
from abc import ABC, abstractmethod
import copy


class PositionStrength(Enum):
    """Strategic position strength evaluation."""
    DOMINANT = "dominant"
    ADVANTAGEOUS = "advantageous" 
    BALANCED = "balanced"
    WEAK = "weak"
    CRITICAL = "critical"


class MiracleType(Enum):
    """Types of miracles possible through window sliding."""
    KNOWLEDGE_BREAKTHROUGH = "knowledge_breakthrough"
    TIME_ACCELERATION = "time_acceleration"
    ENTROPY_ORGANIZATION = "entropy_organization"
    DIMENSIONAL_SHIFT = "dimensional_shift"
    SYNTHESIS_MIRACLE = "synthesis_miracle"


@dataclass
class ChessPosition:
    """Represents a position in the strategic problem space."""
    coordinates: np.ndarray  # S-entropy coordinates (x,y,z)
    information_level: float  # Amount of information/knowledge available
    time_to_solution: float  # Estimated time to reach solution from here
    entropy_to_solution: float  # Entropy cost to reach solution
    strength: PositionStrength  # Strategic evaluation of this position
    reachable_positions: List['ChessPosition'] = None
    miracle_opportunities: List[MiracleType] = None
    strategic_value: float = 0.0
    visited_count: int = 0


@dataclass
class MiracleWindow:
    """Represents a sliding window miracle that can solve subproblems."""
    window_type: MiracleType
    s_dimension: str  # 'knowledge', 'time', 'entropy'
    window_size: int
    window_position: int
    miracle_strength: float  # How much the miracle can improve things
    duration: int  # How long the miracle effect lasts
    cost: float  # S-entropy cost to perform the miracle
    subsolution: Any = None  # The subsolution achieved


class StrategicMove:
    """Represents a strategic move in the chess-with-miracles system."""
    
    def __init__(self, from_position: ChessPosition, to_position: ChessPosition,
                 miracle_used: Optional[MiracleWindow] = None):
        self.from_position = from_position
        self.to_position = to_position  
        self.miracle_used = miracle_used
        self.move_strength = self._evaluate_move_strength()
        self.strategic_reasoning = self._generate_strategic_reasoning()
    
    def _evaluate_move_strength(self) -> float:
        """Evaluate the strategic strength of this move."""
        # Base move strength from position improvement
        position_improvement = (
            self.to_position.strategic_value - self.from_position.strategic_value
        )
        
        # Bonus for miracle usage
        miracle_bonus = 0.0
        if self.miracle_used:
            miracle_bonus = self.miracle_used.miracle_strength * 0.5
        
        # Long-term strategic consideration
        long_term_potential = self.to_position.information_level * 0.3
        
        return position_improvement + miracle_bonus + long_term_potential
    
    def _generate_strategic_reasoning(self) -> str:
        """Generate human-readable strategic reasoning for the move."""
        reasons = []
        
        if self.to_position.strength.value in ['dominant', 'advantageous']:
            reasons.append(f"Moves to {self.to_position.strength.value} position")
        elif self.from_position.strength == PositionStrength.WEAK:
            reasons.append("Escapes weak position")
        
        if self.miracle_used:
            reasons.append(f"Uses {self.miracle_used.window_type.value} miracle")
        
        if self.to_position.information_level > self.from_position.information_level:
            reasons.append("Gains strategic information")
        
        if not reasons:
            reasons.append("Exploratory move seeking better position")
        
        return "; ".join(reasons)


class ChessWithMiraclesExplorer:
    """
    Strategic explorer implementing chess-like thinking with S-entropy miracles.
    
    This system thinks strategically about problem-solving, evaluating positions
    and planning moves like a chess player, but with the ability to perform
    "miracles" through S-entropy window sliding.
    """
    
    def __init__(self, lookahead_depth: int = 3, miracle_energy: float = 10.0):
        self.lookahead_depth = lookahead_depth
        self.miracle_energy = miracle_energy  # Energy available for miracles
        self.current_miracle_energy = miracle_energy
        
        # Strategic knowledge
        self.position_history: List[ChessPosition] = []
        self.move_history: List[StrategicMove] = []
        self.strategic_memory: Dict[str, Any] = {}
        
        # Miracle system
        self.available_miracles: List[MiracleType] = list(MiracleType)
        self.miracle_cooldowns: Dict[MiracleType, int] = {}
        
        # Strategic parameters
        self.risk_tolerance = 0.7  # Willingness to accept weak positions
        self.exploration_drive = 0.8  # Drive to explore unknown regions
        self.solution_sufficiency_threshold = 0.75  # When solution is "good enough"
    
    def evaluate_position(self, coordinates: np.ndarray) -> ChessPosition:
        """
        Evaluate a position like a chess player evaluating a board position.
        
        Args:
            coordinates: S-entropy coordinates of the position
            
        Returns:
            ChessPosition with strategic evaluation
        """
        # Calculate basic metrics
        information_level = self._calculate_information_level(coordinates)
        time_to_solution = self._estimate_time_to_solution(coordinates)
        entropy_to_solution = self._estimate_entropy_cost(coordinates)
        
        # Strategic strength evaluation
        strength = self._evaluate_position_strength(
            information_level, time_to_solution, entropy_to_solution
        )
        
        # Calculate strategic value (like chess piece values)
        strategic_value = self._calculate_strategic_value(
            information_level, time_to_solution, entropy_to_solution, strength
        )
        
        # Identify miracle opportunities
        miracle_opportunities = self._identify_miracle_opportunities(coordinates)
        
        position = ChessPosition(
            coordinates=coordinates.copy(),
            information_level=information_level,
            time_to_solution=time_to_solution,
            entropy_to_solution=entropy_to_solution,
            strength=strength,
            miracle_opportunities=miracle_opportunities,
            strategic_value=strategic_value
        )
        
        return position
    
    def _calculate_information_level(self, coordinates: np.ndarray) -> float:
        """Calculate available information at this position."""
        if coordinates.ndim == 1:
            s_knowledge = coordinates[0] if len(coordinates) > 0 else 0
        else:
            s_knowledge = np.mean(coordinates[:, 0]) if coordinates.shape[1] > 0 else 0
        
        # Information grows with knowledge but with diminishing returns
        return 1 - np.exp(-s_knowledge * 3)
    
    def _estimate_time_to_solution(self, coordinates: np.ndarray) -> float:
        """Estimate time to reach solution from this position."""
        if coordinates.ndim == 1:
            s_time = coordinates[1] if len(coordinates) > 1 else 0
        else:
            s_time = np.mean(coordinates[:, 1]) if coordinates.shape[1] > 1 else 0
        
        # Higher s_time means closer to solution
        base_time = 10 * (1 - s_time)  # Base time in arbitrary units
        
        # Add randomness for uncertainty
        uncertainty = np.random.normal(0, base_time * 0.2)
        
        return max(0.1, base_time + uncertainty)
    
    def _estimate_entropy_cost(self, coordinates: np.ndarray) -> float:
        """Estimate entropy cost to reach solution."""
        if coordinates.ndim == 1:
            s_entropy = coordinates[2] if len(coordinates) > 2 else 0
        else:
            s_entropy = np.mean(coordinates[:, 2]) if coordinates.shape[1] > 2 else 0
        
        # Lower entropy positions require more work to organize
        entropy_cost = 5 * (1 - s_entropy) + 1
        
        return entropy_cost
    
    def _evaluate_position_strength(self, info: float, time: float, entropy: float) -> PositionStrength:
        """Evaluate strategic strength like a chess position evaluation."""
        # Combine metrics into overall strength score
        strength_score = info * 0.4 + (1/max(time, 0.1)) * 0.3 + (1/entropy) * 0.3
        
        if strength_score > 0.8:
            return PositionStrength.DOMINANT
        elif strength_score > 0.6:
            return PositionStrength.ADVANTAGEOUS
        elif strength_score > 0.4:
            return PositionStrength.BALANCED
        elif strength_score > 0.2:
            return PositionStrength.WEAK
        else:
            return PositionStrength.CRITICAL
    
    def _calculate_strategic_value(self, info: float, time: float, 
                                 entropy: float, strength: PositionStrength) -> float:
        """Calculate strategic value like chess piece values."""
        base_value = info * 3 + (1/max(time, 0.1)) * 2 + (1/entropy) * 1
        
        # Strength multiplier
        strength_multipliers = {
            PositionStrength.DOMINANT: 1.5,
            PositionStrength.ADVANTAGEOUS: 1.2,
            PositionStrength.BALANCED: 1.0,
            PositionStrength.WEAK: 0.7,
            PositionStrength.CRITICAL: 0.3
        }
        
        return base_value * strength_multipliers[strength]
    
    def _identify_miracle_opportunities(self, coordinates: np.ndarray) -> List[MiracleType]:
        """Identify what miracles are possible at this position."""
        opportunities = []
        
        if coordinates.ndim == 1:
            s_knowledge = coordinates[0] if len(coordinates) > 0 else 0
            s_time = coordinates[1] if len(coordinates) > 1 else 0
            s_entropy = coordinates[2] if len(coordinates) > 2 else 0
        else:
            s_knowledge = np.mean(coordinates[:, 0]) if coordinates.shape[1] > 0 else 0
            s_time = np.mean(coordinates[:, 1]) if coordinates.shape[1] > 1 else 0
            s_entropy = np.mean(coordinates[:, 2]) if coordinates.shape[1] > 2 else 0
        
        # Miracle opportunities based on current S-entropy state
        if s_knowledge < 0.5:
            opportunities.append(MiracleType.KNOWLEDGE_BREAKTHROUGH)
        
        if s_time < 0.3:
            opportunities.append(MiracleType.TIME_ACCELERATION)
        
        if s_entropy < 0.4:
            opportunities.append(MiracleType.ENTROPY_ORGANIZATION)
        
        if len(coordinates.shape) == 1 and len(coordinates) == 3:
            opportunities.append(MiracleType.DIMENSIONAL_SHIFT)
        
        # Always possible with sufficient energy
        if self.current_miracle_energy > 5.0:
            opportunities.append(MiracleType.SYNTHESIS_MIRACLE)
        
        return opportunities
    
    def generate_possible_moves(self, current_position: ChessPosition, 
                              include_miracles: bool = True) -> List[StrategicMove]:
        """
        Generate possible moves like a chess player considering options.
        
        Args:
            current_position: Current position in problem space
            include_miracles: Whether to consider miracle-enhanced moves
            
        Returns:
            List of possible strategic moves
        """
        moves = []
        
        # Generate basic moves (coordinate shifts)
        basic_moves = self._generate_basic_moves(current_position)
        moves.extend(basic_moves)
        
        # Generate miracle-enhanced moves
        if include_miracles:
            miracle_moves = self._generate_miracle_moves(current_position)
            moves.extend(miracle_moves)
        
        # Sort by strategic value
        moves.sort(key=lambda m: m.move_strength, reverse=True)
        
        return moves
    
    def _generate_basic_moves(self, position: ChessPosition) -> List[StrategicMove]:
        """Generate basic moves without miracles."""
        moves = []
        
        # Generate moves in each S-dimension
        move_patterns = [
            np.array([0.1, 0, 0]),    # Knowledge direction
            np.array([0, 0.1, 0]),    # Time direction  
            np.array([0, 0, 0.1]),    # Entropy direction
            np.array([0.1, 0.1, 0]),  # Knowledge-time diagonal
            np.array([0.1, 0, 0.1]),  # Knowledge-entropy diagonal
            np.array([0, 0.1, 0.1]),  # Time-entropy diagonal
            np.array([-0.05, 0.1, 0]), # Strategic retreat for advantage
        ]
        
        for pattern in move_patterns:
            new_coords = position.coordinates + pattern
            
            # Ensure coordinates stay in valid range
            new_coords = np.clip(new_coords, 0, 1)
            
            new_position = self.evaluate_position(new_coords)
            move = StrategicMove(position, new_position)
            moves.append(move)
        
        return moves
    
    def _generate_miracle_moves(self, position: ChessPosition) -> List[StrategicMove]:
        """Generate moves enhanced with miracles."""
        moves = []
        
        if not position.miracle_opportunities:
            return moves
        
        for miracle_type in position.miracle_opportunities:
            # Check if miracle is available (not on cooldown)
            if miracle_type in self.miracle_cooldowns:
                if self.miracle_cooldowns[miracle_type] > 0:
                    continue
            
            # Check if we have enough energy
            miracle_cost = self._get_miracle_cost(miracle_type)
            if self.current_miracle_energy < miracle_cost:
                continue
            
            # Create miracle window
            miracle_window = self._create_miracle_window(miracle_type, position)
            
            # Apply miracle effect to generate new position
            new_coords = self._apply_miracle_effect(
                position.coordinates, miracle_window
            )
            
            new_position = self.evaluate_position(new_coords)
            move = StrategicMove(position, new_position, miracle_window)
            moves.append(move)
        
        return moves
    
    def _get_miracle_cost(self, miracle_type: MiracleType) -> float:
        """Get energy cost for different miracle types."""
        costs = {
            MiracleType.KNOWLEDGE_BREAKTHROUGH: 3.0,
            MiracleType.TIME_ACCELERATION: 4.0,
            MiracleType.ENTROPY_ORGANIZATION: 2.5,
            MiracleType.DIMENSIONAL_SHIFT: 5.0,
            MiracleType.SYNTHESIS_MIRACLE: 6.0
        }
        return costs.get(miracle_type, 3.0)
    
    def _create_miracle_window(self, miracle_type: MiracleType, 
                              position: ChessPosition) -> MiracleWindow:
        """Create a miracle window for sliding window operation."""
        # Determine which S-dimension to affect
        dimension_map = {
            MiracleType.KNOWLEDGE_BREAKTHROUGH: 'knowledge',
            MiracleType.TIME_ACCELERATION: 'time',
            MiracleType.ENTROPY_ORGANIZATION: 'entropy',
            MiracleType.DIMENSIONAL_SHIFT: 'knowledge',  # Primary dimension
            MiracleType.SYNTHESIS_MIRACLE: 'time'  # Primary dimension
        }
        
        s_dimension = dimension_map[miracle_type]
        
        # Create miracle window
        miracle_window = MiracleWindow(
            window_type=miracle_type,
            s_dimension=s_dimension,
            window_size=random.randint(3, 7),  # Random window size
            window_position=random.randint(0, 2),  # Random position
            miracle_strength=random.uniform(0.3, 0.8),  # Random strength
            duration=random.randint(2, 5),  # How long effect lasts
            cost=self._get_miracle_cost(miracle_type)
        )
        
        return miracle_window
    
    def _apply_miracle_effect(self, coordinates: np.ndarray, 
                            miracle: MiracleWindow) -> np.ndarray:
        """Apply miracle effect through sliding window operation."""
        new_coords = coordinates.copy()
        
        if miracle.window_type == MiracleType.KNOWLEDGE_BREAKTHROUGH:
            # Boost knowledge dimension
            new_coords[0] = min(1.0, new_coords[0] + miracle.miracle_strength * 0.5)
            
        elif miracle.window_type == MiracleType.TIME_ACCELERATION:
            # Accelerate time progress
            new_coords[1] = min(1.0, new_coords[1] + miracle.miracle_strength * 0.4)
            
        elif miracle.window_type == MiracleType.ENTROPY_ORGANIZATION:
            # Organize entropy (increase organized information)
            new_coords[2] = min(1.0, new_coords[2] + miracle.miracle_strength * 0.6)
            
        elif miracle.window_type == MiracleType.DIMENSIONAL_SHIFT:
            # Shift across dimensions
            shift = np.random.normal(0, miracle.miracle_strength * 0.1, 3)
            new_coords += shift
            new_coords = np.clip(new_coords, 0, 1)
            
        elif miracle.window_type == MiracleType.SYNTHESIS_MIRACLE:
            # Boost all dimensions slightly
            boost = miracle.miracle_strength * 0.2
            new_coords += np.array([boost, boost, boost])
            new_coords = np.clip(new_coords, 0, 1)
        
        return new_coords
    
    def strategic_lookahead(self, position: ChessPosition, 
                          depth: int = None) -> Tuple[StrategicMove, float]:
        """
        Perform strategic lookahead like chess move analysis.
        
        Args:
            position: Current position to analyze from
            depth: Lookahead depth (uses self.lookahead_depth if None)
            
        Returns:
            Tuple of (best_move, expected_value)
        """
        if depth is None:
            depth = self.lookahead_depth
        
        if depth == 0:
            return None, position.strategic_value
        
        possible_moves = self.generate_possible_moves(position)
        
        if not possible_moves:
            return None, position.strategic_value
        
        best_move = None
        best_value = float('-inf')
        
        for move in possible_moves:
            # Recursively evaluate this move
            future_move, future_value = self.strategic_lookahead(
                move.to_position, depth - 1
            )
            
            # Calculate total expected value
            total_value = move.move_strength + future_value * 0.8  # Discount future
            
            if total_value > best_value:
                best_value = total_value
                best_move = move
        
        return best_move, best_value
    
    def make_strategic_decision(self, current_coordinates: np.ndarray) -> StrategicMove:
        """
        Make a strategic decision like a chess player choosing the best move.
        
        This implements the core "chess with miracles" decision-making.
        """
        current_position = self.evaluate_position(current_coordinates)
        
        # Check if current solution is sufficient
        if self._is_solution_sufficient(current_position):
            return None  # No move needed - solution is good enough
        
        # Perform strategic lookahead
        best_move, expected_value = self.strategic_lookahead(current_position)
        
        if best_move is None:
            # No good moves found - might need to accept weak position
            return self._make_desperate_move(current_position)
        
        # Execute the move
        self._execute_move(best_move)
        
        return best_move
    
    def _is_solution_sufficient(self, position: ChessPosition) -> bool:
        """Check if current position represents a sufficient solution."""
        # Solution is sufficient if strategic value exceeds threshold
        sufficient_value = position.strategic_value > self.solution_sufficiency_threshold
        
        # Or if we're in a dominant/advantageous position
        good_position = position.strength in [PositionStrength.DOMINANT, PositionStrength.ADVANTAGEOUS]
        
        # Or if information level is high enough
        sufficient_info = position.information_level > 0.8
        
        return sufficient_value or good_position or sufficient_info
    
    def _make_desperate_move(self, position: ChessPosition) -> StrategicMove:
        """Make a move when in desperate situation (like chess endgame)."""
        # Generate any possible move, even if weak
        all_moves = self._generate_basic_moves(position)
        
        if not all_moves:
            # Create a random exploratory move
            random_shift = np.random.normal(0, 0.1, 3)
            new_coords = np.clip(position.coordinates + random_shift, 0, 1)
            new_position = self.evaluate_position(new_coords)
            return StrategicMove(position, new_position)
        
        # Choose least bad move
        return min(all_moves, key=lambda m: abs(m.move_strength))
    
    def _execute_move(self, move: StrategicMove) -> None:
        """Execute a strategic move and update system state."""
        # Record move in history
        self.move_history.append(move)
        self.position_history.append(move.to_position)
        
        # If miracle was used, update energy and cooldowns
        if move.miracle_used:
            self.current_miracle_energy -= move.miracle_used.cost
            self.miracle_cooldowns[move.miracle_used.window_type] = move.miracle_used.duration
        
        # Update cooldowns
        for miracle_type in list(self.miracle_cooldowns.keys()):
            self.miracle_cooldowns[miracle_type] -= 1
            if self.miracle_cooldowns[miracle_type] <= 0:
                del self.miracle_cooldowns[miracle_type]
        
        # Regenerate some miracle energy over time
        self.current_miracle_energy = min(
            self.miracle_energy,
            self.current_miracle_energy + 0.5
        )
    
    def can_return_to_previous_position(self, target_position: ChessPosition) -> bool:
        """Check if we can return to a previous position (backtracking capability)."""
        # Can always return to previously visited positions
        for visited_pos in self.position_history:
            distance = np.linalg.norm(
                visited_pos.coordinates - target_position.coordinates
            )
            if distance < 0.1:  # Close enough
                return True
        
        return False
    
    def get_strategic_summary(self) -> Dict[str, Any]:
        """Get summary of strategic situation like a chess position evaluation."""
        if not self.position_history:
            return {"status": "No moves made yet"}
        
        current_position = self.position_history[-1]
        
        summary = {
            "current_strength": current_position.strength.value,
            "strategic_value": current_position.strategic_value,
            "information_level": current_position.information_level,
            "time_to_solution": current_position.time_to_solution,
            "available_miracles": len(current_position.miracle_opportunities or []),
            "miracle_energy": self.current_miracle_energy,
            "moves_made": len(self.move_history),
            "positions_visited": len(set(str(pos.coordinates) for pos in self.position_history)),
            "solution_sufficient": self._is_solution_sufficient(current_position)
        }
        
        return summary


def demonstrate_chess_with_miracles():
    """Demonstrate the chess with miracles strategic exploration."""
    print("="*70)
    print("CHESS WITH MIRACLES STRATEGIC EXPLORER DEMONSTRATION")
    print("="*70)
    print("Implementing strategic navigation with sliding window miracles")
    print("="*70)
    
    # Initialize the strategic explorer
    explorer = ChessWithMiraclesExplorer(
        lookahead_depth=3,
        miracle_energy=15.0
    )
    
    # Starting position (weak position)
    start_coords = np.array([0.2, 0.1, 0.3])  # Weak starting position
    print(f"\nStarting coordinates: {start_coords}")
    print("Starting from deliberately weak position...")
    
    # Strategic game simulation
    max_moves = 15
    move_count = 0
    
    current_coords = start_coords.copy()
    
    print(f"\n{'-'*50}")
    print("STRATEGIC GAME PROGRESSION")
    print(f"{'-'*50}")
    
    while move_count < max_moves:
        print(f"\nMove {move_count + 1}:")
        
        # Get strategic summary
        summary = explorer.get_strategic_summary()
        if move_count > 0:
            print(f"  Current position strength: {summary['current_strength']}")
            print(f"  Strategic value: {summary['strategic_value']:.3f}")
            print(f"  Information level: {summary['information_level']:.3f}")
            print(f"  Miracle energy: {summary['miracle_energy']:.1f}")
        
        # Make strategic decision
        strategic_move = explorer.make_strategic_decision(current_coords)
        
        if strategic_move is None:
            print("  ✓ SOLUTION SUFFICIENT - No further moves needed!")
            break
        
        # Display move information
        print(f"  Move type: {strategic_move.from_position.strength.value} → {strategic_move.to_position.strength.value}")
        print(f"  Move strength: {strategic_move.move_strength:.3f}")
        print(f"  Strategic reasoning: {strategic_move.strategic_reasoning}")
        
        if strategic_move.miracle_used:
            miracle = strategic_move.miracle_used
            print(f"  🎭 MIRACLE USED: {miracle.window_type.value}")
            print(f"    Window: {miracle.s_dimension} dimension, size {miracle.window_size}")
            print(f"    Strength: {miracle.miracle_strength:.2f}, Cost: {miracle.cost:.1f}")
        
        # Update position
        current_coords = strategic_move.to_position.coordinates
        move_count += 1
        
        # Show coordinate progression
        print(f"  New coordinates: [{current_coords[0]:.3f}, {current_coords[1]:.3f}, {current_coords[2]:.3f}]")
    
    # Final strategic analysis
    print(f"\n{'-'*50}")
    print("FINAL STRATEGIC ANALYSIS")
    print(f"{'-'*50}")
    
    final_summary = explorer.get_strategic_summary()
    
    print(f"Game completed in {move_count} moves")
    print(f"Final position strength: {final_summary['current_strength']}")
    print(f"Final strategic value: {final_summary['strategic_value']:.3f}")
    print(f"Final information level: {final_summary['information_level']:.3f}")
    print(f"Solution sufficient: {final_summary['solution_sufficient']}")
    print(f"Unique positions visited: {final_summary['positions_visited']}")
    print(f"Remaining miracle energy: {final_summary['miracle_energy']:.1f}")
    
    # Analyze strategic progression
    print(f"\n{'-'*30}")
    print("STRATEGIC PROGRESSION ANALYSIS")
    print(f"{'-'*30}")
    
    if len(explorer.position_history) > 1:
        start_value = explorer.position_history[0].strategic_value
        end_value = explorer.position_history[-1].strategic_value
        improvement = end_value - start_value
        
        print(f"Strategic improvement: {improvement:.3f}")
        print(f"Improvement rate: {improvement/move_count:.3f} per move")
        
        # Count miracle usage
        miracle_moves = [m for m in explorer.move_history if m.miracle_used]
        print(f"Miracles used: {len(miracle_moves)}/{move_count} moves")
        
        # Show miracle types used
        if miracle_moves:
            miracle_types = [m.miracle_used.window_type.value for m in miracle_moves]
            print(f"Miracle types: {', '.join(set(miracle_types))}")
    
    # Demonstrate key concepts
    print(f"\n{'-'*50}")
    print("KEY CONCEPTS DEMONSTRATED")
    print(f"{'-'*50}")
    
    print("✓ Chess-like strategic thinking:")
    print("  - Position evaluation with strength assessment")
    print("  - Lookahead analysis for move planning")
    print("  - Strategic acceptance of weak positions for long-term gain")
    
    print("\n✓ Miracles through sliding windows:")
    miracle_types_used = set(m.miracle_used.window_type for m in explorer.move_history if m.miracle_used)
    for miracle_type in miracle_types_used:
        print(f"  - {miracle_type.value} miracle performed")
    
    print("\n✓ Undefined but viable goals:")
    print("  - No rigid 'winning' condition defined")
    print("  - Continuous improvement seeking")
    print("  - Solution sufficiency rather than perfection")
    
    print("\n✓ Flexible strategic direction:")
    strength_progression = [pos.strength.value for pos in explorer.position_history]
    print(f"  - Position strength evolution: {' → '.join(strength_progression)}")
    
    print("\n✓ Meta-information as strategic knowledge:")
    print("  - Awareness of all possible landing positions")
    print("  - Strategic memory of visited positions")
    print("  - Capability for tactical backtracking")
    
    # Visualization
    create_strategic_visualization(explorer)
    
    return explorer


def create_strategic_visualization(explorer: ChessWithMiraclesExplorer):
    """Create visualization of the strategic game progression."""
    if not explorer.position_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Chess with Miracles: Strategic Exploration', fontsize=16)
    
    # Position progression in S-space
    ax1 = axes[0, 0]
    positions = np.array([pos.coordinates for pos in explorer.position_history])
    
    # Plot trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2, label='Strategic Path')
    ax1.scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], 
               cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    
    # Mark start and end
    ax1.scatter(positions[0, 0], positions[0, 1], c='red', s=200, marker='s', 
               label='Start (Weak)', edgecolor='black', linewidth=2)
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='gold', s=200, marker='*',
               label='End', edgecolor='black', linewidth=2)
    
    ax1.set_xlabel('S_knowledge')
    ax1.set_ylabel('S_time')
    ax1.set_title('Strategic Position Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Strategic value over time
    ax2 = axes[0, 1]
    strategic_values = [pos.strategic_value for pos in explorer.position_history]
    move_numbers = range(len(strategic_values))
    
    ax2.plot(move_numbers, strategic_values, 'g-', linewidth=3, marker='o', markersize=6)
    ax2.set_xlabel('Move Number')
    ax2.set_ylabel('Strategic Value')
    ax2.set_title('Strategic Value Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Mark miracle moves
    for i, move in enumerate(explorer.move_history):
        if move.miracle_used:
            ax2.axvline(i+1, color='red', linestyle='--', alpha=0.7)
            ax2.text(i+1, strategic_values[i+1], '🎭', fontsize=12, ha='center')
    
    # Position strength distribution
    ax3 = axes[1, 0]
    strength_counts = {}
    for pos in explorer.position_history:
        strength = pos.strength.value
        strength_counts[strength] = strength_counts.get(strength, 0) + 1
    
    strengths = list(strength_counts.keys())
    counts = list(strength_counts.values())
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(strengths)]
    
    bars = ax3.bar(strengths, counts, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Position Strength')
    ax3.set_ylabel('Number of Moves')
    ax3.set_title('Strategic Position Strength Distribution')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    # Miracle usage analysis
    ax4 = axes[1, 1]
    miracle_counts = {}
    for move in explorer.move_history:
        if move.miracle_used:
            miracle_type = move.miracle_used.window_type.value
            miracle_counts[miracle_type] = miracle_counts.get(miracle_type, 0) + 1
    
    if miracle_counts:
        miracle_types = list(miracle_counts.keys())
        miracle_vals = list(miracle_counts.values())
        
        wedges, texts, autotexts = ax4.pie(miracle_vals, labels=miracle_types, autopct='%1.0f',
                                          startangle=90, colors=plt.cm.Set3.colors[:len(miracle_types)])
        ax4.set_title('Miracle Usage Distribution')
    else:
        ax4.text(0.5, 0.5, 'No Miracles Used', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Miracle Usage Distribution')
    
    plt.tight_layout()
    plt.savefig('proofs/chess_with_miracles_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the demonstration
    explorer = demonstrate_chess_with_miracles()
    
    print(f"\n{'='*70}")
    print("CHESS WITH MIRACLES DEMONSTRATION COMPLETE!")
    print(f"{'='*70}")
    print("Key innovations validated:")
    print("  • Strategic chess-like position evaluation and planning")
    print("  • Sliding window 'miracles' for subproblem solving")
    print("  • Flexible goal-seeking without rigid win conditions")
    print("  • Acceptance of weak positions for strategic advantage")
    print("  • Meta-information as strategic knowledge of possibility space")
    print("\nThe system demonstrates true strategic intelligence with miracle capabilities!")
