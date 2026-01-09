"""
System Topology Experiment
Validates categorical spaces, completion operators, equivalence classes,
S-distance metric, and 3^k branching structure.
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
from pathlib import Path


@dataclass
class CategoricalState:
    """A state in categorical space."""
    id: int
    s_k: float  # Knowledge entropy
    s_t: float  # Temporal entropy
    s_e: float  # Evolution entropy
    completed: bool = False
    completion_time: float = -1.0
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    
    def to_s_coord(self) -> np.ndarray:
        return np.array([self.s_k, self.s_t, self.s_e])


class CategoricalSpace:
    """Implementation of categorical space with completion dynamics."""
    
    def __init__(self, n_states: int = 100):
        self.states: Dict[int, CategoricalState] = {}
        self.completion_order: List[int] = []
        self.current_time = 0.0
        
        # Generate states with S-coordinates
        np.random.seed(42)
        for i in range(n_states):
            state = CategoricalState(
                id=i,
                s_k=np.random.random(),
                s_t=np.random.random(),
                s_e=np.random.random()
            )
            self.states[i] = state
        
        # Build partial order (DAG structure)
        self._build_partial_order()
    
    def _build_partial_order(self):
        """Build partial order based on S-coordinate proximity."""
        state_list = list(self.states.values())
        for i, s1 in enumerate(state_list):
            for j, s2 in enumerate(state_list):
                if i != j:
                    # s1 < s2 if s1 has lower temporal coordinate
                    if s1.s_t < s2.s_t and np.random.random() < 0.3:
                        s1.successors.add(s2.id)
                        s2.predecessors.add(s1.id)
    
    def complete_state(self, state_id: int) -> bool:
        """Complete a categorical state (irreversible)."""
        state = self.states[state_id]
        
        # Check if already completed
        if state.completed:
            return False
        
        # Check order compatibility - all predecessors must be completed
        for pred_id in state.predecessors:
            if not self.states[pred_id].completed:
                return False
        
        # Complete the state
        state.completed = True
        state.completion_time = self.current_time
        self.completion_order.append(state_id)
        self.current_time += np.random.exponential(0.1)
        return True
    
    def get_completion_trajectory(self) -> List[np.ndarray]:
        """Get the trajectory of completed states."""
        return [self.states[sid].to_s_coord() for sid in self.completion_order]
    
    def compute_s_distance(self, traj1: List[np.ndarray], traj2: List[np.ndarray]) -> float:
        """Compute S-distance between two trajectories."""
        # Align trajectories by length
        min_len = min(len(traj1), len(traj2))
        if min_len == 0:
            return float('inf')
        
        distance = 0.0
        for i in range(min_len):
            distance += np.linalg.norm(traj1[i] - traj2[i])
        return distance


class EquivalenceClassAnalyzer:
    """Analyze equivalence classes under observables."""
    
    def __init__(self, space: CategoricalSpace, observable_dim: int = 0):
        self.space = space
        self.observable_dim = observable_dim  # Which S-coordinate to use as observable
        self.n_bins = 10
    
    def compute_equivalence_classes(self) -> Dict[int, List[int]]:
        """Partition states into equivalence classes."""
        classes = {}
        for state_id, state in self.space.states.items():
            coord = state.to_s_coord()[self.observable_dim]
            bin_idx = min(int(coord * self.n_bins), self.n_bins - 1)
            if bin_idx not in classes:
                classes[bin_idx] = []
            classes[bin_idx].append(state_id)
        return classes
    
    def compute_degeneracy(self, state_id: int) -> int:
        """Compute degeneracy of a state."""
        classes = self.compute_equivalence_classes()
        state = self.space.states[state_id]
        coord = state.to_s_coord()[self.observable_dim]
        bin_idx = min(int(coord * self.n_bins), self.n_bins - 1)
        return len(classes.get(bin_idx, []))
    
    def compute_richness(self, state_id: int) -> float:
        """Compute categorical richness."""
        state = self.space.states[state_id]
        degeneracy = self.compute_degeneracy(state_id)
        n_downstream = len(state.successors)
        
        # R = log(degeneracy) + log(n_downstream)
        log_deg = np.log(max(1, degeneracy))
        log_down = np.log(max(1, n_downstream))
        return log_deg + log_down


class HierarchicalBranching:
    """Validate 3^k hierarchical branching."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.branching_factor = 3
    
    def generate_hierarchy(self) -> Dict[int, Dict]:
        """Generate 3^k hierarchical structure."""
        hierarchy = {}
        
        def build_level(depth: int, parent_id: str = "root") -> Dict:
            if depth > self.max_depth:
                return {}
            
            node = {
                "id": parent_id,
                "depth": depth,
                "children": [],
                "s_coords": {
                    "k": np.random.random(),
                    "t": np.random.random(),
                    "e": np.random.random()
                }
            }
            
            if depth < self.max_depth:
                for i in range(self.branching_factor):
                    child_id = f"{parent_id}_{i}"
                    child = build_level(depth + 1, child_id)
                    node["children"].append(child)
            
            return node
        
        hierarchy = build_level(0)
        return hierarchy
    
    def count_nodes_at_depth(self, hierarchy: Dict, target_depth: int) -> int:
        """Count nodes at a specific depth."""
        count = 0
        
        def traverse(node: Dict, current_depth: int):
            nonlocal count
            if current_depth == target_depth:
                count += 1
            for child in node.get("children", []):
                traverse(child, current_depth + 1)
        
        traverse(hierarchy, 0)
        return count
    
    def validate_3k_structure(self) -> Dict[int, Tuple[int, int]]:
        """Validate that node count follows 3^k."""
        hierarchy = self.generate_hierarchy()
        results = {}
        
        for depth in range(self.max_depth + 1):
            actual = self.count_nodes_at_depth(hierarchy, depth)
            expected = 3 ** depth
            results[depth] = (actual, expected)
        
        return results


def run_system_topology_experiment() -> Dict:
    """Run comprehensive system topology validation."""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {}
    }
    
    print("=" * 60)
    print("SYSTEM TOPOLOGY EXPERIMENT")
    print("=" * 60)
    
    # Experiment 1: Categorical Space and Completion Dynamics
    print("\n[1] Categorical Space and Completion Dynamics...")
    space = CategoricalSpace(n_states=200)
    
    # Complete states in order
    completion_attempts = 0
    successful_completions = 0
    
    for _ in range(500):  # Try many completions
        # Find uncompleted states with all predecessors completed
        candidates = []
        for sid, state in space.states.items():
            if not state.completed:
                all_pred_complete = all(
                    space.states[p].completed for p in state.predecessors
                )
                if all_pred_complete:
                    candidates.append(sid)
        
        if candidates:
            chosen = np.random.choice(candidates)
            completion_attempts += 1
            if space.complete_state(chosen):
                successful_completions += 1
    
    trajectory = space.get_completion_trajectory()
    
    results["experiments"]["categorical_space"] = {
        "n_states": len(space.states),
        "completion_attempts": completion_attempts,
        "successful_completions": successful_completions,
        "completion_rate": successful_completions / max(1, completion_attempts),
        "trajectory_length": len(trajectory),
        "trajectory_s_coords": [t.tolist() for t in trajectory[:50]]  # First 50
    }
    print(f"   Completed {successful_completions}/{completion_attempts} states")
    
    # Experiment 2: Equivalence Classes and Degeneracy
    print("\n[2] Equivalence Classes and Degeneracy...")
    analyzer = EquivalenceClassAnalyzer(space, observable_dim=0)
    classes = analyzer.compute_equivalence_classes()
    
    degeneracies = []
    richnesses = []
    for sid in list(space.states.keys())[:50]:
        deg = analyzer.compute_degeneracy(sid)
        rich = analyzer.compute_richness(sid)
        degeneracies.append(deg)
        richnesses.append(rich)
    
    results["experiments"]["equivalence_classes"] = {
        "n_classes": len(classes),
        "class_sizes": {str(k): len(v) for k, v in classes.items()},
        "mean_degeneracy": float(np.mean(degeneracies)),
        "std_degeneracy": float(np.std(degeneracies)),
        "mean_richness": float(np.mean(richnesses)),
        "degeneracy_distribution": degeneracies,
        "richness_distribution": richnesses
    }
    print(f"   Found {len(classes)} equivalence classes")
    print(f"   Mean degeneracy: {np.mean(degeneracies):.2f}")
    
    # Experiment 3: S-Distance Metric
    print("\n[3] S-Distance Metric Validation...")
    
    # Generate multiple trajectories
    trajectories = []
    for _ in range(5):
        test_space = CategoricalSpace(n_states=100)
        for _ in range(200):
            candidates = [
                sid for sid, s in test_space.states.items()
                if not s.completed and all(test_space.states[p].completed for p in s.predecessors)
            ]
            if candidates:
                test_space.complete_state(np.random.choice(candidates))
        trajectories.append(test_space.get_completion_trajectory())
    
    # Compute pairwise distances
    s_distances = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            d = space.compute_s_distance(trajectories[i], trajectories[j])
            s_distances.append({
                "pair": (i, j),
                "distance": d,
                "traj_lengths": (len(trajectories[i]), len(trajectories[j]))
            })
    
    # Verify triangle inequality
    triangle_violations = 0
    for i in range(len(trajectories)):
        for j in range(len(trajectories)):
            for k in range(len(trajectories)):
                if i != j and j != k and i != k:
                    d_ij = space.compute_s_distance(trajectories[i], trajectories[j])
                    d_jk = space.compute_s_distance(trajectories[j], trajectories[k])
                    d_ik = space.compute_s_distance(trajectories[i], trajectories[k])
                    if d_ik > d_ij + d_jk + 1e-10:  # Small tolerance
                        triangle_violations += 1
    
    results["experiments"]["s_distance"] = {
        "n_trajectories": len(trajectories),
        "pairwise_distances": s_distances,
        "mean_distance": float(np.mean([d["distance"] for d in s_distances])),
        "triangle_violations": triangle_violations,
        "metric_valid": triangle_violations == 0
    }
    print(f"   Triangle inequality violations: {triangle_violations}")
    
    # Experiment 4: 3^k Hierarchical Branching
    print("\n[4] 3^k Hierarchical Branching...")
    branching = HierarchicalBranching(max_depth=5)
    branching_results = branching.validate_3k_structure()
    
    all_match = True
    branching_data = []
    for depth, (actual, expected) in branching_results.items():
        matches = actual == expected
        all_match = all_match and matches
        branching_data.append({
            "depth": depth,
            "actual": actual,
            "expected": expected,
            "matches": matches
        })
        print(f"   Depth {depth}: actual={actual}, expected={expected}, match={matches}")
    
    results["experiments"]["hierarchical_branching"] = {
        "max_depth": branching.max_depth,
        "branching_factor": branching.branching_factor,
        "depth_counts": branching_data,
        "all_match_3k": all_match
    }
    
    # Experiment 5: Scale Ambiguity
    print("\n[5] Scale Ambiguity (Self-Similarity)...")
    hierarchy = branching.generate_hierarchy()
    
    # Collect S-coordinates at different levels
    level_coords = {i: [] for i in range(branching.max_depth + 1)}
    
    def collect_coords(node: Dict, depth: int):
        coords = node["s_coords"]
        level_coords[depth].append([coords["k"], coords["t"], coords["e"]])
        for child in node.get("children", []):
            collect_coords(child, depth + 1)
    
    collect_coords(hierarchy, 0)
    
    # Check if distributions are similar across levels (scale ambiguity)
    level_stats = {}
    for level, coords in level_coords.items():
        if coords:
            arr = np.array(coords)
            level_stats[level] = {
                "n_nodes": len(coords),
                "mean_k": float(np.mean(arr[:, 0])),
                "mean_t": float(np.mean(arr[:, 1])),
                "mean_e": float(np.mean(arr[:, 2])),
                "std_k": float(np.std(arr[:, 0])),
                "std_t": float(np.std(arr[:, 1])),
                "std_e": float(np.std(arr[:, 2]))
            }
    
    # Compute variance of means across levels (low = scale ambiguity)
    means_k = [s["mean_k"] for s in level_stats.values()]
    means_t = [s["mean_t"] for s in level_stats.values()]
    means_e = [s["mean_e"] for s in level_stats.values()]
    
    scale_ambiguity_score = 1.0 / (1.0 + np.var(means_k) + np.var(means_t) + np.var(means_e))
    
    results["experiments"]["scale_ambiguity"] = {
        "level_statistics": level_stats,
        "variance_of_means": {
            "k": float(np.var(means_k)),
            "t": float(np.var(means_t)),
            "e": float(np.var(means_e))
        },
        "scale_ambiguity_score": float(scale_ambiguity_score),
        "interpretation": "Higher score = more scale-invariant structure"
    }
    print(f"   Scale ambiguity score: {scale_ambiguity_score:.4f}")
    
    print("\n" + "=" * 60)
    print("SYSTEM TOPOLOGY EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return results


def save_results(results: Dict, output_dir: str = "results"):
    """Save results to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "system_topology_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    return output_path


if __name__ == "__main__":
    results = run_system_topology_experiment()
    save_results(results)

