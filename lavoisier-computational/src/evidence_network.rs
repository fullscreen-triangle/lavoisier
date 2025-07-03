use crate::{ComputationalConfig, ComputationalError, ComputationalResult};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

/// High-performance evidence network for massive spectral datasets
pub struct EvidenceNetwork {
    config: ComputationalConfig,
    node_pool: Arc<RwLock<NodePool>>,
    edge_index: Arc<Mutex<EdgeIndex>>,
    evidence_aggregator: EvidenceAggregator,
    memory_manager: MemoryManager,
}

impl EvidenceNetwork {
    pub fn new(config: &ComputationalConfig) -> ComputationalResult<Self> {
        let node_pool = Arc::new(RwLock::new(NodePool::new(config.num_threads)?));
        let edge_index = Arc::new(Mutex::new(EdgeIndex::new()?));
        let evidence_aggregator = EvidenceAggregator::new(config)?;
        let memory_manager = MemoryManager::new(config.memory_limit_gb)?;

        Ok(Self {
            config: config.clone(),
            node_pool,
            edge_index,
            evidence_aggregator,
            memory_manager,
        })
    }

    /// Build evidence network from massive spectral data using streaming approach
    pub fn build_evidence_network(
        &mut self,
        mz_data: &[f64],
        intensity_data: &[f64],
        optimal_noise_level: f64,
    ) -> ComputationalResult<EvidenceResult> {
        // Calculate processing chunks based on memory constraints
        let chunk_size = self.memory_manager.calculate_chunk_size(mz_data.len())?;
        let num_chunks = (mz_data.len() + chunk_size - 1) / chunk_size;

        // Phase 1: Build local evidence networks for each chunk
        let local_networks: Vec<LocalEvidenceNetwork> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = std::cmp::min(start + chunk_size, mz_data.len());

                self.build_local_network(
                    &mz_data[start..end],
                    &intensity_data[start..end],
                    optimal_noise_level,
                    chunk_idx,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Phase 2: Merge local networks into global evidence network
        let global_network = self.merge_local_networks(&local_networks)?;

        // Phase 3: Perform evidence aggregation and scoring
        let evidence_scores = self
            .evidence_aggregator
            .aggregate_evidence(&global_network)?;

        // Phase 4: Extract high-confidence spectral features
        let spectral_features =
            self.extract_spectral_features(&global_network, &evidence_scores)?;

        // Phase 5: Calculate network statistics
        let network_stats = self.calculate_network_statistics(&global_network)?;

        Ok(EvidenceResult {
            spectral_features,
            evidence_scores,
            network_stats,
            processing_chunks: num_chunks,
            total_nodes: global_network.nodes.len(),
            total_edges: global_network.edges.len(),
        })
    }

    /// Build local evidence network for a data chunk
    fn build_local_network(
        &self,
        mz_chunk: &[f64],
        intensity_chunk: &[f64],
        noise_level: f64,
        chunk_idx: usize,
    ) -> ComputationalResult<LocalEvidenceNetwork> {
        // Identify significant peaks in this chunk
        let peaks = self.identify_significant_peaks(mz_chunk, intensity_chunk, noise_level)?;

        // Create evidence nodes for each peak
        let mut nodes = HashMap::new();
        for peak in &peaks {
            let node = EvidenceNode {
                id: format!("chunk_{}_peak_{}", chunk_idx, peak.index),
                mz: peak.mz,
                intensity: peak.intensity,
                evidence_score: peak.significance_score,
                chunk_id: chunk_idx,
                local_index: peak.index,
                isotope_pattern: self.analyze_isotope_pattern(
                    mz_chunk,
                    intensity_chunk,
                    peak.index,
                )?,
                fragmentation_evidence: self.analyze_fragmentation_evidence(
                    mz_chunk,
                    intensity_chunk,
                    peak.index,
                )?,
            };
            nodes.insert(node.id.clone(), node);
        }

        // Create edges between related peaks
        let edges = self.create_local_edges(&nodes, mz_chunk, intensity_chunk)?;

        Ok(LocalEvidenceNetwork {
            chunk_id: chunk_idx,
            nodes,
            edges,
            peak_count: peaks.len(),
        })
    }

    /// Identify significant peaks using statistical analysis
    fn identify_significant_peaks(
        &self,
        mz_data: &[f64],
        intensity_data: &[f64],
        noise_level: f64,
    ) -> ComputationalResult<Vec<SignificantPeak>> {
        let mut peaks = Vec::new();
        let min_peak_separation = 0.01; // m/z units

        // Parallel peak detection with noise filtering
        let potential_peaks: Vec<_> = (1..mz_data.len() - 1)
            .into_par_iter()
            .filter_map(|i| {
                let current_intensity = intensity_data[i];
                let noise_threshold = noise_level * current_intensity;

                // Local maxima detection
                if current_intensity > intensity_data[i - 1]
                    && current_intensity > intensity_data[i + 1]
                    && current_intensity > noise_threshold
                {
                    // Calculate local signal-to-noise ratio
                    let local_noise = self.calculate_local_noise(intensity_data, i, 5);
                    let snr = if local_noise > 0.0 {
                        current_intensity / local_noise
                    } else {
                        0.0
                    };

                    // Statistical significance test
                    let significance_score = self.calculate_significance_score(
                        current_intensity,
                        local_noise,
                        intensity_data,
                        i,
                    );

                    if snr > 3.0 && significance_score > 0.01 {
                        Some(SignificantPeak {
                            index: i,
                            mz: mz_data[i],
                            intensity: current_intensity,
                            snr,
                            significance_score,
                            local_noise,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Remove overlapping peaks (keep highest intensity)
        let mut sorted_peaks = potential_peaks;
        sorted_peaks.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());

        let mut used_mz = HashSet::new();
        for peak in sorted_peaks {
            let mz_key = (peak.mz / min_peak_separation).round() as i64;
            if !used_mz.contains(&mz_key) {
                used_mz.insert(mz_key);
                peaks.push(peak);
            }
        }

        Ok(peaks)
    }

    /// Calculate local noise estimate
    fn calculate_local_noise(&self, intensity_data: &[f64], center: usize, window: usize) -> f64 {
        let start = center.saturating_sub(window);
        let end = std::cmp::min(center + window, intensity_data.len());

        let window_data = &intensity_data[start..end];
        let median = self.calculate_median(window_data);
        let mad = self.calculate_mad(window_data, median);

        mad * 1.4826 // Convert MAD to standard deviation equivalent
    }

    /// Calculate median of slice
    fn calculate_median(&self, data: &[f64]) -> f64 {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate median absolute deviation
    fn calculate_mad(&self, data: &[f64], median: f64) -> f64 {
        let mut deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = deviations.len() / 2;
        if deviations.len() % 2 == 0 {
            (deviations[mid - 1] + deviations[mid]) / 2.0
        } else {
            deviations[mid]
        }
    }

    /// Calculate statistical significance score
    fn calculate_significance_score(
        &self,
        intensity: f64,
        noise: f64,
        intensity_data: &[f64],
        index: usize,
    ) -> f64 {
        if noise <= 0.0 {
            return 0.0;
        }

        let z_score = (intensity - noise) / noise;
        let p_value = 2.0 * (1.0 - self.gaussian_cdf(z_score.abs()));

        // Convert p-value to significance score (higher is more significant)
        1.0 - p_value
    }

    /// Gaussian cumulative distribution function
    fn gaussian_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Analyze isotope pattern around a peak
    fn analyze_isotope_pattern(
        &self,
        mz_data: &[f64],
        intensity_data: &[f64],
        peak_index: usize,
    ) -> ComputationalResult<IsotopePattern> {
        let base_mz = mz_data[peak_index];
        let base_intensity = intensity_data[peak_index];

        // Look for isotope peaks (+1, +2, +3 Da)
        let mut isotope_peaks = Vec::new();

        for isotope_offset in 1..=3 {
            let target_mz = base_mz + isotope_offset as f64;

            // Find closest peak to target m/z
            if let Some(closest_idx) = self.find_closest_peak(mz_data, target_mz, 0.1) {
                let intensity_ratio = intensity_data[closest_idx] / base_intensity;

                isotope_peaks.push(IsotopePeak {
                    offset: isotope_offset,
                    mz: mz_data[closest_idx],
                    intensity_ratio,
                    theoretical_ratio: self.calculate_theoretical_isotope_ratio(isotope_offset),
                });
            }
        }

        Ok(IsotopePattern {
            base_mz,
            isotope_peaks,
            pattern_score: self.score_isotope_pattern(&isotope_peaks),
        })
    }

    /// Find closest peak to target m/z
    fn find_closest_peak(&self, mz_data: &[f64], target_mz: f64, tolerance: f64) -> Option<usize> {
        let mut best_idx = None;
        let mut best_distance = f64::INFINITY;

        for (i, &mz) in mz_data.iter().enumerate() {
            let distance = (mz - target_mz).abs();
            if distance <= tolerance && distance < best_distance {
                best_distance = distance;
                best_idx = Some(i);
            }
        }

        best_idx
    }

    /// Calculate theoretical isotope ratio
    fn calculate_theoretical_isotope_ratio(&self, offset: usize) -> f64 {
        // Simplified isotope ratios for organic compounds
        match offset {
            1 => 0.011,   // C13 ratio
            2 => 0.0006,  // C13+C13 ratio
            3 => 0.00001, // Higher order isotopes
            _ => 0.0,
        }
    }

    /// Score isotope pattern quality
    fn score_isotope_pattern(&self, isotope_peaks: &[IsotopePeak]) -> f64 {
        if isotope_peaks.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        for peak in isotope_peaks {
            let ratio_error =
                (peak.intensity_ratio - peak.theoretical_ratio).abs() / peak.theoretical_ratio;
            let peak_score = (1.0 - ratio_error).max(0.0);
            score += peak_score;
        }

        score / isotope_peaks.len() as f64
    }

    /// Analyze fragmentation evidence
    fn analyze_fragmentation_evidence(
        &self,
        mz_data: &[f64],
        intensity_data: &[f64],
        peak_index: usize,
    ) -> ComputationalResult<FragmentationEvidence> {
        let precursor_mz = mz_data[peak_index];
        let precursor_intensity = intensity_data[peak_index];

        // Look for potential fragment ions (lower m/z)
        let mut fragment_ions = Vec::new();

        for i in 0..peak_index {
            let fragment_mz = mz_data[i];
            let fragment_intensity = intensity_data[i];

            // Check for common neutral losses
            let mass_diff = precursor_mz - fragment_mz;
            let neutral_loss = self.identify_neutral_loss(mass_diff);

            if neutral_loss.is_some() && fragment_intensity > precursor_intensity * 0.05 {
                fragment_ions.push(FragmentIon {
                    mz: fragment_mz,
                    intensity_ratio: fragment_intensity / precursor_intensity,
                    neutral_loss,
                    mass_difference: mass_diff,
                });
            }
        }

        Ok(FragmentationEvidence {
            precursor_mz,
            fragment_ions,
            fragmentation_score: self.score_fragmentation_pattern(&fragment_ions),
        })
    }

    /// Identify common neutral losses
    fn identify_neutral_loss(&self, mass_diff: f64) -> Option<String> {
        let tolerance = 0.01;

        let common_losses = [
            (18.0105, "H2O"),
            (17.0265, "NH3"),
            (28.0313, "CO"),
            (44.0262, "CO2"),
            (46.0055, "COOH"),
            (32.0262, "CH3OH"),
        ];

        for (loss_mass, loss_name) in &common_losses {
            if (mass_diff - loss_mass).abs() < tolerance {
                return Some(loss_name.to_string());
            }
        }

        None
    }

    /// Score fragmentation pattern
    fn score_fragmentation_pattern(&self, fragment_ions: &[FragmentIon]) -> f64 {
        if fragment_ions.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        for fragment in fragment_ions {
            let intensity_score = fragment.intensity_ratio.min(1.0);
            let neutral_loss_bonus = if fragment.neutral_loss.is_some() {
                0.5
            } else {
                0.0
            };
            score += intensity_score + neutral_loss_bonus;
        }

        score / fragment_ions.len() as f64
    }

    /// Create edges between related peaks in local network
    fn create_local_edges(
        &self,
        nodes: &HashMap<String, EvidenceNode>,
        mz_data: &[f64],
        intensity_data: &[f64],
    ) -> ComputationalResult<Vec<EvidenceEdge>> {
        let mut edges = Vec::new();
        let node_list: Vec<_> = nodes.values().collect();

        // Create edges between nodes with relationships
        for i in 0..node_list.len() {
            for j in i + 1..node_list.len() {
                let node1 = &node_list[i];
                let node2 = &node_list[j];

                let edge_weight = self.calculate_edge_weight(node1, node2)?;

                if edge_weight > 0.1 {
                    // Minimum edge weight threshold
                    edges.push(EvidenceEdge {
                        from_node: node1.id.clone(),
                        to_node: node2.id.clone(),
                        weight: edge_weight,
                        relationship_type: self.determine_relationship_type(node1, node2),
                    });
                }
            }
        }

        Ok(edges)
    }

    /// Calculate edge weight between two nodes
    fn calculate_edge_weight(
        &self,
        node1: &EvidenceNode,
        node2: &EvidenceNode,
    ) -> ComputationalResult<f64> {
        let mz_diff = (node1.mz - node2.mz).abs();

        // Isotope relationship
        if mz_diff >= 0.99 && mz_diff <= 3.01 {
            return Ok(0.8);
        }

        // Fragmentation relationship
        if mz_diff > 10.0 {
            let neutral_loss_score = if self.identify_neutral_loss(mz_diff).is_some() {
                0.7
            } else {
                0.3
            };
            return Ok(neutral_loss_score);
        }

        // Intensity correlation
        let intensity_ratio =
            (node1.intensity / node2.intensity).min(node2.intensity / node1.intensity);
        let correlation_score = intensity_ratio * 0.5;

        Ok(correlation_score)
    }

    /// Determine relationship type between nodes
    fn determine_relationship_type(
        &self,
        node1: &EvidenceNode,
        node2: &EvidenceNode,
    ) -> RelationshipType {
        let mz_diff = (node1.mz - node2.mz).abs();

        if mz_diff >= 0.99 && mz_diff <= 3.01 {
            RelationshipType::Isotope
        } else if mz_diff > 10.0 {
            RelationshipType::Fragmentation
        } else {
            RelationshipType::Correlation
        }
    }

    /// Merge local networks into global network
    fn merge_local_networks(
        &self,
        local_networks: &[LocalEvidenceNetwork],
    ) -> ComputationalResult<GlobalEvidenceNetwork> {
        let mut global_nodes = HashMap::new();
        let mut global_edges = Vec::new();

        // Merge all local nodes
        for network in local_networks {
            for (node_id, node) in &network.nodes {
                global_nodes.insert(node_id.clone(), node.clone());
            }
            global_edges.extend(network.edges.clone());
        }

        // Create inter-chunk edges
        let inter_chunk_edges = self.create_inter_chunk_edges(local_networks)?;
        global_edges.extend(inter_chunk_edges);

        Ok(GlobalEvidenceNetwork {
            nodes: global_nodes,
            edges: global_edges,
        })
    }

    /// Create edges between chunks
    fn create_inter_chunk_edges(
        &self,
        local_networks: &[LocalEvidenceNetwork],
    ) -> ComputationalResult<Vec<EvidenceEdge>> {
        let mut inter_edges = Vec::new();

        // For now, simple approach - could be optimized for 100GB+ datasets
        for i in 0..local_networks.len() {
            for j in i + 1..local_networks.len() {
                let edges =
                    self.find_cross_chunk_relationships(&local_networks[i], &local_networks[j])?;
                inter_edges.extend(edges);
            }
        }

        Ok(inter_edges)
    }

    /// Find relationships between nodes in different chunks
    fn find_cross_chunk_relationships(
        &self,
        network1: &LocalEvidenceNetwork,
        network2: &LocalEvidenceNetwork,
    ) -> ComputationalResult<Vec<EvidenceEdge>> {
        let mut edges = Vec::new();

        // Look for isotope and fragmentation relationships across chunks
        for node1 in network1.nodes.values() {
            for node2 in network2.nodes.values() {
                let edge_weight = self.calculate_edge_weight(node1, node2)?;

                if edge_weight > 0.2 {
                    // Higher threshold for inter-chunk edges
                    edges.push(EvidenceEdge {
                        from_node: node1.id.clone(),
                        to_node: node2.id.clone(),
                        weight: edge_weight,
                        relationship_type: self.determine_relationship_type(node1, node2),
                    });
                }
            }
        }

        Ok(edges)
    }

    /// Extract high-confidence spectral features
    fn extract_spectral_features(
        &self,
        network: &GlobalEvidenceNetwork,
        evidence_scores: &HashMap<String, f64>,
    ) -> ComputationalResult<Vec<SpectralFeature>> {
        let mut features = Vec::new();

        for (node_id, node) in &network.nodes {
            let evidence_score = evidence_scores.get(node_id).unwrap_or(&0.0);

            if *evidence_score > 0.5 {
                // High confidence threshold
                features.push(SpectralFeature {
                    mz: node.mz,
                    intensity: node.intensity,
                    evidence_score: *evidence_score,
                    isotope_pattern: node.isotope_pattern.clone(),
                    fragmentation_evidence: node.fragmentation_evidence.clone(),
                    confidence_level: self.classify_confidence_level(*evidence_score),
                });
            }
        }

        // Sort by evidence score
        features.sort_by(|a, b| b.evidence_score.partial_cmp(&a.evidence_score).unwrap());

        Ok(features)
    }

    /// Classify confidence level
    fn classify_confidence_level(&self, evidence_score: f64) -> ConfidenceLevel {
        if evidence_score > 0.9 {
            ConfidenceLevel::High
        } else if evidence_score > 0.7 {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }

    /// Calculate network statistics
    fn calculate_network_statistics(
        &self,
        network: &GlobalEvidenceNetwork,
    ) -> ComputationalResult<NetworkStatistics> {
        let node_count = network.nodes.len();
        let edge_count = network.edges.len();

        let connectivity = if node_count > 1 {
            edge_count as f64 / (node_count * (node_count - 1) / 2) as f64
        } else {
            0.0
        };

        let avg_evidence_score = if !network.nodes.is_empty() {
            network
                .nodes
                .values()
                .map(|n| n.evidence_score)
                .sum::<f64>()
                / node_count as f64
        } else {
            0.0
        };

        Ok(NetworkStatistics {
            node_count,
            edge_count,
            connectivity,
            avg_evidence_score,
            cluster_count: self.count_clusters(network),
        })
    }

    /// Count clusters in network
    fn count_clusters(&self, network: &GlobalEvidenceNetwork) -> usize {
        // Simple connected components counting
        let mut visited = HashSet::new();
        let mut cluster_count = 0;

        for node_id in network.nodes.keys() {
            if !visited.contains(node_id) {
                self.dfs_visit(node_id, network, &mut visited);
                cluster_count += 1;
            }
        }

        cluster_count
    }

    /// Depth-first search for connected components
    fn dfs_visit(
        &self,
        node_id: &str,
        network: &GlobalEvidenceNetwork,
        visited: &mut HashSet<String>,
    ) {
        visited.insert(node_id.to_string());

        for edge in &network.edges {
            let neighbor = if edge.from_node == node_id {
                &edge.to_node
            } else if edge.to_node == node_id {
                &edge.from_node
            } else {
                continue;
            };

            if !visited.contains(neighbor) {
                self.dfs_visit(neighbor, network, visited);
            }
        }
    }
}

/// Evidence aggregator for scoring spectral evidence
struct EvidenceAggregator {
    config: ComputationalConfig,
}

impl EvidenceAggregator {
    fn new(config: &ComputationalConfig) -> ComputationalResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Aggregate evidence across the network
    fn aggregate_evidence(
        &self,
        network: &GlobalEvidenceNetwork,
    ) -> ComputationalResult<HashMap<String, f64>> {
        let mut evidence_scores = HashMap::new();

        // Parallel computation of evidence scores
        let scores: Vec<_> = network
            .nodes
            .par_iter()
            .map(|(node_id, node)| {
                let local_score = self.calculate_local_evidence_score(node);
                let network_score = self.calculate_network_evidence_score(node, network);
                let combined_score = (local_score + network_score) / 2.0;
                (node_id.clone(), combined_score)
            })
            .collect();

        for (node_id, score) in scores {
            evidence_scores.insert(node_id, score);
        }

        Ok(evidence_scores)
    }

    /// Calculate local evidence score for a node
    fn calculate_local_evidence_score(&self, node: &EvidenceNode) -> f64 {
        let base_score = node.evidence_score;
        let isotope_bonus = node.isotope_pattern.pattern_score * 0.3;
        let fragmentation_bonus = node.fragmentation_evidence.fragmentation_score * 0.2;

        (base_score + isotope_bonus + fragmentation_bonus).min(1.0)
    }

    /// Calculate network evidence score considering connections
    fn calculate_network_evidence_score(
        &self,
        node: &EvidenceNode,
        network: &GlobalEvidenceNetwork,
    ) -> f64 {
        let mut network_score = 0.0;
        let mut connection_count = 0;

        for edge in &network.edges {
            if edge.from_node == node.id || edge.to_node == node.id {
                network_score += edge.weight;
                connection_count += 1;
            }
        }

        if connection_count > 0 {
            network_score / connection_count as f64
        } else {
            0.0
        }
    }
}

/// Memory manager for large dataset processing
struct MemoryManager {
    memory_limit_gb: f64,
}

impl MemoryManager {
    fn new(memory_limit_gb: f64) -> ComputationalResult<Self> {
        Ok(Self { memory_limit_gb })
    }

    fn calculate_chunk_size(&self, total_size: usize) -> ComputationalResult<usize> {
        let memory_limit_bytes = (self.memory_limit_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let bytes_per_point = 16; // 8 bytes for m/z + 8 bytes for intensity
        let safety_factor = 8; // Account for intermediate calculations and network structures

        let max_chunk_size = memory_limit_bytes / (bytes_per_point * safety_factor);
        Ok(std::cmp::min(max_chunk_size, total_size))
    }
}

/// Node pool for efficient memory management
struct NodePool {
    nodes: Vec<EvidenceNode>,
    free_indices: Vec<usize>,
}

impl NodePool {
    fn new(capacity: usize) -> ComputationalResult<Self> {
        Ok(Self {
            nodes: Vec::with_capacity(capacity * 1000),
            free_indices: Vec::new(),
        })
    }
}

/// Edge index for fast edge lookups
struct EdgeIndex {
    adjacency_list: HashMap<String, Vec<String>>,
}

impl EdgeIndex {
    fn new() -> ComputationalResult<Self> {
        Ok(Self {
            adjacency_list: HashMap::new(),
        })
    }
}

/// Evidence node representing a spectral peak
#[derive(Debug, Clone)]
pub struct EvidenceNode {
    pub id: String,
    pub mz: f64,
    pub intensity: f64,
    pub evidence_score: f64,
    pub chunk_id: usize,
    pub local_index: usize,
    pub isotope_pattern: IsotopePattern,
    pub fragmentation_evidence: FragmentationEvidence,
}

/// Evidence edge connecting related nodes
#[derive(Debug, Clone)]
pub struct EvidenceEdge {
    pub from_node: String,
    pub to_node: String,
    pub weight: f64,
    pub relationship_type: RelationshipType,
}

/// Relationship types between nodes
#[derive(Debug, Clone)]
pub enum RelationshipType {
    Isotope,
    Fragmentation,
    Correlation,
}

/// Local evidence network for a chunk
#[derive(Debug, Clone)]
struct LocalEvidenceNetwork {
    chunk_id: usize,
    nodes: HashMap<String, EvidenceNode>,
    edges: Vec<EvidenceEdge>,
    peak_count: usize,
}

/// Global evidence network
#[derive(Debug, Clone)]
struct GlobalEvidenceNetwork {
    nodes: HashMap<String, EvidenceNode>,
    edges: Vec<EvidenceEdge>,
}

/// Significant peak identified in spectral data
#[derive(Debug, Clone)]
struct SignificantPeak {
    index: usize,
    mz: f64,
    intensity: f64,
    snr: f64,
    significance_score: f64,
    local_noise: f64,
}

/// Isotope pattern analysis
#[derive(Debug, Clone)]
pub struct IsotopePattern {
    pub base_mz: f64,
    pub isotope_peaks: Vec<IsotopePeak>,
    pub pattern_score: f64,
}

/// Individual isotope peak
#[derive(Debug, Clone)]
pub struct IsotopePeak {
    pub offset: usize,
    pub mz: f64,
    pub intensity_ratio: f64,
    pub theoretical_ratio: f64,
}

/// Fragmentation evidence
#[derive(Debug, Clone)]
pub struct FragmentationEvidence {
    pub precursor_mz: f64,
    pub fragment_ions: Vec<FragmentIon>,
    pub fragmentation_score: f64,
}

/// Fragment ion
#[derive(Debug, Clone)]
pub struct FragmentIon {
    pub mz: f64,
    pub intensity_ratio: f64,
    pub neutral_loss: Option<String>,
    pub mass_difference: f64,
}

/// Spectral feature extracted from evidence network
#[derive(Debug, Clone)]
pub struct SpectralFeature {
    pub mz: f64,
    pub intensity: f64,
    pub evidence_score: f64,
    pub isotope_pattern: IsotopePattern,
    pub fragmentation_evidence: FragmentationEvidence,
    pub confidence_level: ConfidenceLevel,
}

/// Confidence level classification
#[derive(Debug, Clone)]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub connectivity: f64,
    pub avg_evidence_score: f64,
    pub cluster_count: usize,
}

/// Final evidence network result
#[derive(Debug, Clone)]
pub struct EvidenceResult {
    pub spectral_features: Vec<SpectralFeature>,
    pub evidence_scores: HashMap<String, f64>,
    pub network_stats: NetworkStatistics,
    pub processing_chunks: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
}
