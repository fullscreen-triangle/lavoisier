class GPUOscillationHarvester:
    """
    GPU memory bandwidth oscillations =
    large-scale frequency coupling across entire experiment!
    """

    def harvest_experiment_wide_coupling(self, experiment_kb):
        """
        GPU processing of entire experiment →
        harvest bandwidth oscillations = coupling matrix!
        """
        gpu_trace = []

        with GPUMonitor() as monitor:
            # Process all peptides on GPU
            for peptide in experiment_kb.peptides:
                # GPU memory access pattern
                bandwidth = monitor.get_current_bandwidth()

                gpu_trace.append({
                    'peptide': peptide.sequence,
                    'bandwidth': bandwidth,
                    'memory_pattern': monitor.get_access_pattern()
                })

        # GPU bandwidth oscillations = experiment-wide coupling!
        coupling_distribution = analyze_bandwidth_oscillations(gpu_trace)

        return coupling_distribution
