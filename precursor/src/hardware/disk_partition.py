class DiskIOHarvester:
    """
    Disk I/O timing = peptide fragmentation sequences!

    - Sequential reads = linear fragmentation
    - Random access = complex fragmentation patterns
    - I/O latency = fragmentation kinetics
    """

    def harvest_fragmentation_patterns(self, mzml_file):
        """
        Monitor disk I/O while reading MS/MS data
        = fragmentation sequence timing!
        """
        io_trace = []

        with IOMonitor() as monitor:
            # Read spectrum from disk
            spectrum = read_mzml_spectrum(mzml_file)

            # I/O pattern = fragmentation sequence
            io_pattern = monitor.get_io_pattern()

            for fragment in spectrum.fragments:
                # I/O timing for this fragment
                io_time = io_pattern.get_time_for_mz(fragment.mz)

                io_trace.append({
                    'fragment_mz': fragment.mz,
                    'io_latency': io_time,
                    'sequential': is_sequential_access(io_pattern)
                })

        # I/O pattern encodes fragmentation kinetics!
        return fragmentation_sequence_from_io(io_trace)
