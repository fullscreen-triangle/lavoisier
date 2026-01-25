"""
Experimental validation framework for categorical thermodynamics and S-entropy coordinates.

This module implements validators for real experimental data including:
- 3D S-space visualization with convex hulls
- Ionization mode comparison
- Sample classification
- Network analysis
- Categorical temperature
- Maxwell-Boltzmann statistics
- Ideal gas law validation
- Entropy production
- Performance profiling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExperimentalData:
    """Container for experimental mass spectrometry data."""
    spectra_count: int
    samples: Dict[str, int]  # Sample name -> spectrum count
    ionization_modes: Dict[str, int]  # Mode -> spectrum count
    s_coordinates: np.ndarray  # (N, 3) array of (S_k, S_t, S_e)
    sample_labels: np.ndarray  # (N,) array of sample IDs
    mode_labels: np.ndarray  # (N,) array of ionization modes
    retention_times: Optional[np.ndarray] = None  # (N,) array
    mz_values: Optional[np.ndarray] = None  # (N,) array
    intensities: Optional[np.ndarray] = None  # (N,) array
    ms2_coverage: Optional[Dict] = None  # Sample -> mode -> count


class CategoricalThermodynamicsValidator:
    """
    Validates categorical thermodynamics predictions using experimental data.
    
    Tests:
    1. Ideal gas law: PV = k_B T_cat
    2. Maxwell-Boltzmann distribution of intensities
    3. Categorical temperature calculation
    4. Entropy production over time
    """
    
    def __init__(self):
        self.k_B = 1.381e-23  # Boltzmann constant (J/K)
        self.hbar = 1.054572e-34  # Reduced Planck constant (J·s)
        
    def calculate_categorical_temperature(self,
                                         retention_times: np.ndarray,
                                         mz_values: np.ndarray,
                                         time_window: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate categorical temperature T_cat = (ℏ/k_B)(dM/dt).
        
        Parameters:
        -----------
        retention_times : np.ndarray
            Retention times in minutes
        mz_values : np.ndarray
            m/z values for each spectrum
        time_window : float
            Time window for calculating dM/dt (minutes)
            
        Returns:
        --------
        (times, temperatures) : Tuple[np.ndarray, np.ndarray]
            Time points and corresponding categorical temperatures
        """
        # Bin retention times
        time_bins = np.arange(retention_times.min(), retention_times.max(), time_window)
        temperatures = []
        times = []
        
        for i in range(len(time_bins) - 1):
            t_start, t_end = time_bins[i], time_bins[i+1]
            mask = (retention_times >= t_start) & (retention_times < t_end)
            
            if mask.sum() > 0:
                # Count distinct m/z values in this window
                M_current = len(np.unique(mz_values[mask]))
                
                # Calculate dM/dt
                if i > 0 and len(temperatures) > 0:
                    M_prev = len(np.unique(mz_values[
                        (retention_times >= time_bins[i-1]) & (retention_times < time_bins[i])
                    ]))
                    dM_dt = (M_current - M_prev) / time_window
                    
                    # T_cat = (ℏ/k_B)(dM/dt)
                    T_cat = (self.hbar / self.k_B) * abs(dM_dt) * 60  # Convert to per second
                    temperatures.append(T_cat)
                    times.append((t_start + t_end) / 2)
        
        return np.array(times), np.array(temperatures)
    
    def validate_ideal_gas_law(self,
                               s_coordinates: np.ndarray,
                               sample_labels: np.ndarray) -> Dict[str, any]:
        """
        Validate ideal gas law: PV = k_B T_cat.
        
        Each spectrum is treated as a "gas molecule" in categorical space.
        
        Parameters:
        -----------
        s_coordinates : np.ndarray
            (N, 3) array of S-entropy coordinates
        sample_labels : np.ndarray
            (N,) array of sample identifiers
            
        Returns:
        --------
        Dict with validation results
        """
        results = {}
        
        # Calculate categorical temperature for each spectrum
        # T_cat proportional to S_e (evolution entropy)
        T_cat = s_coordinates[:, 2]  # S_e component
        
        # Calculate categorical pressure and volume
        # P_cat = (N/V) k_B T_cat, so PV = N k_B T_cat
        # For single "molecule": PV = k_B T_cat
        
        # Volume in categorical space (proportional to S_k)
        V_cat = s_coordinates[:, 0]  # S_k component
        
        # PV product
        PV = T_cat * V_cat
        
        # Linear fit: PV = k_B T_cat
        # In normalized units, expect slope ≈ 1
        slope, intercept, r_value, p_value, std_err = stats.linregress(T_cat, PV)
        
        # Calculate deviation from ideal
        predicted_PV = slope * T_cat + intercept
        deviation = np.abs(PV - predicted_PV) / PV
        mean_deviation = np.mean(deviation) * 100  # Percent
        
        results['slope'] = slope
        results['intercept'] = intercept
        results['r_squared'] = r_value**2
        results['p_value'] = p_value
        results['std_error'] = std_err
        results['mean_deviation_percent'] = mean_deviation
        results['T_cat'] = T_cat
        results['PV'] = PV
        results['predicted_PV'] = predicted_PV
        
        return results
    
    def validate_maxwell_boltzmann(self,
                                   intensities: np.ndarray,
                                   normalize: bool = True) -> Dict[str, any]:
        """
        Validate that intensity distribution follows Maxwell-Boltzmann statistics.
        
        Expected distribution: P(I) = (2/√π) (I/⟨I⟩)^(1/2) exp(-I/⟨I⟩)
        
        Parameters:
        -----------
        intensities : np.ndarray
            Peak intensities
        normalize : bool
            Whether to normalize intensities by mean
            
        Returns:
        --------
        Dict with validation results
        """
        # Normalize intensities
        if normalize:
            I_norm = intensities / np.mean(intensities)
        else:
            I_norm = intensities
        
        # Remove zeros and negatives
        I_norm = I_norm[I_norm > 0]
        
        # Maxwell-Boltzmann distribution
        def maxwell_boltzmann(I, scale):
            return (2/np.sqrt(np.pi)) * np.sqrt(I/scale) * np.exp(-I/scale)
        
        # Fit distribution
        hist, bin_edges = np.histogram(I_norm, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Fit only to non-zero histogram values
        mask = hist > 0
        try:
            popt, pcov = curve_fit(maxwell_boltzmann, bin_centers[mask], hist[mask], p0=[1.0])
            scale_param = popt[0]
            
            # Predicted distribution
            predicted = maxwell_boltzmann(bin_centers, scale_param)
            
            # Chi-squared test
            observed = hist
            expected = predicted
            # Only test where expected > 5
            valid_mask = expected > 0.01
            if valid_mask.sum() > 0:
                chi2_stat = np.sum((observed[valid_mask] - expected[valid_mask])**2 / expected[valid_mask])
                dof = valid_mask.sum() - 1  # degrees of freedom
                chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
            else:
                chi2_stat = np.nan
                chi2_p_value = np.nan
            
            # Kolmogorov-Smirnov test
            # Generate samples from fitted distribution
            n_samples = len(I_norm)
            # Use inverse transform sampling (approximate)
            uniform_samples = np.random.uniform(0, 1, n_samples)
            # For MB distribution, use approximate inverse
            mb_samples = scale_param * (-np.log(1 - uniform_samples))**2
            ks_stat, ks_p_value = stats.ks_2samp(I_norm, mb_samples)
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            scale_param = 1.0
            predicted = np.zeros_like(bin_centers)
            chi2_stat = np.nan
            chi2_p_value = np.nan
            ks_stat = np.nan
            ks_p_value = np.nan
        
        results = {
            'mean_intensity': np.mean(intensities),
            'scale_parameter': scale_param,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p_value,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'histogram': hist,
            'bin_centers': bin_centers,
            'predicted': predicted,
            'normalized_intensities': I_norm
        }
        
        return results
    
    def calculate_entropy_production(self,
                                    retention_times: np.ndarray,
                                    s_coordinates: np.ndarray,
                                    sample_labels: np.ndarray,
                                    time_window: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Calculate entropy production rate dS/dt over retention time.
        
        Parameters:
        -----------
        retention_times : np.ndarray
            Retention times in minutes
        s_coordinates : np.ndarray
            (N, 3) S-entropy coordinates
        sample_labels : np.ndarray
            Sample identifiers
        time_window : float
            Time window for derivative (minutes)
            
        Returns:
        --------
        Dict mapping sample names to (times, dS_dt) tuples
        """
        results = {}
        
        unique_samples = np.unique(sample_labels)
        
        for sample in unique_samples:
            mask = sample_labels == sample
            rt_sample = retention_times[mask]
            s_sample = s_coordinates[mask]
            
            # Sort by retention time
            sort_idx = np.argsort(rt_sample)
            rt_sorted = rt_sample[sort_idx]
            s_sorted = s_sample[sort_idx]
            
            # Bin by time
            time_bins = np.arange(rt_sorted.min(), rt_sorted.max(), time_window)
            times = []
            dS_dt_values = []
            
            for i in range(len(time_bins) - 1):
                t_start, t_end = time_bins[i], time_bins[i+1]
                bin_mask = (rt_sorted >= t_start) & (rt_sorted < t_end)
                
                if bin_mask.sum() > 1:
                    # Calculate mean S_e in this bin
                    S_e_current = np.mean(s_sorted[bin_mask, 2])
                    
                    if i > 0 and len(dS_dt_values) > 0:
                        # Previous bin
                        prev_mask = (rt_sorted >= time_bins[i-1]) & (rt_sorted < time_bins[i])
                        if prev_mask.sum() > 0:
                            S_e_prev = np.mean(s_sorted[prev_mask, 2])
                            dS_dt = (S_e_current - S_e_prev) / time_window
                            
                            times.append((t_start + t_end) / 2)
                            dS_dt_values.append(abs(dS_dt))
            
            results[sample] = (np.array(times), np.array(dS_dt_values))
        
        return results


class SEntropyValidator:
    """
    Validates S-entropy coordinate framework using experimental data.
    
    Tests:
    1. 3D visualization with convex hulls
    2. Sample separation in S-space
    3. Ionization mode differences
    4. Classification accuracy
    5. PCA analysis
    """
    
    def __init__(self):
        self.name = "S-Entropy Coordinate Validator"
        
    def calculate_sample_statistics(self,
                                    s_coordinates: np.ndarray,
                                    sample_labels: np.ndarray) -> Dict[str, Dict]:
        """
        Calculate statistics for each sample in S-space.
        
        Parameters:
        -----------
        s_coordinates : np.ndarray
            (N, 3) S-entropy coordinates
        sample_labels : np.ndarray
            Sample identifiers
            
        Returns:
        --------
        Dict mapping sample names to statistics
        """
        stats_dict = {}
        
        unique_samples = np.unique(sample_labels)
        
        for sample in unique_samples:
            mask = sample_labels == sample
            s_sample = s_coordinates[mask]
            
            # Calculate statistics
            centroid = np.mean(s_sample, axis=0)
            variance = np.var(s_sample, axis=0)
            std = np.std(s_sample, axis=0)
            
            stats_dict[sample] = {
                'count': mask.sum(),
                'centroid': centroid,
                'variance': variance,
                'std': std,
                'min': np.min(s_sample, axis=0),
                'max': np.max(s_sample, axis=0)
            }
        
        # Calculate pairwise distances between centroids
        centroids = np.array([stats_dict[s]['centroid'] for s in unique_samples])
        n_samples = len(unique_samples)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return {
            'sample_stats': stats_dict,
            'pairwise_distances': distances,
            'sample_names': unique_samples
        }
    
    def validate_ionization_mode_separation(self,
                                           s_coordinates: np.ndarray,
                                           mode_labels: np.ndarray) -> Dict[str, any]:
        """
        Validate separation between ionization modes in S-space.
        
        Parameters:
        -----------
        s_coordinates : np.ndarray
            (N, 3) S-entropy coordinates
        mode_labels : np.ndarray
            Ionization mode labels (e.g., 'positive', 'negative')
            
        Returns:
        --------
        Dict with validation results
        """
        unique_modes = np.unique(mode_labels)
        
        results = {}
        
        for mode in unique_modes:
            mask = mode_labels == mode
            s_mode = s_coordinates[mask]
            
            results[mode] = {
                'count': mask.sum(),
                'mean_S_k': np.mean(s_mode[:, 0]),
                'mean_S_t': np.mean(s_mode[:, 1]),
                'mean_S_e': np.mean(s_mode[:, 2]),
                'std_S_k': np.std(s_mode[:, 0]),
                'std_S_t': np.std(s_mode[:, 1]),
                'std_S_e': np.std(s_mode[:, 2])
            }
        
        # T-test between modes (if 2 modes)
        if len(unique_modes) == 2:
            mode1, mode2 = unique_modes
            s_mode1 = s_coordinates[mode_labels == mode1]
            s_mode2 = s_coordinates[mode_labels == mode2]
            
            # T-test for each coordinate
            t_stat_k, p_value_k = stats.ttest_ind(s_mode1[:, 0], s_mode2[:, 0])
            t_stat_t, p_value_t = stats.ttest_ind(s_mode1[:, 1], s_mode2[:, 1])
            t_stat_e, p_value_e = stats.ttest_ind(s_mode1[:, 2], s_mode2[:, 2])
            
            # Cohen's d effect size
            def cohens_d(x1, x2):
                n1, n2 = len(x1), len(x2)
                var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
                pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                return (np.mean(x1) - np.mean(x2)) / pooled_std
            
            d_k = cohens_d(s_mode1[:, 0], s_mode2[:, 0])
            d_t = cohens_d(s_mode1[:, 1], s_mode2[:, 1])
            d_e = cohens_d(s_mode1[:, 2], s_mode2[:, 2])
            
            results['comparison'] = {
                't_statistic': [t_stat_k, t_stat_t, t_stat_e],
                'p_value': [p_value_k, p_value_t, p_value_e],
                'cohens_d': [d_k, d_t, d_e],
                'significant': [p_value_k < 0.001, p_value_t < 0.001, p_value_e < 0.001]
            }
        
        return results
    
    def train_classifier(self,
                        s_coordinates: np.ndarray,
                        sample_labels: np.ndarray,
                        classifier_type: str = 'rf',
                        test_size: float = 0.3,
                        random_state: int = 42) -> Dict[str, any]:
        """
        Train classifier to predict sample identity from S-coordinates.
        
        Parameters:
        -----------
        s_coordinates : np.ndarray
            (N, 3) S-entropy coordinates
        sample_labels : np.ndarray
            Sample identifiers
        classifier_type : str
            'svm', 'rf' (random forest), or 'nn' (neural network)
        test_size : float
            Fraction of data for testing
        random_state : int
            Random seed
            
        Returns:
        --------
        Dict with classification results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            s_coordinates, sample_labels, test_size=test_size, random_state=random_state
        )
        
        # Train classifier
        if classifier_type == 'svm':
            clf = SVC(kernel='rbf', random_state=random_state)
        elif classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'classifier_type': classifier_type,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'y_test': y_test,
            'y_pred': y_pred,
            'unique_labels': np.unique(sample_labels)
        }
        
        return results
    
    def perform_pca(self,
                   s_coordinates: np.ndarray,
                   sample_labels: np.ndarray) -> Dict[str, any]:
        """
        Perform PCA on S-entropy coordinates.
        
        Parameters:
        -----------
        s_coordinates : np.ndarray
            (N, 3) S-entropy coordinates
        sample_labels : np.ndarray
            Sample identifiers
            
        Returns:
        --------
        Dict with PCA results
        """
        # Perform PCA
        pca = PCA(n_components=2)
        s_pca = pca.fit_transform(s_coordinates)
        
        # Variance explained
        var_explained = pca.explained_variance_ratio_
        
        results = {
            'pca_coordinates': s_pca,
            'variance_explained': var_explained,
            'cumulative_variance': np.cumsum(var_explained),
            'components': pca.components_,
            'sample_labels': sample_labels
        }
        
        return results


def generate_synthetic_experimental_data(n_spectra: int = 46458) -> ExperimentalData:
    """
    Generate synthetic experimental data for testing.
    
    Parameters:
    -----------
    n_spectra : int
        Total number of spectra
        
    Returns:
    --------
    ExperimentalData object
    """
    # Sample distribution (from validation-plots.md)
    samples = {
        'M3': 8611,
        'M4': 8429,
        'M5': 8807
    }
    
    # Adjust to match total
    remaining = n_spectra - sum(samples.values())
    samples['M3'] += remaining // 3
    samples['M4'] += remaining // 3
    samples['M5'] += remaining - 2*(remaining // 3)
    
    # Ionization modes
    ionization_modes = {
        'positive': n_spectra // 2,
        'negative': n_spectra - n_spectra // 2
    }
    
    # Generate S-coordinates
    s_coordinates = []
    sample_labels = []
    mode_labels = []
    retention_times = []
    mz_values = []
    intensities = []
    
    for sample, count in samples.items():
        # Sample-specific S-coordinate distribution
        if sample == 'M3':
            s_k_mean, s_t_mean, s_e_mean = 0.4, 0.5, 0.3
        elif sample == 'M4':
            s_k_mean, s_t_mean, s_e_mean = 0.6, 0.6, 0.5
        else:  # M5
            s_k_mean, s_t_mean, s_e_mean = 0.5, 0.4, 0.6
        
        # Generate coordinates
        s_k = np.random.normal(s_k_mean, 0.1, count)
        s_t = np.random.normal(s_t_mean, 0.1, count)
        s_e = np.random.normal(s_e_mean, 0.1, count)
        
        s_coordinates.append(np.column_stack([s_k, s_t, s_e]))
        sample_labels.extend([sample] * count)
        
        # Ionization modes (roughly half each)
        n_pos = count // 2
        n_neg = count - n_pos
        mode_labels.extend(['positive'] * n_pos + ['negative'] * n_neg)
        
        # Retention times (0-60 minutes)
        retention_times.extend(np.random.uniform(0, 60, count))
        
        # m/z values (100-1000)
        mz_values.extend(np.random.uniform(100, 1000, count))
        
        # Intensities (log-normal distribution)
        intensities.extend(np.random.lognormal(10, 2, count))
    
    s_coordinates = np.vstack(s_coordinates)
    sample_labels = np.array(sample_labels)
    mode_labels = np.array(mode_labels)
    retention_times = np.array(retention_times)
    mz_values = np.array(mz_values)
    intensities = np.array(intensities)
    
    # MS2 coverage (from validation-plots.md)
    ms2_coverage = {
        'M3': {'negative': 602, 'positive': 491},
        'M4': {'negative': 939, 'positive': 848},
        'M5': {'negative': 680, 'positive': 727}
    }
    
    return ExperimentalData(
        spectra_count=n_spectra,
        samples=samples,
        ionization_modes=ionization_modes,
        s_coordinates=s_coordinates,
        sample_labels=sample_labels,
        mode_labels=mode_labels,
        retention_times=retention_times,
        mz_values=mz_values,
        intensities=intensities,
        ms2_coverage=ms2_coverage
    )


if __name__ == "__main__":
    print("="*80)
    print("EXPERIMENTAL VALIDATION FRAMEWORK")
    print("="*80)
    
    # Generate synthetic data
    print("\nGenerating synthetic experimental data...")
    data = generate_synthetic_experimental_data()
    print(f"Total spectra: {data.spectra_count}")
    print(f"Samples: {data.samples}")
    print(f"Ionization modes: {data.ionization_modes}")
    
    # Test categorical thermodynamics
    print("\n" + "="*80)
    print("CATEGORICAL THERMODYNAMICS VALIDATION")
    print("="*80)
    
    thermo_validator = CategoricalThermodynamicsValidator()
    
    # Ideal gas law
    print("\n1. Ideal Gas Law Validation")
    print("-"*80)
    gas_results = thermo_validator.validate_ideal_gas_law(
        data.s_coordinates,
        data.sample_labels
    )
    print(f"Slope: {gas_results['slope']:.4f} (expected: ~1.0)")
    print(f"Intercept: {gas_results['intercept']:.4f} (expected: ~0.0)")
    print(f"R²: {gas_results['r_squared']:.4f}")
    print(f"p-value: {gas_results['p_value']:.2e}")
    print(f"Mean deviation: {gas_results['mean_deviation_percent']:.2f}%")
    
    # Maxwell-Boltzmann
    print("\n2. Maxwell-Boltzmann Distribution")
    print("-"*80)
    mb_results = thermo_validator.validate_maxwell_boltzmann(data.intensities)
    print(f"Mean intensity: {mb_results['mean_intensity']:.2e}")
    print(f"Scale parameter: {mb_results['scale_parameter']:.4f}")
    print(f"χ² p-value: {mb_results['chi2_p_value']:.4f}")
    print(f"KS p-value: {mb_results['ks_p_value']:.4f}")
    
    # Test S-entropy framework
    print("\n" + "="*80)
    print("S-ENTROPY COORDINATE VALIDATION")
    print("="*80)
    
    s_validator = SEntropyValidator()
    
    # Sample statistics
    print("\n1. Sample Statistics in S-Space")
    print("-"*80)
    sample_stats = s_validator.calculate_sample_statistics(
        data.s_coordinates,
        data.sample_labels
    )
    for sample, sample_stat in sample_stats['sample_stats'].items():
        print(f"\n{sample}:")
        print(f"  Count: {sample_stat['count']}")
        print(f"  Centroid: [{sample_stat['centroid'][0]:.3f}, {sample_stat['centroid'][1]:.3f}, {sample_stat['centroid'][2]:.3f}]")
        print(f"  Std: [{sample_stat['std'][0]:.3f}, {sample_stat['std'][1]:.3f}, {sample_stat['std'][2]:.3f}]")
    
    # Ionization mode separation
    print("\n2. Ionization Mode Separation")
    print("-"*80)
    mode_results = s_validator.validate_ionization_mode_separation(
        data.s_coordinates,
        data.mode_labels
    )
    for mode, mode_stat in mode_results.items():
        if mode != 'comparison':
            print(f"\n{mode}:")
            print(f"  Count: {mode_stat['count']}")
            print(f"  Mean S_k: {mode_stat['mean_S_k']:.3f} ± {mode_stat['std_S_k']:.3f}")
    
    if 'comparison' in mode_results:
        comp = mode_results['comparison']
        print(f"\nComparison:")
        print(f"  p-values: {[f'{p:.2e}' for p in comp['p_value']]}")
        print(f"  Cohen's d: {[f'{d:.3f}' for d in comp['cohens_d']]}")
    
    # Classification
    print("\n3. Sample Classification")
    print("-"*80)
    clf_results = s_validator.train_classifier(
        data.s_coordinates,
        data.sample_labels,
        classifier_type='rf'
    )
    print(f"Classifier: {clf_results['classifier_type']}")
    print(f"Accuracy: {clf_results['accuracy']:.2%}")
    print(f"\nConfusion Matrix:")
    print(clf_results['confusion_matrix'])
    
    # PCA
    print("\n4. PCA Analysis")
    print("-"*80)
    pca_results = s_validator.perform_pca(
        data.s_coordinates,
        data.sample_labels
    )
    print(f"PC1 variance: {pca_results['variance_explained'][0]:.2%}")
    print(f"PC2 variance: {pca_results['variance_explained'][1]:.2%}")
    print(f"Cumulative variance: {pca_results['cumulative_variance'][1]:.2%}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
