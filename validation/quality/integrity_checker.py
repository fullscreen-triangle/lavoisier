"""
Integrity Checker Module

Comprehensive data integrity validation including consistency checks,
format validation, and corruption detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import hashlib
import json


@dataclass
class IntegrityResult:
    """Container for integrity check results"""
    check_name: str
    passed: bool
    score: float  # 0-1 score
    issues_found: List[str]
    metadata: Dict[str, Any] = None


class IntegrityChecker:
    """
    Comprehensive data integrity validation
    """
    
    def __init__(self):
        """Initialize integrity checker"""
        self.results = []
        
    def check_data_consistency(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        expected_shape: Optional[Tuple] = None,
        expected_dtype: Optional[type] = None
    ) -> IntegrityResult:
        """
        Check data consistency including shape, type, and basic properties
        
        Args:
            data: Data to check
            expected_shape: Expected data shape
            expected_dtype: Expected data type
            
        Returns:
            IntegrityResult object
        """
        issues = []
        score = 1.0
        
        # Shape consistency
        if expected_shape is not None:
            if hasattr(data, 'shape'):
                if data.shape != expected_shape:
                    issues.append(f"Shape mismatch: expected {expected_shape}, got {data.shape}")
                    score -= 0.3
            else:
                issues.append("Data does not have shape attribute")
                score -= 0.2
        
        # Type consistency
        if expected_dtype is not None:
            if hasattr(data, 'dtype'):
                if data.dtype != expected_dtype:
                    issues.append(f"Type mismatch: expected {expected_dtype}, got {data.dtype}")
                    score -= 0.2
            else:
                issues.append("Data does not have dtype attribute")
                score -= 0.1
        
        # Check for NaN/infinite values
        if isinstance(data, np.ndarray):
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            
            if nan_count > 0:
                issues.append(f"Found {nan_count} NaN values")
                score -= min(0.3, nan_count / data.size)
            
            if inf_count > 0:
                issues.append(f"Found {inf_count} infinite values")
                score -= min(0.2, inf_count / data.size)
        
        elif isinstance(data, pd.DataFrame):
            nan_count = data.isnull().sum().sum()
            if nan_count > 0:
                issues.append(f"Found {nan_count} missing values")
                score -= min(0.3, nan_count / data.size)
        
        score = max(0.0, score)
        passed = len(issues) == 0
        
        result = IntegrityResult(
            check_name="Data Consistency",
            passed=passed,
            score=score,
            issues_found=issues,
            metadata={
                'data_shape': getattr(data, 'shape', None),
                'data_type': type(data).__name__,
                'data_dtype': getattr(data, 'dtype', None)
            }
        )
        
        self.results.append(result)
        return result
    
    def check_value_ranges(
        self,
        data: np.ndarray,
        expected_min: Optional[float] = None,
        expected_max: Optional[float] = None,
        tolerance: float = 0.01
    ) -> IntegrityResult:
        """
        Check if data values are within expected ranges
        
        Args:
            data: Data to check
            expected_min: Expected minimum value
            expected_max: Expected maximum value
            tolerance: Tolerance for range violations (fraction of data)
            
        Returns:
            IntegrityResult object
        """
        issues = []
        score = 1.0
        
        data_flat = data.flatten()
        data_min = np.min(data_flat)
        data_max = np.max(data_flat)
        
        # Check minimum value
        if expected_min is not None:
            violations = np.sum(data_flat < expected_min)
            violation_rate = violations / len(data_flat)
            
            if violation_rate > tolerance:
                issues.append(f"Minimum value violations: {violations} values below {expected_min}")
                score -= min(0.4, violation_rate)
        
        # Check maximum value
        if expected_max is not None:
            violations = np.sum(data_flat > expected_max)
            violation_rate = violations / len(data_flat)
            
            if violation_rate > tolerance:
                issues.append(f"Maximum value violations: {violations} values above {expected_max}")
                score -= min(0.4, violation_rate)
        
        # Check for extreme outliers (beyond 5 standard deviations)
        mean_val = np.mean(data_flat)
        std_val = np.std(data_flat)
        
        if std_val > 0:
            outliers = np.sum(np.abs(data_flat - mean_val) > 5 * std_val)
            outlier_rate = outliers / len(data_flat)
            
            if outlier_rate > 0.01:  # More than 1% outliers
                issues.append(f"Extreme outliers detected: {outliers} values beyond 5σ")
                score -= min(0.2, outlier_rate)
        
        score = max(0.0, score)
        passed = len(issues) == 0
        
        result = IntegrityResult(
            check_name="Value Range Check",
            passed=passed,
            score=score,
            issues_found=issues,
            metadata={
                'data_min': data_min,
                'data_max': data_max,
                'data_mean': np.mean(data_flat),
                'data_std': np.std(data_flat)
            }
        )
        
        self.results.append(result)
        return result
    
    def check_data_corruption(
        self,
        data: np.ndarray,
        reference_checksum: Optional[str] = None
    ) -> IntegrityResult:
        """
        Check for data corruption using checksums and pattern analysis
        
        Args:
            data: Data to check
            reference_checksum: Reference checksum for comparison
            
        Returns:
            IntegrityResult object
        """
        issues = []
        score = 1.0
        
        # Calculate current checksum
        data_bytes = data.tobytes()
        current_checksum = hashlib.md5(data_bytes).hexdigest()
        
        # Compare with reference if provided
        if reference_checksum is not None:
            if current_checksum != reference_checksum:
                issues.append(f"Checksum mismatch: expected {reference_checksum}, got {current_checksum}")
                score -= 0.5
        
        # Check for suspicious patterns that might indicate corruption
        
        # 1. Check for repeated values (potential corruption)
        data_flat = data.flatten()
        unique_values = len(np.unique(data_flat))
        total_values = len(data_flat)
        
        if unique_values < total_values * 0.1:  # Less than 10% unique values
            issues.append(f"Suspicious repetition: only {unique_values} unique values out of {total_values}")
            score -= 0.3
        
        # 2. Check for sudden jumps in sequential data
        if data.ndim == 1 or (data.ndim == 2 and min(data.shape) == 1):
            data_1d = data_flat
            diffs = np.diff(data_1d)
            mean_diff = np.mean(np.abs(diffs))
            std_diff = np.std(diffs)
            
            if std_diff > 0:
                large_jumps = np.sum(np.abs(diffs) > mean_diff + 5 * std_diff)
                jump_rate = large_jumps / len(diffs)
                
                if jump_rate > 0.05:  # More than 5% large jumps
                    issues.append(f"Suspicious data jumps: {large_jumps} large discontinuities")
                    score -= min(0.2, jump_rate)
        
        # 3. Check for zero-padding (potential truncation)
        zero_count = np.sum(data_flat == 0)
        zero_rate = zero_count / len(data_flat)
        
        if zero_rate > 0.5:  # More than 50% zeros
            issues.append(f"Excessive zeros: {zero_count} zero values ({zero_rate:.1%})")
            score -= min(0.2, zero_rate - 0.5)
        
        score = max(0.0, score)
        passed = len(issues) == 0
        
        result = IntegrityResult(
            check_name="Data Corruption Check",
            passed=passed,
            score=score,
            issues_found=issues,
            metadata={
                'checksum': current_checksum,
                'unique_values': unique_values,
                'total_values': total_values,
                'zero_rate': zero_rate
            }
        )
        
        self.results.append(result)
        return result
    
    def check_format_compliance(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        format_spec: Dict[str, Any]
    ) -> IntegrityResult:
        """
        Check compliance with specified data format
        
        Args:
            data: Data to check
            format_spec: Format specification dictionary
            
        Returns:
            IntegrityResult object
        """
        issues = []
        score = 1.0
        
        # Check required fields for DataFrame
        if isinstance(data, pd.DataFrame):
            required_columns = format_spec.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
                score -= 0.4
            
            # Check column types
            column_types = format_spec.get('column_types', {})
            for col, expected_type in column_types.items():
                if col in data.columns:
                    if not data[col].dtype == expected_type:
                        issues.append(f"Column {col} type mismatch: expected {expected_type}, got {data[col].dtype}")
                        score -= 0.1
        
        # Check array properties
        elif isinstance(data, np.ndarray):
            # Check dimensions
            expected_ndim = format_spec.get('ndim')
            if expected_ndim is not None and data.ndim != expected_ndim:
                issues.append(f"Dimension mismatch: expected {expected_ndim}D, got {data.ndim}D")
                score -= 0.3
            
            # Check data type
            expected_dtype = format_spec.get('dtype')
            if expected_dtype is not None and data.dtype != expected_dtype:
                issues.append(f"Data type mismatch: expected {expected_dtype}, got {data.dtype}")
                score -= 0.2
        
        # Check size constraints
        min_size = format_spec.get('min_size')
        max_size = format_spec.get('max_size')
        
        data_size = data.size if hasattr(data, 'size') else len(data)
        
        if min_size is not None and data_size < min_size:
            issues.append(f"Data too small: {data_size} < {min_size}")
            score -= 0.3
        
        if max_size is not None and data_size > max_size:
            issues.append(f"Data too large: {data_size} > {max_size}")
            score -= 0.2
        
        score = max(0.0, score)
        passed = len(issues) == 0
        
        result = IntegrityResult(
            check_name="Format Compliance",
            passed=passed,
            score=score,
            issues_found=issues,
            metadata={
                'format_spec': format_spec,
                'data_size': data_size
            }
        )
        
        self.results.append(result)
        return result
    
    def check_temporal_integrity(
        self,
        timestamps: np.ndarray,
        data: np.ndarray,
        expected_interval: Optional[float] = None
    ) -> IntegrityResult:
        """
        Check temporal integrity for time-series data
        
        Args:
            timestamps: Timestamp array
            data: Associated data array
            expected_interval: Expected time interval between samples
            
        Returns:
            IntegrityResult object
        """
        issues = []
        score = 1.0
        
        # Check timestamp ordering
        if not np.all(timestamps[:-1] <= timestamps[1:]):
            issues.append("Timestamps are not monotonically increasing")
            score -= 0.4
        
        # Check for duplicate timestamps
        unique_timestamps = len(np.unique(timestamps))
        if unique_timestamps < len(timestamps):
            duplicates = len(timestamps) - unique_timestamps
            issues.append(f"Found {duplicates} duplicate timestamps")
            score -= min(0.3, duplicates / len(timestamps))
        
        # Check time intervals
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            
            # Check for negative intervals
            negative_intervals = np.sum(intervals < 0)
            if negative_intervals > 0:
                issues.append(f"Found {negative_intervals} negative time intervals")
                score -= 0.3
            
            # Check interval consistency
            if expected_interval is not None:
                interval_deviations = np.abs(intervals - expected_interval)
                large_deviations = np.sum(interval_deviations > expected_interval * 0.1)
                
                if large_deviations > len(intervals) * 0.05:  # More than 5% deviations
                    issues.append(f"Inconsistent time intervals: {large_deviations} large deviations")
                    score -= 0.2
        
        # Check for data gaps
        if len(timestamps) > 2:
            median_interval = np.median(np.diff(timestamps))
            large_gaps = np.sum(np.diff(timestamps) > median_interval * 3)
            
            if large_gaps > 0:
                issues.append(f"Found {large_gaps} potential data gaps")
                score -= min(0.2, large_gaps / len(timestamps))
        
        score = max(0.0, score)
        passed = len(issues) == 0
        
        result = IntegrityResult(
            check_name="Temporal Integrity",
            passed=passed,
            score=score,
            issues_found=issues,
            metadata={
                'timestamp_range': (np.min(timestamps), np.max(timestamps)),
                'num_timestamps': len(timestamps),
                'unique_timestamps': unique_timestamps
            }
        )
        
        self.results.append(result)
        return result
    
    def comprehensive_integrity_check(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> List[IntegrityResult]:
        """
        Run comprehensive integrity checks
        
        Args:
            data: Data to check
            **kwargs: Additional parameters for specific checks
            
        Returns:
            List of IntegrityResult objects
        """
        results = []
        
        # Basic consistency check
        results.append(self.check_data_consistency(
            data,
            expected_shape=kwargs.get('expected_shape'),
            expected_dtype=kwargs.get('expected_dtype')
        ))
        
        # Value range check for numeric data
        if isinstance(data, np.ndarray):
            results.append(self.check_value_ranges(
                data,
                expected_min=kwargs.get('expected_min'),
                expected_max=kwargs.get('expected_max')
            ))
            
            # Corruption check
            results.append(self.check_data_corruption(
                data,
                reference_checksum=kwargs.get('reference_checksum')
            ))
        
        # Format compliance check
        if 'format_spec' in kwargs:
            results.append(self.check_format_compliance(
                data,
                kwargs['format_spec']
            ))
        
        # Temporal integrity check
        if 'timestamps' in kwargs:
            results.append(self.check_temporal_integrity(
                kwargs['timestamps'],
                data,
                expected_interval=kwargs.get('expected_interval')
            ))
        
        return results
    
    def generate_integrity_report(self) -> pd.DataFrame:
        """Generate comprehensive integrity report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Check': result.check_name,
                'Passed': result.passed,
                'Score': result.score,
                'Issues': len(result.issues_found),
                'Details': '; '.join(result.issues_found) if result.issues_found else 'No issues'
            })
        
        return pd.DataFrame(data)
    
    def get_overall_integrity_score(self) -> float:
        """Calculate overall integrity score"""
        if not self.results:
            return 0.0
        
        return np.mean([r.score for r in self.results])
    
    def get_failed_checks(self) -> List[IntegrityResult]:
        """Get list of failed integrity checks"""
        return [r for r in self.results if not r.passed]
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 