"""
Missing Data Detector Module

Comprehensive detection and analysis of missing data patterns
including systematic missingness, random missingness, and impact assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


@dataclass
class MissingDataResult:
    """Container for missing data analysis results"""
    analysis_type: str
    missing_percentage: float
    pattern_type: str  # 'MCAR', 'MAR', 'MNAR', 'Unknown'
    severity: str  # 'Low', 'Moderate', 'High', 'Critical'
    recommendations: List[str]
    metadata: Dict[str, Any] = None


class MissingDataDetector:
    """
    Comprehensive missing data detection and pattern analysis
    """
    
    def __init__(self):
        """Initialize missing data detector"""
        self.results = []
        
    def detect_missing_patterns(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> MissingDataResult:
        """
        Detect and analyze missing data patterns
        
        Args:
            data: Data to analyze for missing patterns
            
        Returns:
            MissingDataResult object
        """
        if isinstance(data, np.ndarray):
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Calculate missing data statistical_analysis
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        # Analyze missing patterns by column
        column_missing = df.isnull().sum()
        column_missing_pct = (column_missing / len(df)) * 100
        
        # Analyze missing patterns by row
        row_missing = df.isnull().sum(axis=1)
        row_missing_pct = (row_missing / len(df.columns)) * 100
        
        # Determine pattern type
        pattern_type = self._classify_missing_pattern(df)
        
        # Determine severity
        if missing_percentage < 5:
            severity = "Low"
        elif missing_percentage < 15:
            severity = "Moderate"
        elif missing_percentage < 30:
            severity = "High"
        else:
            severity = "Critical"
        
        # Generate recommendations
        recommendations = self._generate_missing_data_recommendations(
            missing_percentage, pattern_type, column_missing_pct, row_missing_pct
        )
        
        result = MissingDataResult(
            analysis_type="Missing Pattern Analysis",
            missing_percentage=missing_percentage,
            pattern_type=pattern_type,
            severity=severity,
            recommendations=recommendations,
            metadata={
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'column_missing_pct': column_missing_pct.to_dict() if hasattr(column_missing_pct, 'to_dict') else column_missing_pct,
                'row_missing_stats': {
                    'mean': row_missing_pct.mean(),
                    'std': row_missing_pct.std(),
                    'max': row_missing_pct.max()
                }
            }
        )
        
        self.results.append(result)
        return result
    
    def _classify_missing_pattern(self, df: pd.DataFrame) -> str:
        """
        Classify the type of missing data pattern
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Pattern type string
        """
        # Create missing indicator matrix
        missing_matrix = df.isnull()
        
        # Test for completely random missingness (Little's MCAR test approximation)
        if missing_matrix.sum().sum() == 0:
            return "No Missing Data"
        
        # Check for systematic patterns
        
        # 1. Check if missingness is concentrated in specific columns
        column_missing_pct = missing_matrix.sum() / len(df)
        if (column_missing_pct > 0.8).any():
            return "MNAR"  # Likely Missing Not At Random
        
        # 2. Check for patterns in missing combinations
        missing_patterns = missing_matrix.groupby(list(missing_matrix.columns)).size()
        
        # If there are very few distinct missing patterns, likely systematic
        unique_patterns = len(missing_patterns)
        total_rows = len(df)
        
        if unique_patterns < total_rows * 0.1:  # Less than 10% unique patterns
            # Check if patterns are related to other variables
            return "MAR"  # Likely Missing At Random (conditional)
        
        # 3. Statistical test for randomness
        try:
            # Chi-square test for independence of missingness patterns
            if len(df.columns) >= 2:
                col1_missing = missing_matrix.iloc[:, 0]
                col2_missing = missing_matrix.iloc[:, 1]
                
                contingency_table = pd.crosstab(col1_missing, col2_missing)
                if contingency_table.shape == (2, 2):
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    
                    if p_value < 0.05:  # Significant dependence
                        return "MAR"
                    else:
                        return "MCAR"  # Missing Completely At Random
        except:
            pass
        
        return "Unknown"
    
    def _generate_missing_data_recommendations(
        self,
        missing_percentage: float,
        pattern_type: str,
        column_missing_pct: pd.Series,
        row_missing_pct: pd.Series
    ) -> List[str]:
        """
        Generate recommendations based on missing data analysis
        
        Args:
            missing_percentage: Overall missing percentage
            pattern_type: Type of missing pattern
            column_missing_pct: Missing percentage by column
            row_missing_pct: Missing percentage by row
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # General recommendations based on severity
        if missing_percentage < 5:
            recommendations.append("Low missing data rate - proceed with standard analysis")
        elif missing_percentage < 15:
            recommendations.append("Moderate missing data - consider imputation methods")
        elif missing_percentage < 30:
            recommendations.append("High missing data - careful imputation or specialized methods required")
        else:
            recommendations.append("Critical missing data - consider data collection improvement")
        
        # Pattern-specific recommendations
        if pattern_type == "MCAR":
            recommendations.append("Missing Completely At Random - listwise deletion acceptable")
            recommendations.append("Simple imputation methods (mean, median) may be sufficient")
        elif pattern_type == "MAR":
            recommendations.append("Missing At Random - use advanced imputation (multiple imputation)")
            recommendations.append("Consider predictive models for imputation")
        elif pattern_type == "MNAR":
            recommendations.append("Missing Not At Random - investigate missingness mechanism")
            recommendations.append("Consider domain-specific imputation or modeling approaches")
        
        # Column-specific recommendations
        if hasattr(column_missing_pct, 'max') and column_missing_pct.max() > 50:
            recommendations.append("Some columns have >50% missing - consider removing these variables")
        
        # Row-specific recommendations
        if hasattr(row_missing_pct, 'max') and row_missing_pct.max() > 50:
            recommendations.append("Some samples have >50% missing - consider removing these samples")
        
        return recommendations
    
    def detect_systematic_missingness(
        self,
        data: pd.DataFrame,
        grouping_variable: Optional[str] = None
    ) -> MissingDataResult:
        """
        Detect systematic missingness patterns
        
        Args:
            data: DataFrame to analyze
            grouping_variable: Variable to group by for systematic analysis
            
        Returns:
            MissingDataResult object
        """
        missing_matrix = data.isnull()
        
        # Overall systematic patterns
        systematic_patterns = []
        
        # 1. Time-based patterns (if index is datetime)
        if hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
            # Check for seasonal or periodic missingness
            data_with_time = data.copy()
            data_with_time['missing_count'] = missing_matrix.sum(axis=1)
            
            # Group by time periods
            monthly_missing = data_with_time.groupby(data_with_time.index.month)['missing_count'].mean()
            if monthly_missing.std() > monthly_missing.mean() * 0.5:
                systematic_patterns.append("Seasonal missingness pattern detected")
        
        # 2. Group-based patterns
        if grouping_variable and grouping_variable in data.columns:
            group_missing = data.groupby(grouping_variable).apply(
                lambda x: x.isnull().sum().sum() / x.size * 100
            )
            
            if group_missing.std() > 10:  # High variation in missing rates between groups
                systematic_patterns.append(f"Systematic missingness varies by {grouping_variable}")
        
        # 3. Variable correlation patterns
        missing_corr = missing_matrix.corr()
        high_corr_pairs = []
        
        for i in range(len(missing_corr.columns)):
            for j in range(i+1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation in missingness
                    high_corr_pairs.append((missing_corr.columns[i], missing_corr.columns[j], corr_val))
        
        if high_corr_pairs:
            systematic_patterns.append(f"Found {len(high_corr_pairs)} variable pairs with correlated missingness")
        
        # Determine pattern type and severity
        if len(systematic_patterns) == 0:
            pattern_type = "Random"
            severity = "Low"
        elif len(systematic_patterns) <= 2:
            pattern_type = "Partially Systematic"
            severity = "Moderate"
        else:
            pattern_type = "Highly Systematic"
            severity = "High"
        
        missing_percentage = missing_matrix.sum().sum() / missing_matrix.size * 100
        
        recommendations = [
            f"Detected {len(systematic_patterns)} systematic patterns",
            "Investigate underlying causes of systematic missingness",
            "Consider pattern-aware imputation methods"
        ]
        
        result = MissingDataResult(
            analysis_type="Systematic Missingness",
            missing_percentage=missing_percentage,
            pattern_type=pattern_type,
            severity=severity,
            recommendations=recommendations,
            metadata={
                'systematic_patterns': systematic_patterns,
                'high_corr_pairs': high_corr_pairs,
                'missing_correlation_matrix': missing_corr.to_dict() if hasattr(missing_corr, 'to_dict') else None
            }
        )
        
        self.results.append(result)
        return result
    
    def assess_missing_data_impact(
        self,
        data: pd.DataFrame,
        target_variable: Optional[str] = None
    ) -> MissingDataResult:
        """
        Assess the impact of missing data on analysis
        
        Args:
            data: DataFrame to analyze
            target_variable: Target variable for impact assessment
            
        Returns:
            MissingDataResult object
        """
        missing_matrix = data.isnull()
        missing_percentage = missing_matrix.sum().sum() / missing_matrix.size * 100
        
        impact_metrics = {}
        
        # 1. Sample size impact
        complete_cases = data.dropna().shape[0]
        original_cases = data.shape[0]
        sample_loss_pct = (1 - complete_cases / original_cases) * 100
        
        impact_metrics['sample_loss_percentage'] = sample_loss_pct
        
        # 2. Variable impact
        variables_with_missing = missing_matrix.sum() > 0
        affected_variables_pct = variables_with_missing.sum() / len(data.columns) * 100
        
        impact_metrics['affected_variables_percentage'] = affected_variables_pct
        
        # 3. Target variable impact (if specified)
        if target_variable and target_variable in data.columns:
            target_missing_pct = missing_matrix[target_variable].sum() / len(data) * 100
            impact_metrics['target_missing_percentage'] = target_missing_pct
            
            # Correlation between target missingness and other variables
            target_missing_corr = missing_matrix.corrwith(missing_matrix[target_variable])
            high_target_corr = target_missing_corr[abs(target_missing_corr) > 0.3]
            impact_metrics['target_correlated_variables'] = len(high_target_corr) - 1  # Exclude self
        
        # 4. Analysis power impact
        if sample_loss_pct > 50:
            power_impact = "Critical"
        elif sample_loss_pct > 25:
            power_impact = "High"
        elif sample_loss_pct > 10:
            power_impact = "Moderate"
        else:
            power_impact = "Low"
        
        # Determine overall severity
        if sample_loss_pct > 50 or missing_percentage > 30:
            severity = "Critical"
        elif sample_loss_pct > 25 or missing_percentage > 15:
            severity = "High"
        elif sample_loss_pct > 10 or missing_percentage > 5:
            severity = "Moderate"
        else:
            severity = "Low"
        
        # Generate recommendations
        recommendations = [
            f"Sample size reduced by {sample_loss_pct:.1f}% with complete case analysis",
            f"Statistical power impact: {power_impact}",
        ]
        
        if sample_loss_pct > 25:
            recommendations.append("Consider imputation to preserve sample size")
        
        if target_variable and impact_metrics.get('target_missing_percentage', 0) > 10:
            recommendations.append("Target variable has significant missingness - investigate carefully")
        
        result = MissingDataResult(
            analysis_type="Missing Data Impact",
            missing_percentage=missing_percentage,
            pattern_type=f"Power Impact: {power_impact}",
            severity=severity,
            recommendations=recommendations,
            metadata=impact_metrics
        )
        
        self.results.append(result)
        return result
    
    def comprehensive_missing_analysis(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> List[MissingDataResult]:
        """
        Run comprehensive missing data analysis
        
        Args:
            data: Data to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of MissingDataResult objects
        """
        results = []
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Basic pattern detection
        results.append(self.detect_missing_patterns(df))
        
        # Systematic missingness detection
        grouping_var = kwargs.get('grouping_variable')
        results.append(self.detect_systematic_missingness(df, grouping_var))
        
        # Impact assessment
        target_var = kwargs.get('target_variable')
        results.append(self.assess_missing_data_impact(df, target_var))
        
        return results
    
    def generate_missing_data_report(self) -> pd.DataFrame:
        """Generate comprehensive missing data report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Analysis Type': result.analysis_type,
                'Missing %': result.missing_percentage,
                'Pattern': result.pattern_type,
                'Severity': result.severity,
                'Key Recommendations': '; '.join(result.recommendations[:2])  # First 2 recommendations
            })
        
        return pd.DataFrame(data)
    
    def get_critical_missing_issues(self) -> List[MissingDataResult]:
        """Get list of critical missing data issues"""
        return [r for r in self.results if r.severity in ['Critical', 'High']]
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 