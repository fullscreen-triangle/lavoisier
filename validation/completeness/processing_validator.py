"""
Processing Validator Module

Comprehensive validation of data processing steps and pipeline integrity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Container for processing validation results"""
    step_name: str
    success_rate: float
    error_count: int
    interpretation: str
    metadata: Dict[str, Any] = None


class ProcessingValidator:
    """
    Comprehensive processing validation
    """
    
    def __init__(self):
        """Initialize processing validator"""
        self.results = []
        
    def validate_processing_step(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        step_name: str
    ) -> ProcessingResult:
        """
        Validate a processing step
        
        Args:
            input_data: Input data
            output_data: Output data
            step_name: Name of processing step
            
        Returns:
            ProcessingResult object
        """
        # Check for processing errors (NaN, inf, etc.)
        input_valid = np.isfinite(input_data).all()
        output_valid = np.isfinite(output_data).all()
        
        # Count errors
        error_count = 0
        if not input_valid:
            error_count += np.sum(~np.isfinite(input_data))
        if not output_valid:
            error_count += np.sum(~np.isfinite(output_data))
        
        # Calculate success rate
        total_elements = input_data.size + output_data.size
        success_rate = 1.0 - (error_count / total_elements)
        
        interpretation = f"{step_name}: {success_rate:.1%} success rate"
        if error_count > 0:
            interpretation += f" ({error_count} errors detected)"
        
        result = ProcessingResult(
            step_name=step_name,
            success_rate=success_rate,
            error_count=error_count,
            interpretation=interpretation,
            metadata={
                'input_shape': input_data.shape,
                'output_shape': output_data.shape,
                'input_valid': input_valid,
                'output_valid': output_valid
            }
        )
        
        self.results.append(result)
        return result
    
    def validate_pipeline_integrity(
        self,
        processing_steps: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> ProcessingResult:
        """
        Validate overall pipeline integrity
        
        Args:
            processing_steps: List of (step_name, input_data, output_data) tuples
            
        Returns:
            ProcessingResult object
        """
        total_errors = 0
        total_elements = 0
        
        for step_name, input_data, output_data in processing_steps:
            step_result = self.validate_processing_step(input_data, output_data, step_name)
            total_errors += step_result.error_count
            total_elements += input_data.size + output_data.size
        
        overall_success_rate = 1.0 - (total_errors / total_elements) if total_elements > 0 else 1.0
        
        interpretation = f"Pipeline integrity: {overall_success_rate:.1%} overall success rate"
        
        result = ProcessingResult(
            step_name="Pipeline Integrity",
            success_rate=overall_success_rate,
            error_count=total_errors,
            interpretation=interpretation,
            metadata={
                'total_steps': len(processing_steps),
                'total_elements': total_elements
            }
        )
        
        self.results.append(result)
        return result
    
    def generate_processing_report(self) -> pd.DataFrame:
        """Generate processing validation report"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'Step': result.step_name,
                'Success Rate': result.success_rate,
                'Errors': result.error_count,
                'Interpretation': result.interpretation
            })
        
        return pd.DataFrame(data)
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self.results = [] 