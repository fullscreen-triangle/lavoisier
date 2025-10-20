"""
Analysis Component Base
=======================

Base classes and interfaces for creating composable, surgically injectable
analysis components that can be integrated into any pipeline.

Key Concepts:
- AnalysisComponent: Base class for all analysis modules
- AnalysisResult: Standardized result container
- AnalysisPipeline: Composer for chaining components
- ComponentRegistry: Discovery and injection system
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path


class AnalysisCategory(Enum):
    """Categories of analysis components"""
    ANNOTATION = "annotation"
    COMPLETENESS = "completeness"
    FEATURES = "features"
    QUALITY = "quality"
    STATISTICAL = "statistical"
    CUSTOM = "custom"


class ComponentStatus(Enum):
    """Execution status of a component"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AnalysisResult:
    """
    Standardized result container for all analysis components.

    Provides uniform interface for accessing results regardless of component type.
    """
    component_name: str
    category: AnalysisCategory
    status: ComponentStatus
    execution_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    interpretations: Dict[str, str] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'component_name': self.component_name,
            'category': self.category.value,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'metrics': self.metrics,
            'interpretations': self.interpretations,
            'metadata': self.metadata,
            'error_message': self.error_message,
            'visualizations_count': len(self.visualizations)
        }

    def to_json(self) -> str:
        """Convert result to JSON (excluding visualizations)"""
        return json.dumps(self.to_dict(), indent=2)

    def passed_threshold(self, metric: str, threshold: float, mode: str = 'greater') -> bool:
        """Check if metric passed threshold"""
        if metric not in self.metrics:
            return False

        value = self.metrics[metric]
        if mode == 'greater':
            return value >= threshold
        elif mode == 'less':
            return value <= threshold
        elif mode == 'equal':
            return abs(value - threshold) < 1e-6
        return False


class AnalysisComponent(ABC):
    """
    Base class for all analysis components.

    Provides standard interface for execution, configuration, and result handling.
    Components are designed to be surgically injected into any pipeline.
    """

    def __init__(
        self,
        name: str,
        category: AnalysisCategory,
        description: str = "",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize analysis component.

        Args:
            name: Component name
            category: Analysis category
            description: Component description
            config: Component-specific configuration
        """
        self.name = name
        self.category = category
        self.description = description
        self.config = config or {}
        self.result: Optional[AnalysisResult] = None
        self._execution_time = 0.0

    @abstractmethod
    def execute(self, data: Any, **kwargs) -> AnalysisResult:
        """
        Execute analysis on input data.

        Args:
            data: Input data (format depends on component)
            **kwargs: Additional parameters

        Returns:
            AnalysisResult object
        """
        pass

    def configure(self, config: Dict[str, Any]):
        """Update component configuration"""
        self.config.update(config)

    def get_result(self) -> Optional[AnalysisResult]:
        """Get last execution result"""
        return self.result

    def reset(self):
        """Reset component state"""
        self.result = None
        self._execution_time = 0.0

    def _create_result(
        self,
        status: ComponentStatus,
        metrics: Dict[str, float],
        interpretations: Dict[str, str],
        visualizations: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> AnalysisResult:
        """Create standardized result object"""
        return AnalysisResult(
            component_name=self.name,
            category=self.category,
            status=status,
            execution_time=self._execution_time,
            metrics=metrics,
            interpretations=interpretations,
            visualizations=visualizations or {},
            metadata=metadata or {},
            error_message=error_message
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', category={self.category.value})>"


class AnalysisPipeline:
    """
    Composer for chaining multiple analysis components.

    Enables surgical injection of components into pipelines with:
    - Sequential execution
    - Parallel execution
    - Conditional execution
    - Result aggregation
    """

    def __init__(self, name: str = "AnalysisPipeline"):
        """
        Initialize analysis pipeline.

        Args:
            name: Pipeline name
        """
        self.name = name
        self.components: List[AnalysisComponent] = []
        self.results: List[AnalysisResult] = []
        self.execution_order: List[str] = []

    def add_component(self, component: AnalysisComponent):
        """Add component to pipeline"""
        self.components.append(component)
        return self

    def add_components(self, components: List[AnalysisComponent]):
        """Add multiple components to pipeline"""
        self.components.extend(components)
        return self

    def inject_at(self, index: int, component: AnalysisComponent):
        """Surgically inject component at specific position"""
        self.components.insert(index, component)
        return self

    def inject_before(self, target_name: str, component: AnalysisComponent):
        """Inject component before target component"""
        for i, comp in enumerate(self.components):
            if comp.name == target_name:
                self.components.insert(i, component)
                return self
        raise ValueError(f"Component '{target_name}' not found")

    def inject_after(self, target_name: str, component: AnalysisComponent):
        """Inject component after target component"""
        for i, comp in enumerate(self.components):
            if comp.name == target_name:
                self.components.insert(i + 1, component)
                return self
        raise ValueError(f"Component '{target_name}' not found")

    def remove_component(self, name: str):
        """Remove component by name"""
        self.components = [c for c in self.components if c.name != name]
        return self

    def execute_sequential(
        self,
        data: Any,
        pass_results: bool = False,
        stop_on_failure: bool = False,
        **kwargs
    ) -> List[AnalysisResult]:
        """
        Execute components sequentially.

        Args:
            data: Input data
            pass_results: Pass previous results to next component
            stop_on_failure: Stop execution if component fails
            **kwargs: Additional parameters for components

        Returns:
            List of AnalysisResult objects
        """
        self.results = []
        self.execution_order = []

        current_data = data

        for component in self.components:
            self.execution_order.append(component.name)

            try:
                # Pass previous result if requested
                if pass_results and self.results:
                    kwargs['previous_result'] = self.results[-1]

                result = component.execute(current_data, **kwargs)
                self.results.append(result)

                # Update data for next component if passing results
                if pass_results:
                    current_data = result

                # Stop if component failed and stop_on_failure is True
                if result.status == ComponentStatus.FAILED and stop_on_failure:
                    break

            except Exception as e:
                error_result = component._create_result(
                    status=ComponentStatus.FAILED,
                    metrics={},
                    interpretations={},
                    error_message=str(e)
                )
                self.results.append(error_result)

                if stop_on_failure:
                    break

        return self.results

    def execute_parallel(
        self,
        data: Any,
        **kwargs
    ) -> List[AnalysisResult]:
        """
        Execute components in parallel (simplified - single threaded for now).

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            List of AnalysisResult objects
        """
        # For now, execute sequentially (can be enhanced with multiprocessing)
        return self.execute_sequential(data, pass_results=False, **kwargs)

    def execute_conditional(
        self,
        data: Any,
        condition: Callable[[AnalysisResult], bool],
        **kwargs
    ) -> List[AnalysisResult]:
        """
        Execute components conditionally based on previous results.

        Args:
            data: Input data
            condition: Function that takes AnalysisResult and returns bool
            **kwargs: Additional parameters

        Returns:
            List of AnalysisResult objects
        """
        self.results = []
        self.execution_order = []

        for component in self.components:
            # Check condition if there are previous results
            if self.results and not condition(self.results[-1]):
                # Skip this component
                skip_result = component._create_result(
                    status=ComponentStatus.SKIPPED,
                    metrics={},
                    interpretations={'reason': 'Condition not met'}
                )
                self.results.append(skip_result)
                continue

            self.execution_order.append(component.name)

            try:
                result = component.execute(data, **kwargs)
                self.results.append(result)
            except Exception as e:
                error_result = component._create_result(
                    status=ComponentStatus.FAILED,
                    metrics={},
                    interpretations={},
                    error_message=str(e)
                )
                self.results.append(error_result)

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        total_time = sum(r.execution_time for r in self.results)
        status_counts = {}
        for status in ComponentStatus:
            status_counts[status.value] = sum(
                1 for r in self.results if r.status == status
            )

        return {
            'pipeline_name': self.name,
            'total_components': len(self.components),
            'executed_components': len(self.results),
            'execution_order': self.execution_order,
            'total_execution_time': total_time,
            'status_counts': status_counts,
            'all_metrics': {
                r.component_name: r.metrics for r in self.results
            }
        }

    def export_results(self, output_path: Union[str, Path]):
        """Export all results to JSON"""
        output_path = Path(output_path)

        export_data = {
            'pipeline_summary': self.get_summary(),
            'component_results': [r.to_dict() for r in self.results]
        }

        with open(output_path, 'w') as f:
            json.dumps(export_data, f, indent=2)

    def __repr__(self) -> str:
        return f"<AnalysisPipeline(name='{self.name}', components={len(self.components)})>"


class ComponentRegistry:
    """
    Registry for discovering and dynamically loading analysis components.

    Enables surgical injection by name without explicit imports.
    """

    def __init__(self):
        """Initialize component registry"""
        self._registry: Dict[str, Type[AnalysisComponent]] = {}
        self._instances: Dict[str, AnalysisComponent] = {}

    def register(self, name: str, component_class: Type[AnalysisComponent]):
        """Register a component class"""
        self._registry[name] = component_class

    def register_instance(self, name: str, instance: AnalysisComponent):
        """Register a component instance"""
        self._instances[name] = instance

    def get_class(self, name: str) -> Optional[Type[AnalysisComponent]]:
        """Get component class by name"""
        return self._registry.get(name)

    def get_instance(self, name: str) -> Optional[AnalysisComponent]:
        """Get component instance by name"""
        return self._instances.get(name)

    def create_instance(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[AnalysisComponent]:
        """Create new instance of registered component"""
        component_class = self._registry.get(name)
        if component_class:
            return component_class(config=config)
        return None

    def list_components(self, category: Optional[AnalysisCategory] = None) -> List[str]:
        """List all registered components, optionally filtered by category"""
        if category is None:
            return list(self._registry.keys())

        # Filter by category (requires instantiation to check)
        filtered = []
        for name, cls in self._registry.items():
            # Try to infer category from name or instantiate to check
            if category.value in name.lower():
                filtered.append(name)
        return filtered

    def inject_into_pipeline(
        self,
        pipeline: AnalysisPipeline,
        component_names: List[str],
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> AnalysisPipeline:
        """
        Surgically inject multiple components into pipeline.

        Args:
            pipeline: Target pipeline
            component_names: List of component names to inject
            configs: Optional configurations for components

        Returns:
            Modified pipeline
        """
        configs = configs or {}

        for name in component_names:
            config = configs.get(name, {})
            instance = self.create_instance(name, config)
            if instance:
                pipeline.add_component(instance)

        return pipeline

    def __repr__(self) -> str:
        return f"<ComponentRegistry(registered={len(self._registry)}, instances={len(self._instances)})>"


# Global registry instance
global_registry = ComponentRegistry()


def register_component(name: str):
    """Decorator for registering components"""
    def decorator(cls):
        global_registry.register(name, cls)
        return cls
    return decorator
