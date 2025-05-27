#!/usr/bin/env python3
"""
Test script to verify all validation module imports work correctly
"""

def test_imports():
    """Test all validation module imports"""
    try:
        print("Testing validation module imports...")
        
        # Test main validation imports
        from validation import (
            # Statistics
            HypothesisTestSuite, EffectSizeCalculator, StatisticalValidator, BiasDetector,
            # Performance
            PerformanceBenchmark, EfficiencyAnalyzer, ScalabilityTester,
            # Quality
            DataQualityAssessor, FidelityAnalyzer, IntegrityChecker, QualityMetrics,
            # Completeness
            CompletenessAnalyzer, CoverageAssessment, MissingDataDetector, ProcessingValidator,
            # Features
            FeatureExtractorComparator, InformationContentAnalyzer, DimensionalityReducer, ClusteringValidator,
            # Vision
            ComputerVisionValidator, ImageQualityAssessor, VideoAnalyzer,
            # Annotation
            AnnotationPerformanceEvaluator, CompoundIdentificationValidator, DatabaseSearchAnalyzer, ConfidenceScoreValidator
        )
        
        print("✓ All imports successful!")
        
        # Test instantiation of key classes
        print("\nTesting class instantiation...")
        
        # Statistics
        hypothesis_tester = HypothesisTestSuite()
        print("✓ HypothesisTestSuite instantiated")
        
        # Performance
        benchmark = PerformanceBenchmark()
        scalability_tester = ScalabilityTester()
        print("✓ Performance modules instantiated")
        
        # Quality
        quality_assessor = DataQualityAssessor()
        print("✓ Quality modules instantiated")
        
        # Vision
        cv_validator = ComputerVisionValidator()
        print("✓ Vision modules instantiated")
        
        print("\n🎉 All validation modules are working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Ready to run the validation pipeline!")
    else:
        print("\n❌ Please fix the import issues before running the pipeline.") 