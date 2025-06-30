# Buhera Tutorials: From Beginner to Advanced

This tutorial series will guide you through learning Buhera, from basic concepts to advanced scientific applications.

## Table of Contents

- [Tutorial 1: Your First Buhera Script](#tutorial-1-your-first-buhera-script)
- [Tutorial 2: Biomarker Discovery](#tutorial-2-biomarker-discovery)
- [Tutorial 3: Drug Metabolism Study](#tutorial-3-drug-metabolism-study)
- [Tutorial 4: Advanced Validation](#tutorial-4-advanced-validation)
- [Tutorial 5: Custom Evidence Networks](#tutorial-5-custom-evidence-networks)
- [Tutorial 6: Performance Optimization](#tutorial-6-performance-optimization)

---

## Tutorial 1: Your First Buhera Script

### Learning Objectives
- Understand basic Buhera script structure
- Write a simple objective with validation
- Run validation and understand output

### Background
We'll create a simple metabolite identification script to learn the fundamentals.

### Step 1: Basic Script Structure

Create a file called `first_script.bh`:

```javascript
// first_script.bh - A simple metabolite identification script

// Import required Lavoisier modules
import lavoisier.mzekezeke
import lavoisier.hatata

// Define our scientific objective
objective MetaboliteIdentification:
    target: "identify known metabolites in plasma samples"
    success_criteria: "confidence >= 0.8 AND false_discovery_rate <= 0.05"
    evidence_priorities: "mass_match,ms2_fragmentation"
    biological_constraints: "plasma_metabolome"
    statistical_requirements: "sample_size >= 10"

// Add basic validation
validate DataQuality:
    check_instrument_capability
    if mass_accuracy > 10_ppm:
        warn("Mass accuracy may be insufficient for confident identification")

// Simple analysis phase
phase Identification:
    dataset = load_dataset(file_path: "plasma_samples.mzML")
    annotations = lavoisier.mzekezeke.build_evidence_network(
        data: dataset,
        objective: "metabolite_identification",
        evidence_types: ["mass_match", "ms2_fragmentation"]
    )
```

### Step 2: Validate Your Script

```bash
buhera validate first_script.bh
```

Expected output:
```
🔍 Validating Buhera script: first_script.bh
✅ Script parsed successfully
📋 Objective: MetaboliteIdentification
📊 Pre-flight validation: 2 checks passed, 0 warnings
✅ Validation PASSED - Script ready for execution
🎯 Estimated success probability: 78.5%
```

### Step 3: Understanding the Output

- **Script parsed successfully**: Syntax is correct
- **Objective**: Shows your defined objective
- **Pre-flight validation**: Number of checks performed
- **Success probability**: Estimated likelihood of achieving your objective

### Exercise 1.1
Modify the success criteria to require 90% confidence and re-validate. Notice how the success probability changes.

### Exercise 1.2
Add another validation rule for sample size:

```javascript
validate SampleSize:
    check_sample_size
    if sample_size < 10:
        abort("Insufficient samples for reliable identification")
```

---

## Tutorial 2: Biomarker Discovery

### Learning Objectives
- Design objectives for biomarker discovery
- Implement comprehensive validation
- Use pathway-focused evidence networks

### Background
Biomarker discovery requires specific validation and evidence weighting. We'll create a diabetes biomarker discovery script.

### Step 1: Define the Biomarker Objective

```javascript
// diabetes_biomarkers.bh - Comprehensive biomarker discovery

import lavoisier.mzekezeke
import lavoisier.hatata
import lavoisier.zengeza
import lavoisier.diggiden

objective DiabetesBiomarkerDiscovery:
    target: "identify metabolites predictive of diabetes progression"
    success_criteria: "sensitivity >= 0.85 AND specificity >= 0.85 AND auc >= 0.9"
    evidence_priorities: "pathway_membership,ms2_fragmentation,mass_match"
    biological_constraints: "glycolysis_upregulated,insulin_resistance_markers"
    statistical_requirements: "sample_size >= 30, effect_size >= 0.5, power >= 0.8"
```

**Key Points:**
- **AUC requirement**: Critical for biomarker performance
- **Pathway membership**: Prioritized for biological relevance
- **Larger sample size**: Required for biomarker validation

### Step 2: Comprehensive Validation

```javascript
validate InstrumentCapability:
    check_instrument_capability
    if target_concentration < instrument_detection_limit:
        abort("Orbitrap cannot detect expected biomarker concentrations")

validate StudyDesign:
    check_sample_size
    if sample_size < 30:
        abort("Insufficient sample size for biomarker discovery")
    if case_control_ratio < 0.5 OR case_control_ratio > 2.0:
        warn("Unbalanced case-control ratio may bias results")

validate BiologicalCoherence:
    check_pathway_consistency
    if glycolysis_markers absent AND diabetes_expected:
        warn("Missing expected glycolysis disruption markers")
    if insulin_signaling_intact AND insulin_resistance_expected:
        warn("Contradictory insulin signaling expectations")
```

### Step 3: Goal-Directed Analysis

```javascript
phase DataAcquisition:
    dataset = load_dataset(
        file_path: "diabetes_cohort.mzML",
        metadata: "clinical_diabetes_data.csv",
        groups: ["control", "prediabetic", "t2dm"],
        focus: "diabetes_progression_markers"
    )

phase QualityControl:
    quality_metrics = lavoisier.validate_data_quality(dataset)
    if quality_metrics.mass_accuracy > 5_ppm:
        abort("Mass accuracy insufficient for biomarker identification")

phase PreprocessingWithContext:
    // Preserve diabetes-relevant signals during noise reduction
    clean_data = lavoisier.zengeza.noise_reduction(
        data: dataset,
        objective_context: "diabetes_biomarker_discovery",
        preserve_patterns: ["glucose_metabolism", "lipid_metabolism", "amino_acids"]
    )

phase BiomarkerDiscovery:
    // Build evidence network optimized for biomarker discovery
    evidence_network = lavoisier.mzekezeke.build_evidence_network(
        data: clean_data,
        objective: "diabetes_biomarker_discovery",
        evidence_types: ["pathway_membership", "ms2_fragmentation", "mass_match"],
        pathway_focus: ["glycolysis", "gluconeogenesis", "lipid_metabolism"],
        evidence_weights: {
            "pathway_membership": 1.3,  // Higher weight for biological relevance
            "ms2_fragmentation": 1.1,   // Structural confirmation important
            "mass_match": 1.0           // Basic identification
        }
    )

phase BiomarkerValidation:
    annotations = lavoisier.hatata.validate_with_objective(
        evidence_network: evidence_network,
        objective: "diabetes_biomarker_discovery",
        confidence_threshold: 0.85,
        clinical_utility_threshold: 0.8
    )
    
    // Test robustness of biomarker candidates
    robustness_test = lavoisier.diggiden.test_biomarker_robustness(
        annotations: annotations,
        perturbation_types: ["noise_injection", "batch_effects"]
    )

phase ClinicalAssessment:
    if annotations.confidence > 0.85 AND robustness_test.stability > 0.8:
        biomarker_candidates = filter_top_candidates(annotations, top_n: 20)
        clinical_performance = assess_clinical_utility(
            candidates: biomarker_candidates,
            clinical_outcomes: "diabetes_progression"
        )
        
        if clinical_performance.auc >= 0.9:
            generate_biomarker_panel(biomarker_candidates)
        else:
            recommend_additional_validation(clinical_performance)
    else:
        abort("Insufficient evidence for reliable biomarker identification")
```

### Step 4: Validate and Execute

```bash
buhera validate diabetes_biomarkers.bh
```

### Exercise 2.1
Create a cancer biomarker discovery script by modifying the diabetes example. Consider:
- Different pathway focus (e.g., "cell_cycle", "apoptosis", "dna_repair")
- Cancer-specific biological constraints
- Appropriate success criteria for cancer biomarkers

### Exercise 2.2
Add a validation rule that checks for batch effects:

```javascript
validate BatchEffects:
    check_batch_consistency
    if significant_batch_effects:
        warn("Batch effects detected - consider batch correction")
```

---

## Tutorial 3: Drug Metabolism Study

### Learning Objectives
- Design experiments for drug metabolism characterization
- Validate extraction methods against objectives
- Implement time-course analysis

### Background
Drug metabolism studies require different evidence priorities and validation than biomarker discovery.

### Step 1: Drug Metabolism Objective

```javascript
// drug_metabolism.bh - Comprehensive drug metabolism characterization

import lavoisier.mzekezeke
import lavoisier.hatata
import lavoisier.zengeza

objective DrugMetabolismCharacterization:
    target: "characterize hepatic metabolism of compound_X"
    success_criteria: "metabolite_coverage >= 0.8 AND pathway_coherence >= 0.7"
    evidence_priorities: "ms2_fragmentation,mass_match,retention_time"
    biological_constraints: "cyp450_involvement,phase2_conjugation"
    statistical_requirements: "sample_size >= 15, technical_replicates >= 3"
```

**Key Differences from Biomarker Discovery:**
- **MS2 fragmentation prioritized**: Structure determination crucial
- **Retention time important**: Chromatographic behavior matters
- **Pathway coherence**: Metabolic transformations must make sense

### Step 2: Method-Specific Validation

```javascript
validate ExtractionMethod:
    if expecting_phase1_metabolites AND using_aqueous_extraction:
        warn("Aqueous extraction may miss lipophilic Phase I metabolites")
    
    if expecting_glucuronides AND using_organic_extraction:
        warn("Organic extraction may miss water-soluble conjugates")
    
    if expecting_sulfates AND ph > 7:
        warn("High pH may cause sulfate hydrolysis")

validate IncubationConditions:
    if temperature != 37_celsius:
        warn("Non-physiological temperature may affect metabolism")
    
    if incubation_time < 30_minutes:
        warn("Short incubation may miss slow metabolic processes")

validate CYP450Activity:
    check_enzyme_activity
    if cyp3a4_activity < 0.5:
        warn("Low CYP3A4 activity - major metabolic pathway may be impaired")
```

### Step 3: Time-Course Analysis

```javascript
phase TimeSeriesAnalysis:
    timepoints = ["0h", "0.5h", "1h", "2h", "4h", "8h", "24h"]
    
    for timepoint in timepoints:
        dataset = load_dataset(
            file_path: f"metabolism_{timepoint}.mzML",
            metadata: f"conditions_{timepoint}.csv"
        )
        
        time_network = lavoisier.mzekezeke.build_evidence_network(
            data: dataset,
            objective: "drug_metabolism_characterization",
            evidence_types: ["ms2_fragmentation", "mass_match"],
            pathway_focus: ["cyp450", "ugt", "sult", "gst"],
            time_context: timepoint
        )
        
        metabolite_identification = lavoisier.hatata.validate_with_objective(
            evidence_network: time_network,
            confidence_threshold: 0.7,
            structural_requirement: "ms2_confirmed"
        )

phase MetabolicPathwayMapping:
    // Reconstruct metabolic pathways from time-series data
    pathway_reconstruction = analyze_metabolic_progression(
        metabolite_timeseries: metabolite_identification,
        parent_compound: "compound_X",
        expected_pathways: ["oxidation", "conjugation", "hydrolysis"]
    )
    
    if pathway_reconstruction.coverage >= 0.8:
        generate_metabolic_map(pathway_reconstruction)
    else:
        suggest_additional_timepoints(pathway_reconstruction)
```

### Exercise 3.1
Create a pharmacokinetic study script that focuses on quantification rather than identification:

```javascript
objective PharmacokineticsStudy:
    target: "quantify compound_X and metabolites in plasma over time"
    success_criteria: "accuracy >= 0.9 AND precision <= 0.1"
    evidence_priorities: "isotope_pattern,retention_time,mass_match"
    // Note: Quantification priorities differ from identification
```

### Exercise 3.2
Add validation for analytical standards:

```javascript
validate AnalyticalStandards:
    if missing_parent_standard:
        abort("Parent compound standard required for quantification")
    
    if missing_metabolite_standards AND quantification_required:
        warn("Metabolite standards missing - semi-quantitative results only")
```

---

## Tutorial 4: Advanced Validation

### Learning Objectives
- Implement complex validation logic
- Create custom validation functions
- Handle conditional validation scenarios

### Background
Advanced scripts require sophisticated validation that considers multiple experimental factors.

### Step 1: Multi-Factor Validation

```javascript
validate ComplexStudyDesign:
    // Check statistical power with multiple factors
    check_sample_size
    effect_size = calculate_expected_effect_size()
    power = calculate_statistical_power(
        sample_size: sample_size,
        effect_size: effect_size,
        alpha: 0.05
    )
    
    if power < 0.8:
        required_n = calculate_required_sample_size(
            effect_size: effect_size,
            power: 0.8,
            alpha: 0.05
        )
        abort(f"Insufficient power ({power:.2f}). Need {required_n} samples.")

validate InstrumentCompatibility:
    // Multi-instrument validation
    if primary_instrument == "orbitrap" AND secondary_instrument == "qtof":
        if mass_accuracy_difference > 2_ppm:
            warn("Large mass accuracy difference between instruments")
    
    if using_multiple_columns:
        check_column_compatibility()
        if retention_time_shift > 0.5_minutes:
            warn("Significant retention time differences between columns")
```

### Step 2: Conditional Validation Chains

```javascript
validate ConditionalChecks:
    // Validation depends on previous conditions
    if study_type == "biomarker_discovery":
        validate_biomarker_requirements()
    else if study_type == "drug_metabolism":
        validate_metabolism_requirements()
    else if study_type == "quantification":
        validate_quantification_requirements()

function validate_biomarker_requirements():
    if sample_size < 30:
        abort("Biomarker discovery requires >= 30 samples")
    
    if missing_clinical_metadata:
        abort("Clinical metadata required for biomarker studies")
    
    check_pathway_diversity()

function validate_metabolism_requirements():
    if missing_enzyme_activity_data:
        warn("Enzyme activity data recommended for metabolism studies")
    
    if incubation_conditions_not_standardized:
        warn("Standardized incubation conditions recommended")

function validate_quantification_requirements():
    if missing_internal_standards:
        abort("Internal standards required for quantification")
    
    if missing_calibration_curve:
        abort("Calibration curve required for accurate quantification")
```

### Step 3: Dynamic Validation

```javascript
validate DynamicConditions:
    // Validation that adapts to data characteristics
    preliminary_scan = quick_data_assessment(dataset)
    
    if preliminary_scan.complexity == "high":
        // High complexity data needs more stringent validation
        minimum_confidence = 0.9
        require_ms2_confirmation = true
    else if preliminary_scan.complexity == "low":
        minimum_confidence = 0.7
        require_ms2_confirmation = false
    
    if preliminary_scan.noise_level > 0.3:
        recommend_additional_cleanup = true
        warn("High noise level detected - consider additional sample cleanup")

validate AdaptiveThresholds:
    // Thresholds that adapt to experimental conditions
    base_confidence = 0.8
    
    if instrument_performance == "high":
        confidence_adjustment = 0.0
    else if instrument_performance == "medium":
        confidence_adjustment = 0.1
    else:
        confidence_adjustment = 0.2
    
    adjusted_confidence = base_confidence + confidence_adjustment
    
    if adjusted_confidence > 0.95:
        warn("Confidence threshold may be too stringent for current conditions")
```

### Exercise 4.1
Create a validation rule that checks for seasonal effects in biological samples:

```javascript
validate SeasonalEffects:
    // Consider if sample collection season affects results
    if study_duration > 6_months AND not_controlling_for_season:
        warn("Long study duration - consider seasonal effects on metabolism")
```

### Exercise 4.2
Implement a validation that checks for appropriate controls:

```javascript
validate ControlSamples:
    if missing_blank_samples:
        abort("Blank samples required for contamination assessment")
    
    if missing_qc_samples:
        warn("QC samples recommended for data quality assessment")
    
    if biological_study AND missing_negative_controls:
        warn("Negative controls recommended for biological studies")
```

---

## Tutorial 5: Custom Evidence Networks

### Learning Objectives
- Design custom evidence weighting schemes
- Implement objective-specific evidence networks
- Optimize evidence integration

### Background
Different research objectives require different evidence weighting strategies.

### Step 1: Custom Evidence Weights

```javascript
// custom_evidence.bh - Advanced evidence network customization

phase CustomEvidenceNetworks:
    // Define objective-specific evidence weights
    if objective_type == "biomarker_discovery":
        evidence_weights = {
            "pathway_membership": 1.5,      // Biological relevance critical
            "clinical_correlation": 1.3,    // Clinical utility important
            "ms2_fragmentation": 1.1,       // Structural confirmation
            "mass_match": 1.0,              // Basic identification
            "retention_time": 0.8           // Less critical for biomarkers
        }
    else if objective_type == "drug_metabolism":
        evidence_weights = {
            "ms2_fragmentation": 1.4,       // Structure critical for metabolism
            "retention_time": 1.2,          // Chromatographic behavior important
            "isotope_pattern": 1.1,         // Confirms molecular formula
            "mass_match": 1.0,              // Basic identification
            "pathway_membership": 0.9       // Less critical than structure
        }
    else if objective_type == "environmental_analysis":
        evidence_weights = {
            "mass_match": 1.3,              // Accurate mass critical
            "isotope_pattern": 1.2,         // Confirms elemental composition
            "retention_time": 1.1,          // Chromatographic confirmation
            "ms2_fragmentation": 1.0,       // Structure helpful but not always available
            "pathway_membership": 0.7       // Less relevant for environmental compounds
        }
```

### Step 2: Confidence Threshold Optimization

```javascript
phase AdaptiveConfidenceThresholds:
    // Optimize confidence thresholds based on evidence quality
    evidence_quality_assessment = assess_evidence_quality(evidence_network)
    
    if evidence_quality_assessment.overall_quality > 0.9:
        confidence_threshold = 0.7  // Can be more permissive with high-quality evidence
    else if evidence_quality_assessment.overall_quality > 0.7:
        confidence_threshold = 0.8  // Standard threshold
    else:
        confidence_threshold = 0.9  // Stricter with lower quality evidence
    
    // Apply adaptive filtering
    filtered_annotations = filter_by_adaptive_confidence(
        annotations: raw_annotations,
        threshold: confidence_threshold,
        quality_metrics: evidence_quality_assessment
    )
```

### Step 3: Multi-Objective Evidence Integration

```javascript
phase MultiObjectiveIntegration:
    // Combine evidence for multiple related objectives
    primary_objective = "biomarker_discovery"
    secondary_objective = "pathway_characterization"
    
    // Build evidence networks for each objective
    biomarker_network = lavoisier.mzekezeke.build_evidence_network(
        data: dataset,
        objective: primary_objective,
        evidence_weights: biomarker_weights
    )
    
    pathway_network = lavoisier.mzekezeke.build_evidence_network(
        data: dataset,
        objective: secondary_objective,
        evidence_weights: pathway_weights
    )
    
    // Integrate evidence from both networks
    integrated_evidence = integrate_multi_objective_evidence(
        primary_network: biomarker_network,
        secondary_network: pathway_network,
        integration_weights: {"primary": 0.7, "secondary": 0.3}
    )
```

### Exercise 5.1
Design evidence weights for a food safety analysis objective:

```javascript
objective FoodSafetyAnalysis:
    target: "detect pesticide residues in food samples"
    // What evidence would be most important for pesticide detection?
    // Consider: regulatory requirements, detection limits, false positives
```

### Exercise 5.2
Create a dynamic evidence weighting system that adapts to sample type:

```javascript
function get_sample_specific_weights(sample_type):
    if sample_type == "plasma":
        return plasma_optimized_weights
    else if sample_type == "urine":
        return urine_optimized_weights
    // Add more sample types...
```

---

## Tutorial 6: Performance Optimization

### Learning Objectives
- Optimize Buhera scripts for large datasets
- Implement efficient validation strategies
- Monitor and improve execution performance

### Background
Large-scale studies require performance optimization without compromising scientific rigor.

### Step 1: Efficient Data Loading

```javascript
phase OptimizedDataLoading:
    // Load data in chunks for large datasets
    if dataset_size > 10_GB:
        chunk_size = calculate_optimal_chunk_size(
            available_memory: system_memory,
            dataset_size: dataset_size
        )
        
        // Process data in chunks
        for chunk in load_dataset_chunks(file_path: "large_dataset.mzML", 
                                        chunk_size: chunk_size):
            partial_network = lavoisier.mzekezeke.build_evidence_network(
                data: chunk,
                objective: current_objective,
                incremental: true
            )
            
            merge_evidence_networks(main_network, partial_network)
    else:
        // Standard loading for smaller datasets
        dataset = load_dataset(file_path: "dataset.mzML")
```

### Step 2: Parallel Validation

```javascript
validate ParallelValidation:
    // Run multiple validation checks in parallel
    validation_tasks = [
        "check_instrument_capability",
        "check_sample_size", 
        "check_pathway_consistency",
        "check_batch_effects"
    ]
    
    // Execute validations in parallel
    validation_results = execute_parallel_validation(validation_tasks)
    
    // Aggregate results
    for task, result in validation_results:
        if result.status == "failed":
            abort(f"Validation failed: {task} - {result.message}")
        else if result.status == "warning":
            warn(f"Validation warning: {task} - {result.message}")
```

### Step 3: Performance Monitoring

```javascript
phase PerformanceMonitoring:
    performance_monitor = start_performance_monitoring()
    
    // Monitor memory usage during evidence network building
    start_memory_tracking()
    evidence_network = lavoisier.mzekezeke.build_evidence_network(
        data: dataset,
        objective: current_objective
    )
    memory_usage = get_memory_usage()
    
    if memory_usage > memory_threshold:
        warn(f"High memory usage detected: {memory_usage} MB")
        suggest_optimization_strategies()
    
    // Monitor execution time
    execution_time = performance_monitor.get_elapsed_time()
    if execution_time > expected_time * 1.5:
        warn(f"Execution taking longer than expected: {execution_time} seconds")
```

### Step 4: Optimization Strategies

```javascript
phase OptimizationStrategies:
    // Choose optimization strategy based on data characteristics
    data_characteristics = analyze_data_characteristics(dataset)
    
    if data_characteristics.sparsity > 0.8:
        // Sparse data - use sparse matrix optimizations
        optimization_strategy = "sparse_matrix"
    else if data_characteristics.dimensionality > 10000:
        // High dimensional data - use dimensionality reduction
        optimization_strategy = "dimensionality_reduction"
    else:
        optimization_strategy = "standard"
    
    // Apply chosen optimization
    optimized_network = lavoisier.mzekezeke.build_evidence_network(
        data: dataset,
        objective: current_objective,
        optimization: optimization_strategy
    )
```

### Exercise 6.1
Create a performance benchmarking script that compares different optimization strategies:

```javascript
phase PerformanceBenchmark:
    strategies = ["standard", "sparse_matrix", "dimensionality_reduction"]
    
    for strategy in strategies:
        start_time = current_time()
        result = run_analysis_with_strategy(strategy)
        end_time = current_time()
        
        benchmark_results[strategy] = {
            "execution_time": end_time - start_time,
            "memory_usage": get_memory_usage(),
            "accuracy": result.accuracy
        }
    
    optimal_strategy = select_optimal_strategy(benchmark_results)
```

### Exercise 6.2
Implement a caching system for repeated analyses:

```javascript
phase CachedAnalysis:
    cache_key = generate_cache_key(dataset, objective, parameters)
    
    if cache_exists(cache_key):
        result = load_from_cache(cache_key)
        warn("Using cached results - set force_recompute=true to recalculate")
    else:
        result = perform_full_analysis()
        save_to_cache(cache_key, result)
```

---

## Summary and Next Steps

You've now learned how to:

1. **Write basic Buhera scripts** with objectives, validation, and phases
2. **Design biomarker discovery studies** with appropriate evidence weighting
3. **Create drug metabolism characterization** scripts with time-course analysis
4. **Implement advanced validation** with conditional logic
5. **Customize evidence networks** for specific research objectives
6. **Optimize performance** for large-scale studies

### Advanced Topics to Explore

- **Multi-instrument integration**: Combining data from different MS platforms
- **Cross-platform validation**: Validating results across different analytical methods
- **Machine learning integration**: Using ML models within Buhera workflows
- **Real-time analysis**: Implementing streaming analysis for online monitoring

### Best Practices Summary

1. **Always start with clear, measurable objectives**
2. **Implement comprehensive validation before analysis**
3. **Document your scientific reasoning in comments**
4. **Use meaningful variable names that reflect scientific concepts**
5. **Test scripts with small datasets before scaling up**
6. **Monitor performance and optimize as needed**

### Getting Help

- Check the [Language Reference](buhera-language-reference.md) for syntax details
- Review the [Integration Guide](buhera-integration.md) for Lavoisier integration
- Join the community discussions for advanced use cases

Happy scripting with Buhera! 