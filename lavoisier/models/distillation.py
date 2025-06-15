"""
Knowledge distillation module for creating specialized LLMs
"""
from typing import Dict, List, Optional, Union, Any, Callable
import os
import json
import time
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import logging
import traceback
from pathlib import Path
import shutil
import subprocess

import numpy as np
from tqdm import tqdm

from lavoisier.core.logging import get_logger
from lavoisier.models.versioning import ModelVersion, ModelMetadata
# from lavoisier.models.papers import PaperAnalyzer  # Commented out - papers not ready yet


class KnowledgeDistiller:
    """
    Knowledge distiller for creating specialized LLMs from academic papers and pipeline data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge distiller
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger("knowledge_distiller")
        
        # Base directory for temporary files
        self.temp_dir = Path(config.get("temp_dir", "/tmp/lavoisier_distillation"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Ollama configuration
        self.ollama_base_model = config.get("ollama_base_model", "llama3")
        self.ollama_path = config.get("ollama_path", "ollama")
        
        # Embedding model configuration
        self.embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        
        # Paper analyzer - commented out until papers are ready
        # self.paper_analyzer = PaperAnalyzer(config)
        self.paper_analyzer = None
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))
        
        self.logger.info("Knowledge distiller initialized")
    
    def distill_academic_model(self, 
                              papers_dir: str,
                              output_path: Optional[str] = None,
                              progress_callback: Optional[Callable[[float, str], None]] = None) -> str:
        """
        Distill knowledge from academic papers into an LLM
        
        Args:
            papers_dir: Directory containing academic papers
            output_path: Path to save the model (default: temp directory)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the created model
        """
        if self.paper_analyzer is None:
            raise NotImplementedError("Paper analysis is not available yet - papers collection in progress")
        
        papers_dir = Path(papers_dir)
        self.logger.info(f"Starting distillation from papers in {papers_dir}")
        
        # Report progress
        if progress_callback:
            progress_callback(0.0, "Starting paper analysis")
        
        # Create a temporary work directory
        work_dir = self.temp_dir / f"academic_distill_{int(time.time())}"
        work_dir.mkdir(exist_ok=True)
        
        try:
            # Get list of PDF files
            pdf_files = list(papers_dir.glob("*.pdf"))
            if not pdf_files:
                raise ValueError(f"No PDF files found in {papers_dir}")
            
            self.logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Analyze papers
            summaries = []
            knowledge_chunks = []
            
            for i, pdf_file in enumerate(pdf_files):
                self.logger.info(f"Analyzing paper {i+1}/{len(pdf_files)}: {pdf_file.name}")
                
                if progress_callback:
                    progress_callback(
                        i / len(pdf_files) * 0.6, 
                        f"Analyzing paper {i+1}/{len(pdf_files)}: {pdf_file.name}"
                    )
                
                # Analyze paper and extract knowledge
                try:
                    summary, chunks = self.paper_analyzer.analyze_paper(str(pdf_file))
                    summaries.append(summary)
                    knowledge_chunks.extend(chunks)
                except Exception as e:
                    self.logger.error(f"Error analyzing {pdf_file.name}: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    # Continue with other papers
            
            self.logger.info(f"Extracted {len(knowledge_chunks)} knowledge chunks from {len(pdf_files)} papers")
            
            if progress_callback:
                progress_callback(0.6, f"Creating knowledge base from {len(knowledge_chunks)} chunks")
            
            # Create knowledge base file
            kb_file = work_dir / "knowledge_base.txt"
            with open(kb_file, "w") as f:
                for chunk in knowledge_chunks:
                    f.write(f"{chunk}\n\n")
            
            self.logger.info(f"Created knowledge base at {kb_file}")
            
            # Create summary file
            summary_file = work_dir / "summaries.txt"
            with open(summary_file, "w") as f:
                for i, summary in enumerate(summaries):
                    f.write(f"Paper {i+1}: {summary}\n\n")
            
            self.logger.info(f"Created summaries at {summary_file}")
            
            # Create model definition (Modelfile)
            modelfile = work_dir / "Modelfile"
            with open(modelfile, "w") as f:
                f.write(f"FROM {self.ollama_base_model}\n\n")
                f.write("PARAMETER temperature 0.7\n")
                f.write("PARAMETER stop \"<|end|>\"\n\n")
                f.write("TEMPLATE \"\"\"{{.System}}\\n\\n{{.Prompt}}\"\"\"\n\n")
                f.write("SYSTEM \"\"\"You are a specialized assistant for mass spectrometry and metabolomics analysis.\n")
                f.write("You have deep knowledge of computational methods for analyzing metabolites, ")
                f.write("including techniques like mass spectrometry, chromatography, and bioinformatics approaches.\n")
                f.write("You can explain complex concepts, interpret spectra, discuss analytical methods, ")
                f.write("and provide advice on experimental design and data analysis in metabolomics research.\n")
                f.write("When responding, cite specific literature where relevant and provide technically accurate information.\n")
                f.write("Your knowledge is based on an extensive collection of academic publications in the field.\"\"\"\n\n")
                
                # Add knowledge base
                f.write(f"PARAMETER num_ctx 8192\n\n")
                
                # Set output path
                if output_path is None:
                    output_path = str(work_dir / "academic_model.bin")
                
                # Create the model using Ollama
                model_name = f"lavoisier_academic_{int(time.time())}"
                
                if progress_callback:
                    progress_callback(0.7, f"Creating model {model_name}")
                
                self.logger.info(f"Creating model {model_name} from {modelfile}")
                
                # Run Ollama create command
                cmd = [
                    self.ollama_path, "create", model_name,
                    "-f", str(modelfile)
                ]
                
                try:
                    process = subprocess.run(
                        cmd,
                        cwd=str(work_dir),
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    self.logger.info(f"Created model {model_name}: {process.stdout}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error creating model: {e.stderr}")
                    raise RuntimeError(f"Failed to create model: {e.stderr}")
                
                if progress_callback:
                    progress_callback(0.9, f"Exporting model {model_name}")
                
                # Export the model
                export_cmd = [
                    self.ollama_path, "export", model_name,
                    "-o", output_path
                ]
                
                try:
                    process = subprocess.run(
                        export_cmd,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    self.logger.info(f"Exported model to {output_path}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error exporting model: {e.stderr}")
                    raise RuntimeError(f"Failed to export model: {e.stderr}")
                
            if progress_callback:
                progress_callback(1.0, f"Model created at {output_path}")
            
            self.logger.info(f"Knowledge distillation complete, model saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error during knowledge distillation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Clean up temporary files if needed
            if self.config.get("cleanup_temp", True):
                try:
                    shutil.rmtree(work_dir)
                    self.logger.info(f"Cleaned up temporary directory {work_dir}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up: {str(e)}")
    
    def distill_pipeline_model(self, 
                             pipeline_type: str,
                             pipeline_data: Dict[str, Any],
                             output_path: Optional[str] = None,
                             progress_callback: Optional[Callable[[float, str], None]] = None) -> str:
        """
        Distill knowledge from pipeline data into an LLM
        
        Args:
            pipeline_type: Type of pipeline (numeric, visual)
            pipeline_data: Pipeline data dictionary
            output_path: Path to save the model (default: temp directory)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the created model
        """
        self.logger.info(f"Starting distillation from {pipeline_type} pipeline data")
        
        # Report progress
        if progress_callback:
            progress_callback(0.0, f"Starting {pipeline_type} pipeline model creation")
        
        # Create a temporary work directory
        work_dir = self.temp_dir / f"{pipeline_type}_distill_{int(time.time())}"
        work_dir.mkdir(exist_ok=True)
        
        try:
            # Extract knowledge from pipeline data
            if progress_callback:
                progress_callback(0.2, "Extracting knowledge from pipeline data")
            
            # Convert pipeline data to knowledge chunks
            knowledge_chunks = self._extract_pipeline_knowledge(pipeline_type, pipeline_data)
            
            self.logger.info(f"Extracted {len(knowledge_chunks)} knowledge chunks from {pipeline_type} pipeline data")
            
            if progress_callback:
                progress_callback(0.5, f"Creating knowledge base from {len(knowledge_chunks)} chunks")
            
            # Create knowledge base file
            kb_file = work_dir / "knowledge_base.txt"
            with open(kb_file, "w") as f:
                for chunk in knowledge_chunks:
                    f.write(f"{chunk}\n\n")
            
            # Create model definition (Modelfile)
            modelfile = work_dir / "Modelfile"
            with open(modelfile, "w") as f:
                f.write(f"FROM {self.ollama_base_model}\n\n")
                f.write("PARAMETER temperature 0.7\n")
                f.write("PARAMETER stop \"<|end|>\"\n\n")
                f.write("TEMPLATE \"\"\"{{.System}}\\n\\n{{.Prompt}}\"\"\"\n\n")
                
                # Different system prompts based on pipeline type
                if pipeline_type == "numeric":
                    f.write("SYSTEM \"\"\"You are a specialized assistant for numerical analysis of mass spectrometry data.\n")
                    f.write("You have deep knowledge of computational methods for processing and analyzing MS data, ")
                    f.write("including peak detection, alignment, normalization, and statistical analysis.\n")
                    f.write("You can explain algorithms, interpret results, and provide guidance on MS data processing.\n")
                    f.write("Your expertise is in numerical methods and computational approaches for metabolomics.\"\"\"\n\n")
                elif pipeline_type == "visual":
                    f.write("SYSTEM \"\"\"You are a specialized assistant for visual analysis of mass spectrometry data.\n")
                    f.write("You have deep knowledge of methods for visualizing and interpreting MS spectra, ")
                    f.write("including techniques for generating and analyzing spectral images and videos.\n")
                    f.write("You can explain visualization approaches, interpret visual patterns in MS data, ")
                    f.write("and provide guidance on visual representation of complex metabolomic information.\"\"\"\n\n")
                
                # Add knowledge base
                f.write(f"PARAMETER num_ctx 8192\n\n")
                
                # Set output path
                if output_path is None:
                    output_path = str(work_dir / f"{pipeline_type}_model.bin")
                
                # Create the model using Ollama
                model_name = f"lavoisier_{pipeline_type}_{int(time.time())}"
                
                if progress_callback:
                    progress_callback(0.7, f"Creating model {model_name}")
                
                self.logger.info(f"Creating model {model_name} from {modelfile}")
                
                # Run Ollama create command
                cmd = [
                    self.ollama_path, "create", model_name,
                    "-f", str(modelfile)
                ]
                
                try:
                    process = subprocess.run(
                        cmd,
                        cwd=str(work_dir),
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    self.logger.info(f"Created model {model_name}: {process.stdout}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error creating model: {e.stderr}")
                    raise RuntimeError(f"Failed to create model: {e.stderr}")
                
                if progress_callback:
                    progress_callback(0.9, f"Exporting model {model_name}")
                
                # Export the model
                export_cmd = [
                    self.ollama_path, "export", model_name,
                    "-o", output_path
                ]
                
                try:
                    process = subprocess.run(
                        export_cmd,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    self.logger.info(f"Exported model to {output_path}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error exporting model: {e.stderr}")
                    raise RuntimeError(f"Failed to export model: {e.stderr}")
                
            if progress_callback:
                progress_callback(1.0, f"Model created at {output_path}")
            
            self.logger.info(f"Knowledge distillation complete, model saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error during knowledge distillation: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Clean up temporary files if needed
            if self.config.get("cleanup_temp", True):
                try:
                    shutil.rmtree(work_dir)
                    self.logger.info(f"Cleaned up temporary directory {work_dir}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up: {str(e)}")
    
    def _extract_pipeline_knowledge(self, pipeline_type: str, pipeline_data: Dict[str, Any]) -> List[str]:
        """
        Extract knowledge chunks from pipeline data
        
        Args:
            pipeline_type: Type of pipeline (numeric, visual)
            pipeline_data: Pipeline data dictionary
            
        Returns:
            List of knowledge chunks
        """
        chunks = []
        
        # Extract knowledge based on pipeline type
        if pipeline_type == "numeric":
            # Extract information about numerical processing
            if "parameters" in pipeline_data:
                chunks.append(f"Numerical processing parameters: {json.dumps(pipeline_data['parameters'], indent=2)}")
            
            if "results" in pipeline_data:
                results = pipeline_data["results"]
                
                if "peaks" in results:
                    peaks_info = f"The analysis identified {len(results['peaks'])} peaks in the MS data."
                    peaks_info += " Key peaks include: "
                    top_peaks = sorted(results['peaks'], key=lambda p: p.get('intensity', 0), reverse=True)[:10]
                    peaks_info += ", ".join([f"m/z {p.get('mz', 'N/A')} (intensity: {p.get('intensity', 'N/A')})" for p in top_peaks])
                    chunks.append(peaks_info)
                
                if "compounds" in results:
                    compounds_info = f"The analysis identified {len(results['compounds'])} potential compounds."
                    compounds_info += " Top potential matches include: "
                    top_compounds = sorted(results['compounds'], key=lambda c: c.get('score', 0), reverse=True)[:5]
                    compounds_info += ", ".join([f"{c.get('name', 'Unknown')} (score: {c.get('score', 'N/A')})" for c in top_compounds])
                    chunks.append(compounds_info)
                
                if "statistics" in results:
                    stats = results["statistics"]
                    stats_info = "Statistical analysis results: "
                    stats_info += f"Signal-to-noise ratio: {stats.get('snr', 'N/A')}, "
                    stats_info += f"Mass accuracy: {stats.get('mass_accuracy', 'N/A')} ppm, "
                    stats_info += f"Resolution: {stats.get('resolution', 'N/A')}"
                    chunks.append(stats_info)
        
        elif pipeline_type == "visual":
            # Extract information about visual processing
            if "parameters" in pipeline_data:
                chunks.append(f"Visual processing parameters: {json.dumps(pipeline_data['parameters'], indent=2)}")
            
            if "results" in pipeline_data:
                results = pipeline_data["results"]
                
                if "images" in results:
                    images_info = f"The analysis generated {len(results['images'])} spectral images."
                    if "primary_dimensions" in results:
                        images_info += f" Primary dimensions for visualization: {results['primary_dimensions']}"
                    chunks.append(images_info)
                
                if "features" in results:
                    features_info = f"The visual analysis extracted {len(results['features'])} visual features."
                    if "key_features" in results:
                        features_info += " Key visual features include: "
                        features_info += ", ".join(results['key_features'])
                    chunks.append(features_info)
                
                if "patterns" in results:
                    patterns_info = "Identified visual patterns: "
                    patterns_info += ", ".join(results['patterns'])
                    chunks.append(patterns_info)
        
        return chunks
    
    def test_model(self, model_path: str, test_queries: List[str]) -> Dict[str, Any]:
        """
        Test a distilled model with a set of queries
        
        Args:
            model_path: Path to the model file
            test_queries: List of test queries
            
        Returns:
            Test results
        """
        self.logger.info(f"Testing model at {model_path} with {len(test_queries)} queries")
        
        # Import model to Ollama
        model_name = f"lavoisier_test_{int(time.time())}"
        import_cmd = [
            self.ollama_path, "import", model_name,
            "-f", model_path
        ]
        
        try:
            process = subprocess.run(
                import_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(f"Imported model as {model_name}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error importing model: {e.stderr}")
            raise RuntimeError(f"Failed to import model: {e.stderr}")
        
        # Test each query
        results = []
        
        try:
            for query in test_queries:
                self.logger.info(f"Testing query: {query}")
                
                # Run the query
                run_cmd = [
                    self.ollama_path, "run", model_name,
                    query
                ]
                
                try:
                    process = subprocess.run(
                        run_cmd,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    response = process.stdout
                    
                    results.append({
                        "query": query,
                        "response": response,
                        "success": True
                    })
                    
                except subprocess.CalledProcessError as e:
                    results.append({
                        "query": query,
                        "response": e.stderr,
                        "success": False
                    })
            
            # Return results
            return {
                "model_name": model_name,
                "num_queries": len(test_queries),
                "successful_queries": sum(1 for r in results if r["success"]),
                "results": results
            }
            
        finally:
            # Remove the test model
            try:
                cleanup_cmd = [
                    self.ollama_path, "rm", model_name
                ]
                subprocess.run(cleanup_cmd, check=False)
                self.logger.info(f"Removed test model {model_name}")
            except Exception as e:
                self.logger.error(f"Error removing test model: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        self.logger.info("Knowledge distiller cleaned up") 