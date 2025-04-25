"""
Paper analyzer for extracting knowledge from academic papers in PDF format
"""
from typing import Dict, List, Optional, Tuple, Any
import os
import re
import json
import logging
import tempfile
from pathlib import Path
import subprocess
import traceback
import math
import time

from tqdm import tqdm
import numpy as np
import fitz  # PyMuPDF

from lavoisier.core.logging import get_logger


class PaperAnalyzer:
    """
    Analyzer for extracting knowledge from academic papers in PDF format
    
    Capabilities:
    - Extract text, tables, figures, and formulas from PDFs
    - Generate summaries of papers
    - Extract structured knowledge chunks for model distillation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the paper analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger("paper_analyzer")
        
        # Extraction parameters
        self.chunk_size = config.get("chunk_size", 1000)  # Characters per chunk
        self.chunk_overlap = config.get("chunk_overlap", 200)  # Overlap between chunks
        self.min_chunk_size = config.get("min_chunk_size", 100)  # Minimum chunk size
        
        # OCR parameters
        self.perform_ocr = config.get("perform_ocr", False)
        self.ocr_tool = config.get("ocr_tool", "tesseract")
        
        # Regular expressions for section detection
        self.section_patterns = [
            r"^\s*(\d+\.?\s+)?Abstract\s*$",
            r"^\s*(\d+\.?\s+)?Introduction\s*$",
            r"^\s*(\d+\.?\s+)?Materials\s+and\s+Methods\s*$",
            r"^\s*(\d+\.?\s+)?Methodology\s*$", 
            r"^\s*(\d+\.?\s+)?Experimental\s*$",
            r"^\s*(\d+\.?\s+)?Results\s*$",
            r"^\s*(\d+\.?\s+)?Results\s+and\s+Discussion\s*$",
            r"^\s*(\d+\.?\s+)?Discussion\s*$",
            r"^\s*(\d+\.?\s+)?Conclusion\s*$",
            r"^\s*(\d+\.?\s+)?References\s*$"
        ]
        
        # Formula detection patterns
        self.formula_pattern = r"\\[a-zA-Z]+|[^a-zA-Z0-9\s\.,;:!?\(\)\[\]\{\}]+"
        
        self.logger.info("Paper analyzer initialized")
    
    def analyze_paper(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        Analyze a paper and extract knowledge
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            A tuple (summary, knowledge_chunks)
        """
        self.logger.info(f"Analyzing paper: {pdf_path}")
        
        try:
            # Extract text from PDF
            all_text, metadata, images = self._extract_from_pdf(pdf_path)
            
            # Extract structured sections
            sections = self._extract_sections(all_text)
            
            # Generate summary
            summary = self._generate_summary(all_text, metadata, sections)
            
            # Extract knowledge chunks
            chunks = self._extract_knowledge_chunks(all_text, sections, metadata, images)
            
            self.logger.info(f"Extracted {len(chunks)} knowledge chunks from {pdf_path}")
            return summary, chunks
            
        except Exception as e:
            self.logger.error(f"Error analyzing paper {pdf_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"Error analyzing paper: {str(e)}", []
    
    def _extract_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Extract text, metadata, and images from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            A tuple (all_text, metadata, images)
        """
        try:
            # Open the PDF
            pdf_document = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "num_pages": len(pdf_document)
            }
            
            # Extract text from each page
            all_text = ""
            images = []
            
            for page_num, page in enumerate(pdf_document):
                # Extract text
                page_text = page.get_text("text")
                all_text += page_text + "\n\n"
                
                # Extract images if needed
                if self.config.get("extract_images", False):
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        if base_image:
                            images.append({
                                "page": page_num,
                                "index": img_index,
                                "width": base_image["width"],
                                "height": base_image["height"],
                                "format": base_image["ext"],
                                "data": base_image["image"]  # Binary image data
                            })
            
            # Clean up text
            all_text = self._clean_text(all_text)
            
            return all_text, metadata, images
            
        except Exception as e:
            self.logger.error(f"Error extracting content from PDF: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken lines
        text = re.sub(r'(\w)- (\w)', r'\1\2', text)
        
        # Fix common OCR errors in scientific texts
        text = re.sub(r'l\(', '(', text)  # lowercase l to opening paren
        text = re.sub(r'l\.', '.', text)  # lowercase l to period
        text = re.sub(r'l,', ',', text)   # lowercase l to comma
        
        return text
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract structured sections from paper text
        
        Args:
            text: Full text of the paper
            
        Returns:
            Dictionary of section names and their contents
        """
        # Split text into lines for better section detection
        lines = text.split('\n')
        
        # Initialize sections dictionary and current section
        sections = {}
        current_section = "preamble"
        sections[current_section] = ""
        
        # Process each line
        for line in lines:
            # Check if line is a section header
            is_section_header = False
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Extract section name (without numbers)
                    section_name = re.sub(r'^\s*\d+\.?\s+', '', line.strip())
                    current_section = section_name.lower()
                    sections[current_section] = ""
                    is_section_header = True
                    break
            
            # If not a section header, add text to current section
            if not is_section_header:
                sections[current_section] += line + "\n"
        
        # Clean up section text
        for section in sections:
            sections[section] = sections[section].strip()
        
        return sections
    
    def _generate_summary(self, text: str, metadata: Dict[str, Any], sections: Dict[str, str]) -> str:
        """
        Generate a summary of the paper
        
        Args:
            text: Full text of the paper
            metadata: Paper metadata
            sections: Extracted sections
            
        Returns:
            Summary string
        """
        # Check if abstract is available
        if "abstract" in sections and sections["abstract"]:
            abstract = sections["abstract"]
        else:
            # Try to find the abstract in the full text
            abstract_match = re.search(r'abstract[.\s]*([^.]+(?:\. [^.]+){2,10})', text, re.IGNORECASE)
            if abstract_match:
                abstract = abstract_match.group(1)
            else:
                abstract = "Abstract not found"
        
        # Get title
        title = metadata.get("title", "Unknown title")
        if not title or title == "":
            # Try to extract from first lines
            first_lines = text.split('\n')[:5]
            for line in first_lines:
                if len(line) > 20 and len(line) < 200:  # Typical title length
                    title = line.strip()
                    break
        
        # Get conclusion if available
        conclusion = ""
        if "conclusion" in sections and sections["conclusion"]:
            conclusion_text = sections["conclusion"]
            # Get first paragraph
            conclusion_paras = conclusion_text.split('\n\n')
            conclusion = conclusion_paras[0] if conclusion_paras else conclusion_text[:500]
        
        # Compose summary
        summary = f"Title: {title}\n\n"
        summary += f"Abstract: {abstract[:500]}...\n\n"
        
        if conclusion:
            summary += f"Conclusion: {conclusion[:500]}...\n\n"
        
        # Add metadata
        if metadata.get("author"):
            summary += f"Authors: {metadata['author']}\n"
        
        return summary
    
    def _extract_knowledge_chunks(self, 
                                text: str, 
                                sections: Dict[str, str], 
                                metadata: Dict[str, Any],
                                images: List[Dict[str, Any]]) -> List[str]:
        """
        Extract knowledge chunks from the paper
        
        Args:
            text: Full text of the paper
            sections: Extracted sections
            metadata: Paper metadata
            images: Extracted images
            
        Returns:
            List of knowledge chunks
        """
        chunks = []
        
        # Add metadata chunk
        title = metadata.get("title", "Unknown title")
        author = metadata.get("author", "Unknown author")
        metadata_chunk = f"Paper: {title}\nAuthor(s): {author}\n"
        if "abstract" in sections and sections["abstract"]:
            metadata_chunk += f"Abstract: {sections['abstract'][:1000]}"
        chunks.append(metadata_chunk)
        
        # Process each section
        important_sections = ["introduction", "materials and methods", "methodology", 
                            "experimental", "results", "results and discussion", 
                            "discussion", "conclusion"]
        
        for section_name, section_text in sections.items():
            if not section_text or len(section_text) < self.min_chunk_size:
                continue
            
            # Prioritize important sections
            if section_name.lower() in important_sections:
                # Split into chunks with overlap
                section_chunks = self._split_into_chunks(
                    section_text, 
                    self.chunk_size, 
                    self.chunk_overlap, 
                    section_name
                )
                chunks.extend(section_chunks)
            else:
                # Less important sections get less chunks
                section_chunks = self._split_into_chunks(
                    section_text, 
                    self.chunk_size * 2,  # Larger chunks
                    self.chunk_overlap // 2,  # Less overlap
                    section_name
                )
                chunks.extend(section_chunks)
        
        # Look for formulas and special content
        formula_chunks = self._extract_formulas(text)
        if formula_chunks:
            chunks.extend(formula_chunks)
        
        # TODO: Process images if needed
        
        return chunks
    
    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int, section_name: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            section_name: Name of the section
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        if len(text) <= chunk_size:
            return [f"Section: {section_name}\n\n{text}"]
        
        # Split into sentences to avoid breaking in the middle of sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                # Save current chunk if it's not too small
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(f"Section: {section_name}\n\n{current_chunk.strip()}")
                
                # Start a new chunk, including overlap if possible
                overlap_text = ""
                if overlap > 0 and len(current_chunk) > overlap:
                    # Try to include complete sentences in the overlap
                    overlap_sentences = re.split(r'(?<=[.!?])\s+', current_chunk[-overlap:])
                    overlap_text = "".join(overlap_sentences[1:]) if len(overlap_sentences) > 1 else ""
                
                current_chunk = overlap_text + sentence + " "
        
        # Add the last chunk if it's not too small
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(f"Section: {section_name}\n\n{current_chunk.strip()}")
        
        return chunks
    
    def _extract_formulas(self, text: str) -> List[str]:
        """
        Extract mathematical formulas from text
        
        Args:
            text: Text containing formulas
            
        Returns:
            List of formula chunks
        """
        formula_chunks = []
        
        # Look for LaTeX-style formulas
        latex_formulas = re.findall(r'\$(.*?)\$', text)
        latex_formulas.extend(re.findall(r'\\\[(.*?)\\\]', text))
        latex_formulas.extend(re.findall(r'\\\((.*?)\\\)', text))
        
        if latex_formulas:
            formula_text = "Mathematical Formulas in the Paper:\n\n"
            for i, formula in enumerate(latex_formulas):
                if len(formula.strip()) > 5:  # Ignore very short formulas that might be false positives
                    formula_text += f"Formula {i+1}: ${formula.strip()}$\n\n"
            
            if len(formula_text) > 100:  # Only add if we found substantial formulas
                formula_chunks.append(formula_text)
        
        # Look for inline equations with patterns like "Eq. 1" or "Equation (2)"
        equation_refs = re.findall(r'Eq(?:uation)?\s*\.?\s*\(?\d+\)?', text)
        if equation_refs:
            for ref in equation_refs:
                # Try to find the context around the equation reference
                context_pattern = f"[^.]*{re.escape(ref)}[^.]*\."
                equation_contexts = re.findall(context_pattern, text)
                
                for context in equation_contexts:
                    if len(context.strip()) > 50:
                        formula_chunks.append(f"Equation Reference: {context.strip()}")
        
        return formula_chunks 