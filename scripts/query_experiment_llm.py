#!/usr/bin/env python3

"""
Query Experiment-Specific LLM

This script demonstrates how to query the experiment-specific LLM created
by the Lavoisier MTBLS1707 analysis. The LLM contains all knowledge from
the analysis and can answer questions about:

1. Analysis results and findings
2. HuggingFace model contributions  
3. Sample-specific insights
4. Methodological recommendations
5. Comparative analysis between pipelines
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add Lavoisier to path
sys.path.append(str(Path(__file__).parent.parent))

from lavoisier.core.logging import get_logger

logger = get_logger("experiment_llm_query")

class ExperimentLLMQueryInterface:
    """Interface for querying experiment-specific LLMs created by Lavoisier"""
    
    def __init__(self, knowledge_base_path: str):
        """
        Initialize the query interface
        
        Args:
            knowledge_base_path: Path to the experiment knowledge base JSON
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {knowledge_base_path}")
        
        # Load knowledge base
        with open(self.knowledge_base_path, 'r') as f:
            self.knowledge_base = json.load(f)
        
        self.model_path = self.knowledge_base.get('experiment_llm_path')
        self.model_metadata = self.knowledge_base.get('model_metadata', {})
        
        logger.info(f"Loaded experiment LLM: {self.model_metadata.get('experiment_id', 'Unknown')}")
        logger.info(f"Samples analyzed: {self.model_metadata.get('samples_analyzed', 0)}")
        logger.info(f"HF models used: {self.model_metadata.get('hf_models_used', [])}")
    
    def query_llm(self, question: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Query the experiment-specific LLM
        
        Args:
            question: Question to ask the LLM
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Querying experiment LLM: {question}")
        
        try:
            # Extract model name from path (assuming Ollama model)
            model_name = Path(self.model_path).stem if self.model_path else "lavoisier_experiment"
            
            # Run ollama query
            cmd = ["ollama", "run", model_name, question]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if process.returncode == 0:
                response = process.stdout.strip()
                
                return {
                    'success': True,
                    'question': question,
                    'response': response,
                    'model_used': model_name,
                    'experiment_id': self.model_metadata.get('experiment_id'),
                    'timestamp': self.model_metadata.get('created_at')
                }
            else:
                error_msg = process.stderr.strip() or "Unknown error"
                logger.error(f"LLM query failed: {error_msg}")
                
                return {
                    'success': False,
                    'question': question,
                    'error': error_msg,
                    'model_used': model_name
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'question': question,
                'error': 'Query timed out after 60 seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'question': question,
                'error': str(e)
            }
    
    def run_predefined_queries(self) -> List[Dict[str, Any]]:
        """Run a set of predefined queries to demonstrate capabilities"""
        
        predefined_queries = [
            "What were the main findings from the MTBLS1707 analysis?",
            "Which HuggingFace models were used and what did each contribute?",
            "How many structure predictions were made by the SpecTUS model?",
            "What was the total number of spectrum embeddings created?",
            "Compare the performance of the numerical vs visual pipelines.",
            "Which samples showed the most interesting metabolite profiles?",
            "What would you recommend for follow-up experiments?",
            "Explain the novel computer vision approach used in this analysis.",
            "How did the knowledge distillation process work for this experiment?",
            "What makes this analysis different from traditional metabolomics workflows?"
        ]
        
        results = []
        
        logger.info(f"Running {len(predefined_queries)} predefined queries...")
        
        for i, query in enumerate(predefined_queries, 1):
            logger.info(f"Query {i}/{len(predefined_queries)}: {query[:50]}...")
            
            result = self.query_llm(query)
            results.append(result)
            
            if result['success']:
                print(f"\n{'='*80}")
                print(f"QUERY {i}: {query}")
                print(f"{'='*80}")
                print(result['response'])
                print()
            else:
                print(f"\n❌ Query {i} failed: {result.get('error', 'Unknown error')}")
        
        return results
    
    def interactive_query_session(self):
        """Start an interactive query session"""
        
        print(f"\n{'='*80}")
        print("🧠 LAVOISIER EXPERIMENT LLM - INTERACTIVE SESSION")
        print(f"{'='*80}")
        print(f"Experiment: {self.model_metadata.get('experiment_id', 'Unknown')}")
        print(f"Dataset: MTBLS1707")
        print(f"Samples Analyzed: {self.model_metadata.get('samples_analyzed', 0)}")
        print(f"HuggingFace Models: {', '.join(self.model_metadata.get('hf_models_used', []))}")
        print(f"Pipelines Used: {', '.join(self.model_metadata.get('pipelines_used', []))}")
        print(f"{'='*80}")
        print("\nThis LLM contains all knowledge from your MTBLS1707 analysis.")
        print("You can ask questions about:")
        print("  • Analysis results and findings")
        print("  • HuggingFace model contributions")
        print("  • Sample-specific insights")
        print("  • Methodological recommendations")
        print("  • Comparative analysis")
        print("\nType 'quit' to exit, 'help' for example queries.\n")
        
        while True:
            try:
                question = input("🔍 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif question.lower() in ['help', 'h']:
                    self._show_example_queries()
                    continue
                elif not question:
                    continue
                
                print("\n🤔 Thinking...")
                result = self.query_llm(question)
                
                if result['success']:
                    print(f"\n🧠 LLM Response:")
                    print(f"{'─'*60}")
                    print(result['response'])
                    print(f"{'─'*60}\n")
                else:
                    print(f"\n❌ Error: {result.get('error', 'Unknown error')}\n")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}\n")
    
    def _show_example_queries(self):
        """Show example queries"""
        example_queries = self.knowledge_base.get('sample_queries', [])
        
        print(f"\n{'─'*60}")
        print("📝 EXAMPLE QUERIES")
        print(f"{'─'*60}")
        
        for i, query in enumerate(example_queries, 1):
            print(f"{i}. {query}")
        
        print(f"{'─'*60}\n")
    
    def export_query_results(self, results: List[Dict[str, Any]], output_path: str):
        """Export query results to JSON file"""
        
        export_data = {
            'experiment_metadata': self.model_metadata,
            'query_results': results,
            'export_timestamp': __import__('time').time(),
            'total_queries': len(results),
            'successful_queries': sum(1 for r in results if r['success'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Query results exported to: {output_path}")

def main():
    """Main execution function"""
    
    # Default paths
    default_knowledge_base = "scripts/results/mtbls1707_analysis/queryable_knowledge/mtbls1707_experiment_llm.json"
    
    # Check if knowledge base exists
    if not Path(default_knowledge_base).exists():
        print(f"❌ Knowledge base not found: {default_knowledge_base}")
        print("\n💡 To create an experiment LLM:")
        print("   1. Run: python scripts/run_mtbls1707_analysis_REAL.py")
        print("   2. Wait for analysis to complete")
        print("   3. The experiment LLM will be created automatically")
        print("\n🔍 Looking for existing knowledge bases...")
        
        # Search for other knowledge bases
        results_dir = Path("scripts/results")
        if results_dir.exists():
            kb_files = list(results_dir.glob("**/queryable_knowledge/*.json"))
            if kb_files:
                print(f"\nFound {len(kb_files)} knowledge base(s):")
                for i, kb_file in enumerate(kb_files, 1):
                    print(f"  {i}. {kb_file}")
                
                choice = input(f"\nSelect knowledge base (1-{len(kb_files)}) or press Enter to exit: ").strip()
                if choice and choice.isdigit() and 1 <= int(choice) <= len(kb_files):
                    default_knowledge_base = str(kb_files[int(choice) - 1])
                else:
                    return
            else:
                print("No knowledge bases found. Please run an analysis first.")
                return
        else:
            print("No results directory found. Please run an analysis first.")
            return
    
    try:
        # Initialize query interface
        query_interface = ExperimentLLMQueryInterface(default_knowledge_base)
        
        print(f"\n🎯 Loaded experiment LLM successfully!")
        
        # Ask user what they want to do
        print("\nWhat would you like to do?")
        print("1. Run predefined demo queries")
        print("2. Start interactive query session")
        print("3. Both")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice in ['1', '3']:
            print("\n🚀 Running predefined demo queries...")
            results = query_interface.run_predefined_queries()
            
            # Export results
            output_path = Path(default_knowledge_base).parent / "demo_query_results.json"
            query_interface.export_query_results(results, str(output_path))
            
            print(f"\n✅ Demo completed! Results exported to: {output_path}")
            print(f"📊 Success rate: {sum(1 for r in results if r['success'])}/{len(results)} queries")
        
        if choice in ['2', '3']:
            if choice == '3':
                input("\nPress Enter to start interactive session...")
            
            query_interface.interactive_query_session()
        
    except Exception as e:
        logger.error(f"Error in query interface: {e}")
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main() 