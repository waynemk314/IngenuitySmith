"""
Instantiator - Most flexible components
Contains state definitions, framework class, and main execution logic
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pathlib import Path
import json
import os
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Import infrastructure and agents
from infrastructure import load_config, route_next_agent, ModelFactory, AgentType
from agents import OrchestratorAgent, CoderAgent, RunnerAgent, ProseAgent

# prompt examples
from prompts import coding_task_prompts


class AgentState(TypedDict):
    """State shared across all agents in the graph"""
    messages: Annotated[List[BaseMessage], add_messages]
    original_request: str
    current_code: str
    execution_results: Dict[str, Any]
    prose_feedback: str
    iteration_count: int
    max_iterations: int
    status: str  # "in_progress", "completed", "failed"
    next_agent: str
    errors: List[str]

class AgenticDevelopmentFramework:
    """Main framework class that orchestrates the development process"""
    
    def __init__(self, config: Optional[Dict[str, str]] = None,
                 coder_initial_prompt: Optional[str] = None,
                 coder_fix_prompt: Optional[str] = None,
                 prose_review_prompt: Optional[str] = None):
        """
        Initialize the framework with optional custom prompts
        
        Args:
            config: Configuration dictionary (defaults to loading from .env)
            coder_initial_prompt: Custom prompt for initial code generation
            coder_fix_prompt: Custom prompt for code fixing
            prose_review_prompt: Custom prompt for code review
        """
        self.config = config or load_config()
        self.model_factory = ModelFactory(self.config)
        
        # Initialize agents with custom prompts if provided
        self.orchestrator = OrchestratorAgent(self.model_factory)
        self.coder = CoderAgent(
            self.model_factory, 
            initial_prompt=coder_initial_prompt,
            fix_prompt=coder_fix_prompt
        )
        self.runner = RunnerAgent(self.config)
        self.prose = ProseAgent(
            self.model_factory,
            review_prompt=prose_review_prompt
        )
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("orchestrator", self.orchestrator)
        workflow.add_node("coder", self.coder)
        workflow.add_node("runner", self.runner)
        workflow.add_node("prose", self.prose)
        
        # Define edges
        workflow.set_entry_point("orchestrator")
        workflow.add_conditional_edges(
            "orchestrator",
            route_next_agent,
            {
                "coder": "coder",
                "runner": "runner", 
                "prose": "prose",
                END: END
            }
        )
        
        # All agents return to orchestrator
        workflow.add_edge("coder", "orchestrator")
        workflow.add_edge("runner", "orchestrator")
        workflow.add_edge("prose", "orchestrator")
        
        return workflow.compile()
    
    def _save_final_output(self, final_state: Dict[str, Any], filename: str = "output.py", 
                          save_metadata: bool = False) -> bool:
        """Save the final code and optionally metadata in the shared directory"""
        print(f"🔍 Debug: Attempting to save final output...")
        print(f"   Filename: {filename}")
        print(f"   Save metadata: {save_metadata}")
        
        try:
            # Check if we have code to save
            if not final_state.get('current_code'):
                print("   ⚠️ No code to save - 'current_code' is empty or missing")
                return False
            
            print(f"   Code length: {len(final_state['current_code'])} characters")
            
            # Get and validate the host script directory
            host_script_dir_str = self.config.get("OUTPUT_SAVE_DIR")
            if not host_script_dir_str:
                print("   ❌ OUTPUT_SAVE_DIR not configured in environment")
                # Fallback to current directory
                host_script_dir_str = "./output"
                print(f"   Using fallback directory: {host_script_dir_str}")
            
            host_script_dir = Path(host_script_dir_str).resolve()
            output_path = host_script_dir / filename
            
            print(f"   Target directory: {host_script_dir}")
            print(f"   Full output path: {output_path}")
            
            # Ensure directory exists
            try:
                host_script_dir.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Directory created/verified: {host_script_dir}")
            except Exception as e:
                print(f"   ❌ Failed to create directory: {e}")
                return False
            
            # Check if directory is writable
            if not os.access(host_script_dir, os.W_OK):
                print(f"   ❌ Directory is not writable: {host_script_dir}")
                return False
            
            # Write the final code
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_state['current_code'])
                print(f"   ✅ Final code saved as: {output_path}")
                
                # Verify the file was written
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    print(f"   ✅ File verification: {file_size} bytes written")
                else:
                    print(f"   ❌ File was not created successfully")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Failed to write code file: {e}")
                return False
            
            # Optionally save metadata
            if save_metadata:
                try:
                    metadata_path = host_script_dir / f"{filename.rsplit('.', 1)[0]}_metadata.json"
                    metadata = {
                        "original_request": final_state.get('original_request', ''),
                        "iterations": final_state.get('iteration_count', 0),
                        "status": final_state.get('status', ''),
                        "execution_results": final_state.get('execution_results', {}),
                        "prose_feedback": final_state.get('prose_feedback', ''),
                        "errors": final_state.get('errors', []),
                        "timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "N/A"
                    }
                    
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, default=str)
                    print(f"   ✅ Metadata saved as: {metadata_path}")
                    
                except Exception as e:
                    print(f"   ⚠️ Failed to save metadata (code still saved): {e}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Unexpected error saving output: {e}")
            import traceback
            print(f"   ❌ Full traceback: {traceback.format_exc()}")
            return False
    
    def develop(self, request: str, max_iterations: int = 5) -> Dict[str, Any]:
        """Main entry point for development requests"""
        print(f"🚀 Starting Development: {request}\n")
        
        initial_state = AgentState(
            messages=[],
            original_request=request,
            current_code="",
            execution_results={},
            prose_feedback="",
            iteration_count=0,
            max_iterations=max_iterations,
            status="starting",
            next_agent="orchestrator",
            errors=[]
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            print("\n" + "="*50)
            print("🎯 DEVELOPMENT COMPLETE")
            print("="*50)
            print(f"Status: {final_state['status']}")
            print(f"Iterations: {final_state['iteration_count']}")
            
            # Debug information about the saving conditions
            print(f"\n🔍 Save Conditions Check:")
            print(f"   Status == 'completed': {final_state['status'] == 'completed'}")
            print(f"   Has current_code: {bool(final_state.get('current_code'))}")
            print(f"   Execution status_code: {final_state.get('execution_results', {}).get('status_code')}")
            print(f"   Status code == 0: {final_state.get('execution_results', {}).get('status_code') == 0}")
            
            # Modified condition - save if we have code, regardless of execution status
            should_save = (
                final_state['status'] == 'completed' and 
                final_state.get('current_code')
            )
            
            # Alternative: Always save if we have code (remove other conditions)
            # should_save = bool(final_state.get('current_code'))
            
            if should_save:
                print("\n💾 Saving Final Output...")
                success = self._save_final_output(final_state, save_metadata=True)
                if not success:
                    print("   ⚠️ Save operation failed - check the debug output above")
            else:
                print("\n💾 Skipping save - conditions not met")
                print("   Either status != 'completed' or no code generated")
            
            if final_state['current_code']:
                print(f"\n📄 Final Code:\n{final_state['current_code']}")
            
            if final_state['execution_results']:
                print(f"\n🔍 Execution Results: {final_state['execution_results']}")
            
            if final_state['prose_feedback']:
                print(f"\n📝 Final Prose Feedback: {final_state['prose_feedback']}")
            
            if final_state['errors']:
                print(f"\n⚠️ Errors Encountered: {final_state['errors']}")
            
            return final_state
            
        except Exception as e:
            print(f"\n❌ Framework error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return {"error": str(e), "status": "failed"}

# --- Usage Example ---
if __name__ == "__main__":
    # Initialize the framework with default prompts
    framework = AgenticDevelopmentFramework()
    
    # Example development request
    result = framework.develop(
        request=coding_task_prompts[0], # use predefined prompts
        max_iterations=5
    )
    
    # Example with custom prompts
    custom_coder_prompt = """You are an expert Python developer. Create efficient, well-documented Python code for this request:

{context}

Requirements:
- Write clean, production-ready code
- Include comprehensive docstrings
- Use type hints where appropriate
- Follow modern Python best practices
- Ensure code is well-tested and robust

Return ONLY the Python code, no explanations."""
    
    # framework_with_custom_prompts = AgenticDevelopmentFramework(
    #     coder_initial_prompt=custom_coder_prompt
    # )
    
    # result = framework_with_custom_prompts.develop(
    #     request="Create a web scraper that extracts article titles from a news website",
    #     max_iterations=3
    # )