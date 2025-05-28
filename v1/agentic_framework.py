from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from enum import Enum
import os
import uuid
import time
from pathlib import Path
import docker
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Model imports (from your validator)
try:
    from langchain_ollama import ChatOllama  # Updated import
except ImportError:
    from langchain_community.chat_models import ChatOllama  # Fallback
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# --- Configuration (from your validator.py) ---
def load_config():
    """Loads configuration from .env file and returns it as a dictionary."""
    load_dotenv()
    return {
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
        "OLLAMA_CODER_MODEL_NAME": os.getenv("OLLAMA_CODER_MODEL_NAME"),
        "OLLAMA_PROSE_MODEL_NAME": os.getenv("OLLAMA_PROSE_MODEL_NAME"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "REMOTE_CODER_MODEL_PROVIDER": os.getenv("REMOTE_CODER_MODEL_PROVIDER"),
        "REMOTE_CODER_MODEL_NAME": os.getenv("REMOTE_CODER_MODEL_NAME"),
        "REMOTE_PROSE_MODEL_PROVIDER": os.getenv("REMOTE_PROSE_MODEL_PROVIDER"),
        "REMOTE_PROSE_MODEL_NAME": os.getenv("REMOTE_PROSE_MODEL_NAME"),
        "RUNNER_DOCKER_IMAGE": os.getenv("RUNNER_DOCKER_IMAGE"),
        "RUNNER_HOST_SCRIPT_DIR": os.getenv("RUNNER_HOST_SCRIPT_DIR"),
        "RUNNER_CONTAINER_SCRIPT_DIR": os.getenv("RUNNER_CONTAINER_SCRIPT_DIR"),
    }

# --- State Definition ---
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

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    CODER = "coder"
    RUNNER = "runner" 
    PROSE = "prose"

# --- Model Factory ---
class ModelFactory:
    def __init__(self, config: Dict[str, str]):
        self.config = config
    
    def get_model(self, agent_type: AgentType, prefer_local: bool = True):
        """Get appropriate model for agent type"""
        if agent_type == AgentType.CODER:
            return self._get_coder_model(prefer_local)
        elif agent_type == AgentType.PROSE:
            return self._get_prose_model(prefer_local)
        else:
            # Default for orchestrator - use a fast model
            return self._get_default_model()
    
    def _get_coder_model(self, prefer_local: bool):
        if prefer_local and self.config["OLLAMA_CODER_MODEL_NAME"]:
            return ChatOllama(
                model=self.config["OLLAMA_CODER_MODEL_NAME"],
                base_url=self.config["OLLAMA_BASE_URL"],
                temperature=0.1
            )
        elif self.config["REMOTE_CODER_MODEL_PROVIDER"] == "openai":
            return ChatOpenAI(
                model_name=self.config["REMOTE_CODER_MODEL_NAME"],
                openai_api_key=self.config["OPENAI_API_KEY"],
                temperature=0.1
            )
        elif self.config["REMOTE_CODER_MODEL_PROVIDER"] == "anthropic":
            return ChatAnthropic(
                model_name=self.config["REMOTE_CODER_MODEL_NAME"],
                anthropic_api_key=self.config["ANTHROPIC_API_KEY"],
                temperature=0.1
            )
        else:
            raise ValueError("No coder model configured")
    
    def _get_prose_model(self, prefer_local: bool):
        if prefer_local and self.config["OLLAMA_PROSE_MODEL_NAME"]:
            return ChatOllama(
                model=self.config["OLLAMA_PROSE_MODEL_NAME"],
                base_url=self.config["OLLAMA_BASE_URL"],
                temperature=0
            )
        elif self.config["REMOTE_PROSE_MODEL_PROVIDER"] == "openai":
            return ChatOpenAI(
                model_name=self.config["REMOTE_PROSE_MODEL_NAME"],
                openai_api_key=self.config["OPENAI_API_KEY"],
                temperature=0
            )
        elif self.config["REMOTE_PROSE_MODEL_PROVIDER"] == "anthropic":
            return ChatAnthropic(
                model_name=self.config["REMOTE_PROSE_MODEL_NAME"],
                anthropic_api_key=self.config["ANTHROPIC_API_KEY"],
                temperature=0
            )
        else:
            raise ValueError("No prose model configured")
    
    def _get_default_model(self):
        # Use fastest available model for orchestrator
        if self.config["OPENAI_API_KEY"]:
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=self.config["OPENAI_API_KEY"],
                temperature=0
            )
        elif self.config["ANTHROPIC_API_KEY"]:
            return ChatAnthropic(
                model_name="claude-3-haiku-20240307",
                anthropic_api_key=self.config["ANTHROPIC_API_KEY"],
                temperature=0
            )
        else:
            raise ValueError("No default model available")

# --- Agent Implementations ---
class OrchestratorAgent:
    def __init__(self, model_factory: ModelFactory):
        self.model = model_factory.get_model(AgentType.ORCHESTRATOR)
    
    def __call__(self, state: AgentState) -> AgentState:
        """Orchestrator decides next steps based on current state"""
        print(f"🎭 Orchestrator - Iteration {state['iteration_count']}")
        
        # Initialize if first run
        if state["iteration_count"] == 0:
            state["next_agent"] = "coder"
            state["status"] = "in_progress"
            state["iteration_count"] = 1
            print("   → Routing to Coder for initial implementation")
            return state
        
        # Check if we've hit max iterations
        if state["iteration_count"] >= state["max_iterations"]:
            state["status"] = "completed"
            state["next_agent"] = "end"
            print("   → Max iterations reached, ending")
            return state
        
        # Decision logic based on current state
        if state["current_code"] and not state["execution_results"]:
            state["next_agent"] = "runner"
            print("   → Code exists but not tested, routing to Runner")
        elif state["execution_results"] and state["execution_results"].get("status_code") != 0:
            state["next_agent"] = "coder"
            print("   → Execution failed, routing back to Coder for fixes")
        elif state["execution_results"] and not state["prose_feedback"]:
            state["next_agent"] = "prose"
            print("   → Code executes successfully, routing to Prose for review")
        elif state["prose_feedback"] and "issues" in state["prose_feedback"].lower():
            state["next_agent"] = "coder"
            print("   → Prose found issues, routing back to Coder for cleanup")
        else:
            state["status"] = "completed"
            state["next_agent"] = "end"
            print("   → All checks passed, task completed!")
        
        return state

class CoderAgent:
    def __init__(self, model_factory: ModelFactory):
        self.model = model_factory.get_model(AgentType.CODER)
    
    def __call__(self, state: AgentState) -> AgentState:
        """Coder generates or fixes Python code"""
        print("👨‍💻 Coder Agent Working...")
        
        # Build context for the coder
        context = f"Original Request: {state['original_request']}\n"
        
        if state["current_code"]:
            context += f"\nCurrent Code:\n{state['current_code']}\n"
        
        if state["execution_results"]:
            context += f"\nExecution Results:\n{state['execution_results']}\n"
        
        if state["prose_feedback"]:
            context += f"\nProse Feedback:\n{state['prose_feedback']}\n"
        
        # Create coding prompt
        if not state["current_code"]:
            prompt = f"""You are a Python coding expert. Create clean, working Python code for this request:

{context}

Requirements:
- Write complete, executable Python code
- Include proper error handling
- Add docstrings and comments
- Follow PEP 8 standards
- Make it production-ready

Return ONLY the Python code, no explanations."""
        else:
            prompt = f"""You are a Python coding expert. Fix the existing code based on the feedback:

{context}

Requirements:
- Fix any execution errors
- Address prose feedback if provided
- Maintain existing functionality
- Follow PEP 8 standards
- Return ONLY the corrected Python code, no explanations."""
        
        try:
            response = self.model.invoke(prompt)
            new_code = response.content.strip()
            
            # Extract code from markdown if present
            if "```python" in new_code:
                new_code = new_code.split("```python")[1].split("```")[0].strip()
            elif "```" in new_code:
                new_code = new_code.split("```")[1].split("```")[0].strip()
            
            state["current_code"] = new_code
            state["execution_results"] = {}  # Clear previous results
            state["prose_feedback"] = ""    # Clear previous feedback
            state["iteration_count"] += 1
            
            print(f"   ✅ Generated {len(new_code)} characters of code")
            
        except Exception as e:
            print(f"   ❌ Coder error: {e}")
            state["errors"].append(f"Coder error: {e}")
        
        return state

class RunnerAgent:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.docker_client = docker.from_env()
    
    def __call__(self, state: AgentState) -> AgentState:
        """Runner executes Python code in Docker container"""
        print("🏃 Runner Agent Testing Code...")
        
        if not state["current_code"]:
            print("   ❌ No code to execute")
            state["errors"].append("No code provided to runner")
            return state
        
        try:
            # Create unique script file
            script_name = f"test_run_{uuid.uuid4().hex[:8]}.py"
            host_script_dir = Path(self.config["RUNNER_HOST_SCRIPT_DIR"])
            host_script_path = host_script_dir / script_name
            
            # Ensure directory exists
            host_script_dir.mkdir(parents=True, exist_ok=True)
            
            # Write code to file
            with open(host_script_path, "w") as f:
                f.write(state["current_code"])
            
            # Prepare Docker execution
            container_script_dir = self.config["RUNNER_CONTAINER_SCRIPT_DIR"].replace('\\', '/')
            container_script_path = f"{container_script_dir}/{script_name}"
            
            volumes_dict = {
                str(host_script_dir.resolve()): {
                    'bind': container_script_dir,
                    'mode': 'rw'
                }
            }
            
            # Run in Docker
            container = self.docker_client.containers.run(
                image=self.config["RUNNER_DOCKER_IMAGE"],
                command=["python3", container_script_path],
                volumes=volumes_dict,
                detach=True,
                remove=True
            )
            
            result = container.wait()
            logs = container.logs()
            
            execution_result = {
                "status_code": result["StatusCode"],
                "output": logs.decode('utf-8'),
                "timestamp": time.time()
            }
            
            state["execution_results"] = execution_result
            
            if result["StatusCode"] == 0:
                print(f"   ✅ Code executed successfully")
            else:
                print(f"   ❌ Code execution failed (exit code: {result['StatusCode']})")
            
            # Cleanup
            if host_script_path.exists():
                host_script_path.unlink()
            
        except Exception as e:
            print(f"   ❌ Runner error: {e}")
            state["errors"].append(f"Runner error: {e}")
            state["execution_results"] = {
                "status_code": -1,
                "output": f"Runner error: {e}",
                "timestamp": time.time()
            }
        
        return state

class ProseAgent:
    def __init__(self, model_factory: ModelFactory):
        self.model = model_factory.get_model(AgentType.PROSE)
    
    def __call__(self, state: AgentState) -> AgentState:
        """Prose agent reviews code for style and best practices"""
        print("📝 Prose Agent Reviewing Code...")
        
        if not state["current_code"]:
            print("   ❌ No code to review")
            return state
        
        prompt = f"""You are a Python code reviewer focused on brevity and PEP 8 compliance.

Review this code:

```python
{state['current_code']}
```

Provide feedback on:
1. PEP 8 compliance issues
2. Code brevity improvements
3. Readability enhancements
4. Best practices

If the code is good, respond with "APPROVED: Code meets standards."
If issues exist, provide specific, actionable feedback starting with "ISSUES FOUND:" followed by numbered points.

Be concise but thorough."""
        
        try:
            response = self.model.invoke(prompt)
            feedback = response.content.strip()
            state["prose_feedback"] = feedback
            
            if "APPROVED" in feedback:
                print("   ✅ Code approved by prose review")
            else:
                print("   📋 Issues found, feedback provided")
            
        except Exception as e:
            print(f"   ❌ Prose error: {e}")
            state["errors"].append(f"Prose error: {e}")
            state["prose_feedback"] = "Error during prose review"
        
        return state

# --- Router Function ---
def route_next_agent(state: AgentState) -> str:
    """Route to next agent based on orchestrator decision"""
    next_agent = state.get("next_agent", "end")
    if next_agent == "end":
        return END
    return next_agent

# --- Main Framework Class ---
class AgenticDevelopmentFramework:
    def __init__(self, config: Optional[Dict[str, str]] = None):
        self.config = config or load_config()
        self.model_factory = ModelFactory(self.config)
        
        # Initialize agents
        self.orchestrator = OrchestratorAgent(self.model_factory)
        self.coder = CoderAgent(self.model_factory)
        self.runner = RunnerAgent(self.config)
        self.prose = ProseAgent(self.model_factory)
        
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
            return {"error": str(e), "status": "failed"}

# --- Usage Example ---
if __name__ == "__main__":
    # Initialize the framework
    framework = AgenticDevelopmentFramework()
    
    # Example development request
    result = framework.develop(
        request="Create a Python function that calculates the value of pi to 10 decimal places using the Monte Carlo method.",
        max_iterations=5
    )