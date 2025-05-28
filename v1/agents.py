"""
Agent Definitions - Semi-fixed components
Contains all agent implementations with configurable prompts
"""

from typing import Dict, Any, Optional
import uuid
import time
from pathlib import Path
import docker
from infrastructure import ModelFactory, AgentType

class OrchestratorAgent:
    """Orchestrator agent that manages workflow and routing decisions"""
    
    def __init__(self, model_factory: ModelFactory):
        self.model = model_factory.get_model(AgentType.ORCHESTRATOR)
    
    def __call__(self, state) -> dict:
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
    """Coder agent that generates and fixes Python code"""
    
    DEFAULT_INITIAL_PROMPT = """You are a Python coding expert. Create clean, working Python code for this request:

{context}

Requirements:
- Write complete, executable Python code
- Include proper error handling
- Add docstrings and comments
- Follow PEP 8 standards
- Make it production-ready

Return ONLY the Python code, no explanations."""

    DEFAULT_FIX_PROMPT = """You are a Python coding expert. Fix the existing code based on the feedback:

{context}

Requirements:
- Fix any execution errors
- Address prose feedback if provided
- Maintain existing functionality
- Follow PEP 8 standards
- Return ONLY the corrected Python code, no explanations."""
    
    def __init__(self, model_factory: ModelFactory, 
                 initial_prompt: Optional[str] = None,
                 fix_prompt: Optional[str] = None):
        self.model = model_factory.get_model(AgentType.CODER)
        self.initial_prompt = initial_prompt or self.DEFAULT_INITIAL_PROMPT
        self.fix_prompt = fix_prompt or self.DEFAULT_FIX_PROMPT
    
    def __call__(self, state) -> dict:
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
        
        # Choose appropriate prompt
        if not state["current_code"]:
            prompt = self.initial_prompt.format(context=context)
        else:
            prompt = self.fix_prompt.format(context=context)
        
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
    """Runner agent that executes Python code in Docker containers"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.docker_client = docker.from_env()
    
    def __call__(self, state) -> dict:
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
    """Prose agent that reviews code for style and best practices"""
    
    DEFAULT_REVIEW_PROMPT = """You are a Python code reviewer that takes working code and refactors or formats it
according to best practices. Input is the final (or current) code solution, and output is a improved
version of the code (functionally identical but cleaner).

Review this code:

```python
{code}
```
Primary Objective
Refactor the provided code for clarity, style, and maintainability without changing its core functionality or logic.
Core Responsibilities
Style and Formatting

PEP 8 Compliance: Ensure all code adheres to PEP 8 style guidelines
Import Organization: Organize imports following PEP 8 standards (standard library, third-party, local imports)
Line Length: Keep lines under 79 characters where practical
Whitespace: Use appropriate spacing around operators, after commas, etc.

Code Quality Improvements

Variable Naming: Replace unclear variable names with descriptive, meaningful names
Function Naming: Ensure function names clearly describe their purpose
Logic Simplification: Refactor overly complex logic into simpler, more readable forms
Redundancy Removal: Eliminate duplicate code, unused variables, and unnecessary complexity

Documentation

Docstrings: Add clear docstrings to functions, classes, and modules following PEP 257
Inline Comments: Add explanatory comments for complex logic or non-obvious code sections
Type Hints: Add type annotations where they improve clarity

Critical Boundaries

DO NOT alter the high-level logic or algorithmic approach
DO NOT add new features or functionality
DO NOT change the input/output behavior of functions
DO NOT modify the overall program structure or architecture

Output Requirements

Provide the refactored code
Include a brief summary of changes made
If significant changes were made, provide a diff or highlight the key modifications
Confirm that functionality remains unchanged

Quality Checklist
Before finalizing, verify:

 Code follows PEP 8 guidelines
 Variable and function names are descriptive
 Complex logic is simplified where possible
 Appropriate comments and docstrings are added
 No redundant code remains
 Original functionality is preserved
 Code is more readable and maintainable than the original
"""
    
    def __init__(self, model_factory: ModelFactory, 
                 review_prompt: Optional[str] = None):
        self.model = model_factory.get_model(AgentType.PROSE)
        self.review_prompt = review_prompt or self.DEFAULT_REVIEW_PROMPT
    
    def __call__(self, state) -> dict:
        """Prose agent reviews code for style and best practices"""
        print("📝 Prose Agent Reviewing Code...")
        
        if not state["current_code"]:
            print("   ❌ No code to review")
            return state
        
        prompt = self.review_prompt.format(code=state["current_code"])
        
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