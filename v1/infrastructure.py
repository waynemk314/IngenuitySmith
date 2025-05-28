"""
Deep Infrastructure - Most stable components
Contains core imports, configuration, routing, and model factory
"""

from typing import Dict, Any
import os
import docker
from dotenv import load_dotenv
from enum import Enum

# LangGraph imports
from langgraph.graph import END
from langgraph.graph.message import add_messages

# Model imports
try:
    from langchain_ollama import ChatOllama  # Updated import
except ImportError:
    from langchain_community.chat_models import ChatOllama  # Fallback
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Agent Type Enum (needed for ModelFactory)
class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    CODER = "coder"
    RUNNER = "runner" 
    PROSE = "prose"

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

def route_next_agent(state) -> str:
    """Route to next agent based on orchestrator decision"""
    next_agent = state.get("next_agent", "end")
    if next_agent == "end":
        return END
    return next_agent

class ModelFactory:
    """Factory class for creating appropriate models for different agent types"""
    
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