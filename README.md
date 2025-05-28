# IngenuitySmith
An agentic framework for python development that will ultimately build/finetune DNN and optimize hyperparameters. For completeness, the project with use wsl, docker, and CUDA.

I am persuing a Master's in AI at UT Austin and there is not currently an agentic class. This effort is to supplement my education in these critical technologies.

The name IngenuitySmith is a nod to the Ingenuity helicopter and the inventive facet of ingenuity plus the builder aspect of smithing.

# v1
This is a LangGraph based agent system that genrgenerates > validates > optimizes (overly verbose) code

## File Structure:

**infrastructure.py** - Deep infrastructure (most stable)

<ul>
<li>All core imports and dependencies
<li>Configuration loading
<li>Model factory for different LLM providers
<li>Routing function
<li>AgentType enum</ul>


**agents.py** - Agent definitions (semi-fixed)

<ul>
<li>All agent implementations
<li>Enhanced: CoderAgent and ProseAgent now accept custom prompts as parameters
<li>Maintains default prompts for backward compatibility
<li>Clean separation of agent logic</ul>


**instantiator.py** - Framework instantiation (most flexible)

<ul>
<li>AgentState definition
<li>Main AgenticDevelopmentFramework class
<li>Enhanced: Framework constructor accepts custom prompts
<li>Main execution logic and usage examples</ul>