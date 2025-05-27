import os
import uuid
import time
from dotenv import load_dotenv
import docker # type: ignore
from pathlib import Path

# --- Langchain specific imports ---
# Ollama
from langchain_community.chat_models import ChatOllama

# OpenAI
from langchain_openai import ChatOpenAI

# Anthropic
from langchain_anthropic import ChatAnthropic

# --- Configuration Loading ---
def load_config():
    """Loads configuration from .env file and returns it as a dictionary."""
    load_dotenv()
    config = {
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
    print("--- Configuration Loaded ---")
    for key, value in config.items():
        print(f"{key}: {'SET' if value else 'NOT SET'}")
    print("----------------------------\n")
    return config

# --- Validation Functions ---

def validate_ollama(base_url, coder_model_name, prose_model_name):
    print("--- Validating Ollama ---")
    if not base_url:
        print("❌ OLLAMA_BASE_URL not set. Skipping Ollama validation.")
        return False
    if not coder_model_name:
        print("⚠️ OLLAMA_CODER_MODEL_NAME not set. Cannot test coder model.")
    if not prose_model_name:
        print("⚠️ OLLAMA_PROSE_MODEL_NAME not set. Cannot test prose model.")

    all_ok = True

    if coder_model_name:
        try:
            print(f"Attempting to connect to Ollama Coder model: {coder_model_name} at {base_url}")
            llm = ChatOllama(model=coder_model_name, base_url=base_url, temperature=0)
            response = llm.invoke("Briefly, what is Python?")
            print(f"✅ Ollama Coder ({coder_model_name}) responded (first 30 chars): {response.content[:30]}...")
        except Exception as e:
            print(f"❌ Error connecting to Ollama Coder model ({coder_model_name}): {e}")
            all_ok = False

    if prose_model_name:
        try:
            print(f"Attempting to connect to Ollama Prose model: {prose_model_name} at {base_url}")
            llm = ChatOllama(model=prose_model_name, base_url=base_url, temperature=0)
            response = llm.invoke("Briefly, what is a language model?")
            print(f"✅ Ollama Prose ({prose_model_name}) responded (first 30 chars): {response.content[:30]}...")
        except Exception as e:
            print(f"❌ Error connecting to Ollama Prose model ({prose_model_name}): {e}")
            all_ok = False
    
    if all_ok and coder_model_name and prose_model_name:
        print("✅ Ollama validation successful (for specified models).")
    elif all_ok and (coder_model_name or prose_model_name):
        print("🆗 Ollama validation partially successful (for specified models).")
    else:
        print("❌ Ollama validation failed or incomplete.")
    print("-------------------------\n")
    return all_ok

def validate_openai(api_key, model_name="gpt-3.5-turbo"):
    print("--- Validating OpenAI ---")
    if not api_key:
        print("ℹ️ OPENAI_API_KEY not set. Skipping OpenAI validation.")
        return True # Not a failure, just skipped

    try:
        print(f"Attempting to connect to OpenAI model: {model_name}")
        llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, temperature=0)
        response = llm.invoke("Briefly, what is OpenAI?")
        print(f"✅ OpenAI ({model_name}) responded (first 30 chars): {response.content[:30]}...")
        print("✅ OpenAI validation successful.")
    except Exception as e:
        print(f"❌ Error connecting to OpenAI ({model_name}): {e}")
        print("❌ OpenAI validation failed.")
        return False
    print("-----------------------\n")
    return True

def validate_anthropic(api_key, model_name="claude-3-haiku-20240307"):
    print("--- Validating Anthropic ---")
    if not api_key:
        print("ℹ️ ANTHROPIC_API_KEY not set. Skipping Anthropic validation.")
        return True # Not a failure, just skipped

    try:
        print(f"Attempting to connect to Anthropic model: {model_name}")
        llm = ChatAnthropic(model_name=model_name, anthropic_api_key=api_key, temperature=0)
        response = llm.invoke("Briefly, what is Anthropic?")
        print(f"✅ Anthropic ({model_name}) responded (first 30 chars): {response.content[:30]}...")
        print("✅ Anthropic validation successful.")
    except Exception as e:
        print(f"❌ Error connecting to Anthropic ({model_name}): {e}")
        print("❌ Anthropic validation failed.")
        return False
    print("--------------------------\n")
    return True

def validate_docker_runner(image_name, host_script_dir_str, container_script_dir_str):
    print("--- Validating Docker Runner ---")
    if not all([image_name, host_script_dir_str, container_script_dir_str]):
        print("❌ Missing Docker runner configuration (image, host dir, or container dir). Skipping.")
        return False

    host_script_dir = Path(host_script_dir_str)
    try:
        host_script_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Host script directory confirmed/created: {host_script_dir}")
    except Exception as e:
        print(f"❌ Could not create or access host script directory {host_script_dir}: {e}")
        print("   Please ensure the path is correct and you have write permissions from WSL2.")
        return False

    # Create a unique filename for the test script
    temp_script_name = f"test_docker_run_{uuid.uuid4().hex[:8]}.py"
    host_script_path = host_script_dir / temp_script_name
    container_script_path = Path(container_script_dir_str) / temp_script_name

    # Test script content
    script_content = (
        "import sys\n"
        "print('Hello from Docker Runner inside the container!')\n"
        "print('SUCCESS_MARKER_STDOUT')\n"
        "sys.stderr.write('This is a test error message to stderr.\\n')\n"
        "sys.stderr.write('ERROR_MARKER_STDERR\\n')\n"
        "sys.exit(0)\n" # Successful exit
    )

    try:
        with open(host_script_path, "w") as f:
            f.write(script_content)
        print(f"✅ Temporary script created: {host_script_path}")

        client = docker.from_env()
        print(f"Attempting to pull Docker image: {image_name} (if not present)...")
        try:
            client.images.pull(image_name)
            print(f"✅ Docker image {image_name} is available.")
        except docker.errors.APIError as e:
            print(f"❌ Failed to pull Docker image {image_name}: {e}")
            print("   Ensure Docker daemon is running and image name is correct.")
            return False


        print(f"Running script in Docker container. Image: {image_name}")
        print(f"Host path: {host_script_path} -> Container path: {container_script_path}")
        
        volumes_dict = {
            str(host_script_dir.resolve()): { # Use resolved absolute path
                'bind': container_script_dir_str,
                'mode': 'rw' # read-write, though read-only ('ro') would also work for just execution
            }
        }
        command_to_run = ["python3", str(container_script_path)]

        container = client.containers.run(
            image=image_name, 
            command=command_to_run,
            volumes=volumes_dict,
            remove=True, # Equivalent to --rm
            detach=False, # Run in foreground and get logs/status
            stdout=True,
            stderr=True
        )
        
        # In docker-py >5.0.0, 'container' is the logs if detach=False
        # For older versions, you'd do:
        # status_code = container.wait()['StatusCode']
        # stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
        # stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
        # For newer versions where run(detach=False) returns bytes:
        stdout = container.decode('utf-8') # Assuming output is utf-8
        stderr = "" # With detach=False, stdout and stderr are combined in the result or handled differently.
                       # To get them separately and the status code, it's often easier to run detached
                       # and then wait and get logs. Let's adjust for clarity:

        container_obj = client.containers.run(
            image=image_name,
            command=command_to_run,
            volumes=volumes_dict,
            detach=True # Detach to get container object
        )
        result = container_obj.wait() # Wait for completion
        status_code = result['StatusCode']
        stdout_bytes = container_obj.logs(stdout=True, stderr=False)
        stderr_bytes = container_obj.logs(stdout=False, stderr=True)
        container_obj.remove() # Clean up container

        stdout = stdout_bytes.decode('utf-8').strip()
        stderr = stderr_bytes.decode('utf-8').strip()

        print(f"Container Status Code: {status_code}")
        print(f"Container STDOUT:\n{stdout}")
        print(f"Container STDERR:\n{stderr}")

        if status_code == 0 and "SUCCESS_MARKER_STDOUT" in stdout and "ERROR_MARKER_STDERR" in stderr:
            print("✅ Docker Runner validation successful.")
            result_ok = True
        else:
            print("❌ Docker Runner validation failed. Check output and status code.")
            if "SUCCESS_MARKER_STDOUT" not in stdout:
                print("   - SUCCESS_MARKER_STDOUT not found in stdout.")
            if "ERROR_MARKER_STDERR" not in stderr:
                 print("   - ERROR_MARKER_STDERR not found in stderr.")
            if status_code != 0:
                print(f"   - Expected status code 0, got {status_code}.")
            result_ok = False
        
        return result_ok

    except docker.errors.DockerException as e:
        print(f"❌ Docker SDK error: {e}")
        print("   Ensure Docker daemon is running and accessible from WSL2.")
        print("   If using Docker Desktop, ensure WSL integration is enabled for your distro.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred during Docker validation: {e}")
        return False
    finally:
        if host_script_path.exists():
            try:
                host_script_path.unlink()
                print(f"✅ Temporary script deleted: {host_script_path}")
            except Exception as e:
                print(f"⚠️ Could not delete temporary script {host_script_path}: {e}")
        print("----------------------------\n")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Environment Validation Script...\n")
    cfg = load_config()

    # Validate individual components
    ollama_ok = validate_ollama(
        cfg["OLLAMA_BASE_URL"],
        cfg["OLLAMA_CODER_MODEL_NAME"],
        cfg["OLLAMA_PROSE_MODEL_NAME"]
    )

    openai_ok = validate_openai(
        cfg["OPENAI_API_KEY"],
        # You could use cfg["REMOTE_CODER_MODEL_NAME"] if provider is openai, etc.
        # For simplicity, using a default common model for validation.
        model_name="gpt-3.5-turbo" 
    )

    anthropic_ok = validate_anthropic(
        cfg["ANTHROPIC_API_KEY"],
        model_name="claude-3-haiku-20240307" # Common cheap model for testing
    )

    docker_ok = validate_docker_runner(
        cfg["RUNNER_DOCKER_IMAGE"],
        cfg["RUNNER_HOST_SCRIPT_DIR"],
        cfg["RUNNER_CONTAINER_SCRIPT_DIR"]
    )

    print("\n--- Validation Summary ---")
    print(f"Ollama: {'✅ PASSED' if ollama_ok else '❌ FAILED / SKIPPED'}")
    print(f"OpenAI: {'✅ PASSED' if openai_ok else '❌ FAILED / SKIPPED'}")
    print(f"Anthropic: {'✅ PASSED' if anthropic_ok else '❌ FAILED / SKIPPED'}")
    print(f"Docker Runner: {'✅ PASSED' if docker_ok else '❌ FAILED'}")
    print("--------------------------")

    if ollama_ok and openai_ok and anthropic_ok and docker_ok: # Adjust based on which ones are critical
        print("\n🎉 All critical components seem to be configured and working correctly!")
    else:
        print("\n⚠️ Some components failed validation. Please review the logs above.")