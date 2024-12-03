import logging
from typing import List, Union, Dict, Any
from pydantic import BaseModel
import anthropic
import openai
import os
import dotenv
import requests
import json

dotenv.load_dotenv()

log = logging.getLogger(__name__)

class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    pass

class RateLimitError(LLMError):
    pass

class APIConnectionError(LLMError):
    pass

class APIStatusError(LLMError):
    def __init__(self, status_code: int, response: Dict[str, Any]):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Received non-200 status code: {status_code}")

class LLM:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.prev_prompt: Union[str, None] = None
        self.prev_response: Union[str, None] = None

    def _validate_response(self, response_text: str, response_model: BaseModel) -> BaseModel:
        try:
            #input(response_text)
            if response_text.startswith("```") and response_text.endswith("```"):
                # Extract content between code blocks
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])  # Skip first and last lines with ```

            # First try to parse as JSON to ensure valid format
            response_json = json.loads(response_text)

            # Convert boolean values to strings
            def convert_bools(obj):
                if isinstance(obj, dict):
                    return {k: convert_bools(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_bools(item) for item in obj]
                elif isinstance(obj, bool):
                    return str(obj).title()  # Convert True/False to "True"/"False"
                return obj

            response_json = convert_bools(response_json)

            # Convert back to JSON string with proper boolean formatting
            response_text = json.dumps(response_json)

            return response_model.model_validate_json(response_text)
        except Exception as e:
            log.warning("Response validation failed", exc_info=e)
            raise LLMError(f"Validation failed: {str(e)}") from e

    def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def chat(self, user_prompt: str, response_model: BaseModel = None, max_tokens: int = 4096) -> Union[BaseModel, str]:
        self._add_to_history("user", user_prompt)
        messages = self.create_messages(user_prompt)
        response = self.send_message(messages, max_tokens, response_model)
        response_text = self.get_response(response)
        
        if response_model:
            response_text = self._validate_response(response_text, response_model)
        
        self._add_to_history("assistant", str(response_text))
        return response_text

class Claude(LLM):
    def __init__(self, model: str = "claude-3-opus-20240229", system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = anthropic.Anthropic()
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": user_prompt}]

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        try:
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages
            )
        except anthropic.APIConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError("Request was rate-limited") from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        return response.content[0].text

class ChatGPT(LLM):
    def __init__(self, model: str, base_url: str, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = model

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}]
        if self.model in ["o1-preview"]:
            messages = [{"role": "user", "content": user_prompt}]

        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model=None) -> Dict[str, Any]:
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            if self.model in ["o1-preview"]:
                del params["max_tokens"]

            # Add response format configuration if a model is provided
            if response_model:
                params["response_format"] = {
                    "type": "json_object"
                }

            return self.client.chat.completions.create(**params)
        except openai.APIConnectionError as e:
            raise APIConnectionError("The server could not be reached") from e
        except openai.RateLimitError as e:
            raise RateLimitError("Request was rate-limited; consider backing off") from e
        except openai.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e
        except Exception as e:
            raise LLMError(f"An unexpected error occurred: {str(e)}") from e

    def get_response(self, response: Dict[str, Any]) -> str:
        response = response.choices[0].message.content
        return response

class Ollama(LLM):
    def __init__(self, model: str = "llama2", system_prompt: str = "", base_url: str = "http://localhost:11434") -> None:
        super().__init__(system_prompt)
        self.model = model
        self.base_url = base_url

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                }
            )
            if response.status_code != 200:
                raise APIStatusError(response.status_code, response.json())
            return response.json()
        except requests.ConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e

    def get_response(self, response: Dict[str, Any]) -> str:
        return response.get("response", "")

def create_llm(llm_type: str = "claude", **kwargs) -> LLM:
    """Factory function to create LLM instances."""
    llm_classes = {
        "claude": Claude,
        "gpt": GPT,
        "ollama": Ollama
    }
    
    if llm_type not in llm_classes:
        raise ValueError(f"Unknown LLM type: {llm_type}. Available types: {list(llm_classes.keys())}")
    
    return llm_classes[llm_type](**kwargs)

def initialize_llm(llm_arg: str, system_prompt: str = "") -> Claude | ChatGPT | Ollama:
    llm_arg = llm_arg.lower()
    if llm_arg == 'claude':
        anth_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        anth_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        llm = Claude(anth_model, anth_base_url, system_prompt)
    elif llm_arg == 'gpt':
        openai_model = os.getenv("OPENAI_MODEL", "chatgpt-4o-latest")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        llm = ChatGPT(openai_model, openai_base_url, system_prompt)
    elif llm_arg == 'ollama':
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api/generate")
        llm = Ollama(ollama_model, ollama_base_url, system_prompt)
    else:
        raise ValueError(f"Invalid LLM argument: {llm_arg}\nValid options are: claude, gpt, ollama")
    return llm
