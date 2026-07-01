import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .tools import default_tools


load_dotenv()

# Default OpenAI chat model. Can be overridden via the OPENAI_MODEL env var.
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")


class ServiceManager:
    """Central manager that owns the LLM and the tool set.

    It is the single source of truth for the "hot update" feature: both the model
    and the tools can be swapped at runtime without restarting the process.
    """

    def __init__(self):
        print("Initializing LLM and tools...")
        self._model_name = DEFAULT_MODEL
        self._llm = self._create_llm(self._model_name)
        self._tools = default_tools
        print("ServiceManager initialized.")
        self.print_services()

    @staticmethod
    def _create_llm(model_name: str) -> ChatOpenAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Placeholder so the object can be constructed (e.g. during tests).
            # Real requests will still fail until a valid key is provided.
            print("Warning: OPENAI_API_KEY environment variable is not set!")
            api_key = "sk-not-set"
        # base_url is optional; accept both OPENAI_BASE_URL and OPENAI_API_BASE.
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        kwargs = {"model": model_name, "temperature": 0, "streaming": True, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    def get_llm(self) -> ChatOpenAI:
        return self._llm

    def get_tools(self) -> list:
        return self._tools

    def update_llm(self, model_name: str) -> None:
        """Hot-swap the LLM to a different OpenAI model."""
        print(f"[Hot Update] Updating LLM model to: {model_name}")
        self._model_name = model_name
        self._llm = self._create_llm(model_name)
        self.print_services()

    def update_tools(self, new_tools: list) -> None:
        """Hot-swap the available tool set."""
        print("[Hot Update] Updating the tool list...")
        self._tools = new_tools
        self.print_services()

    def print_services(self) -> None:
        print("--- Current service status ---")
        print(f"  Model: {self._model_name}")
        print(f"  Tools: {[tool.name for tool in self._tools]}")
        print("------------------------------")

    def get_services_status(self) -> dict:
        return {
            "model": self._model_name,
            "tools": [tool.name for tool in self._tools],
        }


# Module-level singleton shared across the application.
service_manager = ServiceManager()
