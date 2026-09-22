# Package src.generation
from src.generation.llm_client import GeminiLegalGenerator, GenerationResult
from src.generation.prompts import LEGAL_SYSTEM_PROMPT, build_user_prompt

__all__ = [
    "GeminiLegalGenerator",
    "GenerationResult",
    "LEGAL_SYSTEM_PROMPT",
    "build_user_prompt"
]
