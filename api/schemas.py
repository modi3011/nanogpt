from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    max_new_tokens: int = Field(default=400, ge=10, le=800)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=200, ge=1, le=500)

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    model: str  # "stories" or "code"

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    status: str           # "ok" or "degraded"
    stories_loaded: bool
    code_loaded: bool