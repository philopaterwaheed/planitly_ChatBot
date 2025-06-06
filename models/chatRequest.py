
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from models import User
# Pydantic model for chat request
class ChatRequest(BaseModel):
    message: str
    user: Dict[str,Any]
    ai_subject_ids: Optional[List[str]] = None

