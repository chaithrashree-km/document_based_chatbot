from pydantic import BaseModel
from typing import Optional

class Chat(BaseModel):
    question: str