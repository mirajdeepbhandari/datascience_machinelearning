from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(...,description="Enter the news article text to classify.")