from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
import os

app = FastAPI()

# --- ADD THIS CORS SECTION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows the grader to reach your API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------

# Use your AI Pipe token and the correct base URL
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1" 
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(ge=1, le=5)

@app.post("/comment")
def analyze_comment(request: CommentRequest):
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-mini", 
            messages=[
                {"role": "system", "content": "Analyze sentiment and rating."},
                {"role": "user", "content": request.comment}
            ],
            response_format=SentimentResponse,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
