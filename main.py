import os
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Initialize FastAPI
app = FastAPI()

# 2. FIX: Enable CORS so the grader website can reach your Render URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Configure AI Pipe Client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), # Your ey... token from Render Env
    base_url="https://aipipe.org/openai/v1"   # The AI Pipe gateway
)

# 4. Define Data Models for Structured Output
class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    # Enforce exactly these three strings
    sentiment: Literal["positive", "negative", "neutral"]
    # Enforce an integer between 1 and 5
    rating: int = Field(ge=1, le=5)

# 5. The Analysis Endpoint
@app.post("/comment")
def analyze_sentiment(request: CommentRequest):
    try:
        # Use .parse for guaranteed Structured JSON Output
        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-mini", # Specific model for this course
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a strict sentiment analyzer. Scale your ratings precisely:\n"
                        "5: ONLY for superlatives (e.g., 'Amazing', 'Perfect', 'Incredible', 'Best').\n"
                        "4: For general positive feedback (e.g., 'Solid', 'Good', 'Satisfied', 'Well-structured').\n"
                        "3: For neutral or average feedback.\n"
                        "2: For negative or disappointing feedback.\n"
                        "1: For highly negative or angry feedback.\n\n"
                        "IMPORTANT: If the user is just 'satisfied' or says it's 'pretty good', give a 4, NOT a 5."
                    )
                },
                {"role": "user", "content": request.comment}
            ],
            response_format=SentimentResponse,
        )
        
        # Extract the structured JSON data
        return completion.choices[0].message.parsed

    except Exception as e:
        # Return the error message to help you debug in Render logs
        raise HTTPException(status_code=500, detail=str(e))
