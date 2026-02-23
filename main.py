from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.getenv("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDM3NTZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.ZDq34qobOlVytVxsS6mxprtpXTeaxxCS4ApCyCHOQzY"))

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"""
            Analyze the sentiment of this comment.
            Return JSON only.

            Comment: {request.comment}
            """,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        # Extract JSON safely
        output_text = response.output[0].content[0].text
        result = json.loads(output_text)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
