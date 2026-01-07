from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS so the frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        user_message = request.message
        response = client.responses.create(
            model="gpt-5-nano",
            instructions="You are StudyBuddy, a helpful AI tutoring assistant. Your job is to help students learn by:\n\n- Explaining concepts clearly and at the right level for the student\n- Breaking down complex ideas into simpler pieces\n- Providing examples to illustrate your explanations\n- Encouraging questions and curiosity\n- Being patient and supportive\n\nWhen a student asks you something:\n1. First, understand what they're trying to learn\n2. Explain the concept in clear, simple language\n3. Use concrete examples when possible\n4. Check if they understood by asking if they need clarification\n\nKeep your explanations concise but thorough. If a concept is complicated, break it into smaller parts. Always be encouraging and make learning feel approachable, not intimidating.",
            input=user_message
        )
        return {"reply": response.output_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")
