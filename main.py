import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from src.logger import logging as logger

# Import your custom AI engines
from src.components.vectorization_and_chunking import GitaHybridRetriever
from src.components.hybrid_search_engine import GitaSearchPipeline


# 1. Initialize FastAPI & Globals

app = FastAPI(title="Gita Insight AI", version="1.0")

db_retriever = None
search_pipeline = None
llm_client = None


# 2. Server Startup Sequence

@app.on_event("startup")
async def startup_event():
    global db_retriever, search_pipeline, llm_client
    
    load_dotenv()
    GEMINI_API = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables!")
        
    llm_client = genai.Client(api_key=GEMINI_API)

    logger.info("Booting up Vector and BM25 Databases...")
    db_retriever = GitaHybridRetriever(json_data_path="./data/gita_structured.json")
    
    logger.info("Initializing Search Pipeline...")
    search_pipeline = GitaSearchPipeline(db_retriever)
    
    logger.info("Server is armed and ready!")


# 3. Pydantic Data Models

class ChatRequest(BaseModel):
    scenario: str

class VerseDetail(BaseModel):
    chapter_name: str
    verse: str
    sanskrit: str
    translation: str

class ChatResponse(BaseModel):
    spiritual_summary: str
    shlokas: list[VerseDetail]


# 4. The Core AI API Endpoint

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_gita(request: ChatRequest):
    try:
        # A. Execute the Hybrid Search
        top_verses = search_pipeline.execute_search(user_query=request.scenario, top_k=2)
        
        if not top_verses:
            raise HTTPException(status_code=404, detail="Could not find relevant verses.")

        # B. Format context for the Summarizer
        context_string = ""
        for v in top_verses:
            context_string += f"\n- Chapter: {v['chapter_name']}, Verse: {v['verse']}\n"
            context_string += f"  Sanskrit: {v['sanskrit']}\n"
            context_string += f"  Translation: {v['translation']}\n"

        # C. Define the AI Persona
        system_instruction = """
        You are a highly empathetic, wise spiritual guide grounded in the teachings of the Bhagavad Gita. 
        The user will present a modern life scenario or problem.
        You will also be provided with specific verses retrieved from the Bhagavad Gita.
        
        Your task:
        Synthesize these specific verses into a clear, comforting, and deeply philosophical 
        summary that addresses the user's exact problem. 
        Speak directly to the user's pain or confusion. Do not sound like an academic textbook.
        Reference the provided verses naturally in your advice. Keep it under 3 paragraphs.
        """
        
        prompt = f"User's Scenario: {request.scenario}\n\nRetrieved Gita Verses:{context_string}"

        # D. Execute Summarization using Gemma 4 31B
        SUMMARY_MODEL_NAME = "models/gemma-4-31b-it" 
        
        response = llm_client.models.generate_content(
            model=SUMMARY_MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.6 
            )
        )
        
        # E. Return the JSON Payload
        return ChatResponse(
            spiritual_summary=response.text.strip(),
            shlokas=[VerseDetail(**v) for v in top_verses]
        )

    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")


# 5. Serve the Frontend Webpage

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


# 6. Run the Server

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)