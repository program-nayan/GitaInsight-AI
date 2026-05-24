import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from src.logger import logging as logger
from src.components.parse_geeta import GitaIngestor
from src.components.vectorization_and_chunking import GitaHybridRetriever
from src.components.hybrid_search_engine import GitaSearchPipeline

def main_execution_pipeline(user_scenario: str):
    logger.info("=== STARTING MASTER EXECUTION PIPELINE ===")
    

    # 1. Environment & API Setup

    load_dotenv()
    GEMINI_API = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API:
        logger.error("API Key missing! Please set GEMINI_API_KEY in your .env file.")
        sys.exit(1)
        
    client = genai.Client(api_key=GEMINI_API)
    json_path = "./data/gita_structured.json"
    
    
    # 2. Ingestion Check (Build-Time logic)
    
    if not Path(json_path).exists():
        logger.warning(f"Structured JSON not found at {json_path}. Triggering Ingestor Pipeline...")
        ingestor = GitaIngestor()
        ingestor.run_ingestion()
    else:
        logger.info("Structured JSON dataset found. Skipping ingestion phase.")

    
    # 3. Booting the Search Engines (Run-Time logic)
    
    logger.info("Booting Dual-Engine Database (ChromaDB + BM25)...")
    retriever_db = GitaHybridRetriever(json_data_path=json_path)
    
    logger.info("Initializing Hybrid Search Pipeline...")
    search_engine = GitaSearchPipeline(retriever_db)
    
    
    # 4. Executing the Search
    
    logger.info(f"Searching for verses relevant to: '{user_scenario}'")
    top_verses = search_engine.execute_search(user_query=user_scenario, top_k=2)
    
    if not top_verses:
        logger.warning("No verses found.")
        return
        
    
    # 5. Formatting Context for the Gemma 31B Summarizer
    
    logger.info("Formatting context for the final summarization generation...")
    context_string = ""
    for v in top_verses:
        context_string += f"\n- Chapter: {v['chapter_name']}, Verse: {v['verse']}\n"
        context_string += f"  Sanskrit: {v['sanskrit']}\n"
        context_string += f"  Translation: {v['translation']}\n"
        
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
    
    prompt = f"User's Scenario: {user_scenario}\n\nRetrieved Gita Verses:{context_string}"
    
    
    # 6. Generate the Final Summary
    
    logger.info("Generating final spiritual summary...")

    SUMMARY_MODEL_NAME = "models/gemma-4-31b-it" 
    
    try:
        response = client.models.generate_content(
            model=SUMMARY_MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.6 
            )
        )
        final_summary = response.text.strip()
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        final_summary = "Error generating summary."

    
    # 7. Print Output to Terminal
    
    print("\n" + "="*80)
    print("👤 USER SCENARIO:")
    print(user_scenario)
    print("-" * 80)
    
    print("📜 RETRIEVED SHLOKAS (RRF Ranked):")
    for v in top_verses:
        print(f"[{v['chapter_name']} - Verse {v['verse']}]")
        print(f"{v['sanskrit']}")
        print(f"Meaning: {v['translation']}\n")
    print("-" * 80)
    
    print("✨ SPIRITUAL GUIDANCE (LLM Output):")
    print(final_summary)
    print("="*80 + "\n")
    
    return {
        "summary": final_summary,
        "shlokas": top_verses
    }

if __name__ == "__main__":
    # Test your completed backend pipeline!
    sample_scenario = "I am working 14 hours a day to get a promotion, but my colleague just got it instead. I feel betrayed, completely broken, and I want to quit my job."
    
    main_execution_pipeline(sample_scenario)