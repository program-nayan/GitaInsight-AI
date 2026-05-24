import os
import sys
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from google import genai
from src.logger import logging as logger
import time

# Import your custom AI engines
from src.components.vectorization_and_chunking import GitaHybridRetriever
from src.components.hybrid_search_engine import GitaSearchPipeline


# 1. Initialize Flask Application

app = Flask(__name__)


# 2. Boot the Databases (Global Memory)

logger.info("Booting up Vector and BM25 Databases...")
load_dotenv()
GEMINI_API = os.getenv("GEMINI_API_KEY")
if not GEMINI_API:
    logger.error("GEMINI_API_KEY not found in environment variables!")
    sys.exit(1)

llm_client = genai.Client(api_key=GEMINI_API)
db_retriever = GitaHybridRetriever(json_data_path="./data/gita_structured.json")
search_pipeline = GitaSearchPipeline(db_retriever)
logger.info("Flask Server is armed and ready!")


# 3. Serve the Frontend Webpage

@app.route('/')
def home():
    """Serves the index.html file when you open the browser."""
    return send_file('index.html')


# 4. The Core AI API Endpoint

@app.route('/api/chat', methods=['POST'])
def chat_with_gita():
    """Receives the frontend request, searches the Gita, and returns the LLM summary."""
    try:
        # Extract the user's message from the incoming JSON
        data = request.get_json()
        user_scenario = data.get('scenario', '')
        
        if not user_scenario:
            return jsonify({"error": "No scenario provided"}), 400

        # A. Execute the Hybrid Search
        top_verses = search_pipeline.execute_search(user_query=user_scenario, top_k=2)
        
        if not top_verses:
            return jsonify({"error": "Could not find relevant verses."}), 404

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
        
        prompt = f"User's Scenario: {user_scenario}\n\nRetrieved Gita Verses:{context_string}"

        # D. Execute Summarization using Gemma 4 31B
        SUMMARY_MODEL_NAME = "models/gemma-4-31b-it" 
        for attempt in range(3):
            try:
                response = llm_client.models.generate_content(
                    model=SUMMARY_MODEL_NAME,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.6 
                    )
                )
                summary_text = response.text.strip()
                break # Success! Break out of the loop
                
            except Exception as e:
                if "500" in str(e) and attempt < 2:
                    logger.warning(f"Summarizer server busy (500). Retrying... (Attempt {attempt + 1}/3)")
                    time.sleep(2)
                else:
                    logger.error(f"Summarization failed: {str(e)}")
                    return jsonify({"error": "Google's AI servers are currently overloaded. Please try again in a moment."}), 500
                
        
        # E. Return the JSON Payload
        return jsonify({
            "spiritual_summary": response.text.strip(),
            "shlokas": top_verses
        })

    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({"error": "Internal Server Error during inference."}), 500


# 5. Run the Server

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)