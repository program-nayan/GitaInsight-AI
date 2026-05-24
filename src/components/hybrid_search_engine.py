import string
import numpy as np
from src.logger import logging as logger
import sys
import os
from dotenv import load_dotenv
from google import genai
from src.exception import CustomException
from src.components.vectorization_and_chunking import GitaHybridRetriever

class GitaSearchPipeline:
    def __init__(self, retriever_db):
        """
        Takes an already initialized GitaHybridRetriever database instance.
        """
        self.db = retriever_db

        load_dotenv()
        GEMINI_API = os.getenv("GEMINI_API_KEY")
        self.MODEL_NAME = "models/gemma-4-26b-a4b-it"
        self.client = genai.Client(api_key= GEMINI_API)
        logger.info("Gita Search Pipeline initialized and linked to Hybrid DB.")

    def _rewrite_query(self, user_query):
        """
        Sends the user's raw query to Gemma 4 26B to extract philosophical themes.
        """
        logger.info(f"Raw User Query: '{user_query}'")
        
        system_prompt = """
        You are a semantic query expander. Analyze the user's modern problem. 
        Extract the core emotion and translate it into Bhagavad Gita philosophical themes 
        (e.g., duty, detachment, illusion, anger, lamentation, karma,etc). 
        Return ONLY a single string of comma-separated keywords. Do not write sentences.
        """

        try:
            # CORRECT GOOGLE GENAI SYNTAX
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=f"User Problem: {user_query}",
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2 # Low temperature for strict keyword extraction
                )
            )
            expanded_query = response.text.strip()
        except Exception as e:
            logger.error(f"LLM Rewriting failed. Error: {str(e)}")
            expanded_query = ""
            raise CustomException(e, sys)

        
        if not expanded_query:
            expanded_query = "anger, frustration with authority, abandon prescribed duties, attachment to results"
        
        logger.info(f"Gemma 26B Expanded Query: '{expanded_query}'")
        return expanded_query

    def _calculate_rrf(self, vector_indices, bm25_indices, rrf_k=60):
        """
        Applies Reciprocal Rank Fusion to merge and rank results from both engines.
        """
        rrf_scores = {}
        
        def apply_rrf_to_list(ranked_list):
            for rank, doc_idx in enumerate(ranked_list):
                # RRF Formula: 1 / (k + rank) -> Rank is 0-indexed, so we add 1.
                score = 1.0 / (rrf_k + rank + 1)
                
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = 0.0
                rrf_scores[doc_idx] += score

        # Apply math to both result sets
        apply_rrf_to_list(vector_indices)
        apply_rrf_to_list(bm25_indices)
        
        # Sort documents by their combined RRF score in descending order
        sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_rrf

    def execute_search(self, user_query, top_k=3, rrf_k=60):
        """
        The main public method to execute the Hybrid Search.
        """
        # 1. Rewrite the query
        expanded_query = self._rewrite_query(user_query)
        combined_query = user_query + " " + expanded_query
        
        # We fetch a larger pool from the DBs to ensure good overlap for RRF
        pool_size = top_k * 3 

        # 2. Query the Dense Vector Database (ChromaDB)
        vector_results = self.db.vector_collection.query(
            query_texts=[expanded_query],
            n_results=pool_size
        )
        
        vector_ranked_indices = []
        if vector_results['ids'] and len(vector_results['ids']) > 0:
            for chunk_id in vector_results['ids'][0]:
                idx = int(chunk_id.split('_')[1])
                vector_ranked_indices.append(idx)

        # 3. Query the Sparse Keyword Database (BM25)
        clean_query = combined_query.lower().translate(str.maketrans('', '', string.punctuation))
        tokenized_query = clean_query.split()
        
        bm25_raw_scores = self.db.bm25_engine.get_scores(tokenized_query)
        bm25_ranked_indices = np.argsort(bm25_raw_scores)[::-1][:pool_size].tolist()

        # 4. Fuse the results using RRF
        sorted_rrf_results = self._calculate_rrf(vector_ranked_indices, bm25_ranked_indices, rrf_k)

        # 5. Build the clean final payload
        final_results = []
        logger.info(f"\n--- TOP {top_k} HYBRID MATCHES ---")
        
        for rank, (doc_idx, score) in enumerate(sorted_rrf_results[:top_k]):
            meta = self.db.metadata[doc_idx]
            logger.info(f"Rank {rank+1} | RRF Score: {score:.4f} | Chapter {meta['chapter']} Verse {meta['verse']}")
            
            final_results.append({
                "chapter_name": meta['chapter_name'],
                "verse": meta['verse'],
                "sanskrit": meta['sanskrit'],
                "translation": meta['translation'],
                "rrf_score": score
            })
            
        return final_results
    
# if __name__ == "__main__":
#     # 1. Boot up the Storage Engine (Your previous class)
#     # This automatically loads JSON, chunks it, and builds Chroma & BM25
#     retriever_db = GitaHybridRetriever(json_data_path="./data/gita_structured.json")
    
#     # 2. Boot up the Search Engine (The new class)
#     search_pipeline = GitaSearchPipeline(retriever_db)
    
#     # 3. Execute a search!
#     user_problem = "I hate my boss and I want to quit my job."
#     results = search_pipeline.execute_search(user_query=user_problem, top_k=3)