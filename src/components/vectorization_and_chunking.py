import json
import os
import sys
import string
from src.logger import logging as logger
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


class GitaHybridRetriever:
    def __init__(self, json_data_path, chunk_size=350, overlap=50, collection_name="gita_dense_vectors"):
        """
        Instantiates and automatically builds the dual-engine search pipeline.
        """
        self.json_data_path = json_data_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.collection_name = collection_name
        
        self.documents = []   
        self.metadata = []    
        self.bm25_engine = None
        self.vector_collection = None
        
        logger.info("Initializing GitaHybridRetriever pipeline...")
        
        # Automatically trigger the internal build sequence
        raw_verses = self._load_json_data()
        self._build_chunked_corpus(raw_verses)
        self._init_vector_db()
        self._init_bm25()
        
        logger.info("Hybrid Retriever is fully armed and operational.")

    def _load_json_data(self):
        """[Internal] Reads the structured JSON file from disk."""
        if not os.path.exists(self.json_data_path):
            logger.error(f"Cannot find JSON data at {self.json_data_path}")
            raise FileNotFoundError(f"Missing file: {self.json_data_path}")
            
        with open(self.json_data_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            
        logger.info(f"Successfully loaded {len(raw_data)} verses from JSON.")
        return raw_data

    def _build_chunked_corpus(self, raw_verses):
        """[Internal] Slices verses into Translation-Anchored overlapping windows."""
        logger.info("Chunking text using Translation-Anchored strategy...")
        
        for verse_obj in raw_verses:
            translation = verse_obj.get("translation", "")
            purport = verse_obj.get("purport", "")
            
            base_metadata = {
                "chapter": verse_obj.get("chapter", ""),
                "chapter_name": verse_obj.get("chapter_name", ""),
                "verse": verse_obj.get("verse", ""),
                "sanskrit": verse_obj.get("sanskrit", ""),
                "translation": translation
            }
            
            if not purport:
                text_to_embed = f"TRANSLATION: {translation}"
                self.documents.append(text_to_embed)
                self.metadata.append(base_metadata)
                continue
                
            words = purport.split()
            
            if len(words) <= self.chunk_size:
                text_to_embed = f"TRANSLATION: {translation}\n\nPURPORT: {purport}"
                self.documents.append(text_to_embed)
                self.metadata.append(base_metadata)
            else:
                for i in range(0, len(words), self.chunk_size - self.overlap):
                    chunk_words = words[i : i + self.chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    text_to_embed = f"TRANSLATION: {translation}\n\nPURPORT: {chunk_text}"
                    self.documents.append(text_to_embed)
                    self.metadata.append(base_metadata)
                    
        logger.info(f"Created {len(self.documents)} total searchable chunks.")

    def _init_vector_db(self):
        """[Internal] Initializes ChromaDB semantic search engine."""
        logger.info("Booting up ChromaDB vector storage...")
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=emb_fn
        )
        
        existing_count = self.vector_collection.count()
        if existing_count == 0:
            logger.info("Vector DB empty. Encoding documents (this may take a minute)...")
            ids = [f"chunk_{i}" for i in range(len(self.documents))]
            self.vector_collection.add(
                documents=self.documents,
                metadatas=self.metadata,
                ids=ids
            )
            logger.info(f"Successfully embedded {len(self.documents)} chunks.")
        else:
            logger.info(f"Vector DB already contains {existing_count} embedded chunks. Skipping insertion.")

    def _init_bm25(self):
        """[Internal] Initializes BM25 exact-keyword search engine."""
        logger.info("Booting up BM25 Sparse Keyword search engine...")
        
        def tokenize(text):
            clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))
            return clean_text.split()
            
        tokenized_corpus = [tokenize(doc) for doc in self.documents]
        self.bm25_engine = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 Engine indexed {len(tokenized_corpus)} chunks.")


# if __name__ == "__main__":

#     retriever = GitaHybridRetriever(json_data_path="./data/gita_structured.json")