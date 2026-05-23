import pdfplumber
import re
import json
import os
import requests
from pathlib import Path    
import sys
from datetime import datetime
from src.logger import logging
from src.exception import CustomException

# Import the local Sanskrit transliteration engine tools
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:
    print("Please install the transliteration package: pip install indic_transliteration")
    sys.exit(1)


# Official chapter names from Bhagavad-gita As It Is
CHAPTER_NAMES = {
    "ONE": "Observing the Armies on the Battlefield of Kurukṣetra",
    "TWO": "Contents of the Gītā Summarized",
    "THREE": "Karma-yoga",
    "FOUR": "Transcendental Knowledge",
    "FIVE": "Karma-yoga—Action in Kṛṣṇa Consciousness",
    "SIX": "Dhyāna-yoga",
    "SEVEN": "Knowledge of the Absolute",
    "EIGHT": "Attaining the Supreme",
    "NINE": "The Most Confidential Knowledge",
    "TEN": "The Opulence of the Absolute",
    "ELEVEN": "The Universal Form",
    "TWELVE": "Devotional Service",
    "THIRTEEN": "Nature, the Enjoyer, and Consciousness",
    "FOURTEEN": "The Three Modes of Material Nature",
    "FIFTEEN": "The Yoga of the Supreme Person",
    "SIXTEEN": "The Divine and Demoniac Natures",
    "SEVENTEEN": "The Divisions of Faith",
    "EIGHTEEN": "Conclusion—The Perfection of Renunciation"
}

def convert_prabhupada_to_iast(text):
    """
    Global dictionary mapping legacy BBT/Balaram fonts to standard IAST Unicode.
    """
    if not text:
        return ""

    balaram_to_iast = {
        'ä': 'ā', 'é': 'ī', 'ü': 'ū', 'å': 'ṛ', 'è': 'ṝ',
        'õ': 'ḷ', 'ì': 'ṅ', 'ï': 'ñ', 'ö': 'ṭ', 'ò': 'ḍ',
        'ç': 'ś', 'ë': 'ṇ', 'à': 'ṁ', 'ñ': 'ṣ', 'ù': 'ḥ'
    }

    normalized = text.lower()
    for legacy_char, iast_char in balaram_to_iast.items():
        normalized = normalized.replace(legacy_char, iast_char)

    normalized = normalized.replace('-', ' ')
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def extract_raw_text(pdf_path):
    logging.info(f"Attempting to initialize raw I/O stream on path target: {pdf_path}")
    full_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            start_index = 39
            end_index = 910  # Extended to ensure Ch 18.78 is safely captured
            target_pages = pdf.pages[start_index:end_index]
            
            total_pages = len(target_pages)
            logging.info(f"Stream established. Constraining extraction from PDF Page 40 to 910. Total targets: {total_pages} pages.")
            
            for i, page in enumerate(target_pages):
                text = page.extract_text()
                if text:
                    full_text.append(text)
                
                if (i + 1) % 50 == 0 or (i + 1) == total_pages:
                    logging.info(f"Page extraction heartbeat: {i + 1}/{total_pages} chunks captured.")
                    
    except Exception as e:
        logging.error(f"Fatal crash reading structure of source file at step execution: {str(e)}")
        raise CustomException(e, sys)
        
    return "\n".join(full_text)

def parse_gita_text(raw_text):
    logging.info("Initializing line-by-line regex parsing token validation pass.")
    
    chapter_pattern = re.compile(r'CHAPTER\s+([A-Z\s\-]+|\d+)', re.IGNORECASE)
    
    # UPGRADED: Catches "TEXT 12 & 13", "TEXT 12, 13", and spaces around hyphens
    text_pattern = re.compile(r'^TEXTS?\s+(\d+[\s\-–&,]+\d*|\d+)', re.IGNORECASE)
    
    translation_pattern = re.compile(r'^TRANSLATION$', re.IGNORECASE)
    purport_pattern = re.compile(r'^PURPORT$', re.IGNORECASE)

    lines = raw_text.split('\n')
    all_verses = []
    current_chapter = "1" 
    
    current_verse = None
    state = "LOOKING_FOR_TEXT" 
    temp_buffer = []

    for line_idx, line in enumerate(lines):
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
            
        chap_match = chapter_pattern.match(cleaned_line)
        if chap_match:
            if current_verse and temp_buffer:
                if state == "PURPORT":
                    current_verse["purport"] = " ".join(temp_buffer)
                temp_buffer = []
                all_verses.append(current_verse)
                current_verse = None
            
            current_chapter = chap_match.group(1).strip().upper()
            logging.info(f"Encountered transition boundary -> Mapping Chapter: {current_chapter}")
            state = "LOOKING_FOR_TEXT"
            continue

        text_match = text_pattern.match(cleaned_line)
        if text_match:
            if current_verse:
                if state == "PURPORT":
                    current_verse["purport"] = " ".join(temp_buffer).strip()
                elif state == "TRANSLATION" and not current_verse["translation"]:
                    current_verse["translation"] = " ".join(temp_buffer).strip()
                all_verses.append(current_verse)
                
            # Retrieve the official name from our dictionary mapped dictionary
            mapped_chapter_name = CHAPTER_NAMES.get(current_chapter, "Unknown Chapter Name")

            current_verse = {
                "chapter": current_chapter,
                "chapter_name": mapped_chapter_name,
                "verse": text_match.group(1).strip(),
                "sanskrit": "",
                "transliteration": "",
                "synonyms": "",
                "translation": "",
                "purport": ""
            }
            temp_buffer = []
            state = "SANSKRIT_TRANSLIT"
            continue

        if translation_pattern.match(cleaned_line):
            if current_verse and state == "SANSKRIT_TRANSLIT":
                current_verse["transliteration"] = "\n".join(temp_buffer).strip()
            elif current_verse and state == "SYNONYMS":
                current_verse["synonyms"] = " ".join(temp_buffer).strip()
                
            temp_buffer = []
            state = "TRANSLATION"
            continue
            
        if purport_pattern.match(cleaned_line):
            if current_verse and state == "TRANSLATION":
                current_verse["translation"] = " ".join(temp_buffer).strip()
            temp_buffer = []
            state = "PURPORT"
            continue

        if state == "SANSKRIT_TRANSLIT":
            if "—" in cleaned_line or (";" in cleaned_line and not any(char.isdigit() for char in cleaned_line)):
                current_verse["transliteration"] = "\n".join(temp_buffer).strip()
                temp_buffer = [cleaned_line]
                state = "SYNONYMS"
            else:
                temp_buffer.append(cleaned_line)
        elif state in ["SYNONYMS", "TRANSLATION", "PURPORT"]:
            temp_buffer.append(cleaned_line)

    if current_verse:
        if state == "PURPORT":
            current_verse["purport"] = " ".join(temp_buffer).strip()
        all_verses.append(current_verse)

    # Local Transliteration to Devanagari Conversion Pass
    logging.info("Executing global Balaram -> IAST -> Devanagari Sanskrit conversion.")
    total_verses = len(all_verses)
    
    for idx, v in enumerate(all_verses):
        if v["transliteration"]:
            raw_translit = v["transliteration"]
            
            clean_lines = []
            for line in raw_translit.split('\n'):
                if any(c in line for c in ['*', '\\', '+', '}', '{', ']', '[', '(', ')']):
                    continue
                clean_lines.append(line)
                
            cleaned_block = " ".join(clean_lines)
            sanitized_iast = convert_prabhupada_to_iast(cleaned_block)
            
            v["transliteration"] = sanitized_iast
            
            try:
                devanagari_output = transliterate(sanitized_iast, sanscript.IAST, sanscript.DEVANAGARI)
                v["sanskrit"] = devanagari_output
            except Exception as translit_err:
                logging.warning(f"Conversion anomaly at Ch {v['chapter']} Verse {v['verse']}: {str(translit_err)}")
                v["sanskrit"] = "N/A"
                
        if (idx + 1) % 50 == 0 or (idx + 1) == total_verses:
            logging.info(f"Sanskrit generation status: {idx + 1}/{total_verses} records processed.")

    logging.info("\n" + "="*60)
    logging.info("VERSE COUNT BREAKDOWN VS EXPECTED:")
    
    expected_counts = {
        "ONE": 46, "TWO": 72, "THREE": 43, "FOUR": 42, "FIVE": 29,
        "SIX": 47, "SEVEN": 30, "EIGHT": 28, "NINE": 34, "TEN": 42,
        "ELEVEN": 55, "TWELVE": 20, "THIRTEEN": 35, "FOURTEEN": 27,
        "FIFTEEN": 20, "SIXTEEN": 24, "SEVENTEEN": 28, "EIGHTEEN": 78
    }
    
    chapter_actual_counts = {}
    actual_verse_total = 0
    
    for v in all_verses:
        chap = v["chapter"]
        chapter_actual_counts[chap] = chapter_actual_counts.get(chap, 0)
        
        # Extract all numbers from strings like "TEXT 22 & 23" or "40-41"
        verse_num_str = v["verse"]
        numbers_found = re.findall(r'\d+', verse_num_str)
        
        if len(numbers_found) > 1:
            start_v = int(numbers_found[0])
            end_v = int(numbers_found[-1])
            count_for_this_record = (end_v - start_v + 1)
        else:
            count_for_this_record = 1
            
        chapter_actual_counts[chap] += count_for_this_record
        actual_verse_total += count_for_this_record
            
    for chap, expected in expected_counts.items():
        found = chapter_actual_counts.get(chap, 0)
        if found == expected:
            logging.info(f"Chapter {chap:10}: {found}/{expected} [PERFECT]")
        else:
            logging.warning(f"Chapter {chap:10}: {found}/{expected} <--- MISSING {expected - found} VERSES!")
        
    logging.info("="*60)
    logging.info(f"Total JSON Records Generated: {len(all_verses)}")
    logging.info(f"Total Underlying Verses Parsed: {actual_verse_total} / 700")
    logging.info("="*60 + "\n")

    return all_verses



def download_gita_pdf(url="https://www.prabhupada-books.de/pdf/Bhagavad-gita-As-It-Is.pdf", dest_dir="./data"):
    # Ensure the target directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Extract the filename from the URL and construct the full file path
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_dir, filename)
    
    try:
        print(f"Starting download from: {url}")
        # Send a GET request to the URL with streaming enabled for large files
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Write the file in chunks to avoid using too much memory
        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    
        print(f"Download successful! File saved to: {filepath}")
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the download: {e}")




if __name__ == "__main__":
    logging.info("System Initialized. Ingestion runner tracking active.")

    if not Path("./data/Bhagavad-gita-As-It-Is.pdf").exists():
        download_gita_pdf()
        
    pdf_file_path = "./data/Bhagavad-gita-As-It-Is.pdf"
    output_json_path = "./data/gita_structured.json"
    
    try:
        raw_pdf_text = extract_raw_text(pdf_file_path)
        structured_data = parse_gita_text(raw_pdf_text)
        
        logging.info(f"Dumping database structure cache to format: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Task Complete. Processed payload array count: {len(structured_data)} verse records.")
        
    except FileNotFoundError:
        logging.critical(f"Abort. Targeted physical book file asset not located at: '{pdf_file_path}'")
    except Exception as e:
        logging.critical(f"Pipeline crashed due to unhandled runtime exception: {str(e)}")