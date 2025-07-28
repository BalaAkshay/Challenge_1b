# src/new_main.py
import fitz  # PyMuPDF
import json
import re
import os
from collections import Counter
from datetime import datetime
import math
from typing import List, Dict, Any, Tuple, Set

# --- NEW: Import the DocumentProcessor from its new location ---
from src.document_processor_with_MLModel import DocumentProcessor 

# Import SentenceTransformer
from sentence_transformers import SentenceTransformer, util

# Define the absolute path to your downloaded MiniLM model
# Based on your file structure and common Docker setups, /app is likely your root.
# Adjust this path if your 'sentence_transformer_models' directory is elsewhere relative to /app.
MINILM_MODEL_PATH = "/app/sentence_transformer_models/all-MiniLM-L6-v2" 

class PersonaDrivenProcessor:
    """
    A class to process multiple PDF documents, extracting relevant sections
    based on a specific persona and job-to-be-done, enhanced by DocumentProcessor's capabilities,
    and now using Sentence Transformers (MiniLM) for semantic relevance.
    """
    
    def __init__(self, documents_dir: str, persona: str, job_to_be_done: str):
        """
        Initialize the processor with documents, persona, and job description.
        
        Args:
            documents_dir (str): Directory containing PDF documents
            persona (str): Description of the user persona
            job_to_be_done (str): Specific task the persona needs to accomplish
        """
        self.documents_dir = documents_dir
        self.persona = persona
        self.job_to_be_done = job_to_be_done
        self.documents = []
        self.document_contents = {}
        
        # --- NEW: Load Sentence Transformer Model for PersonaDrivenProcessor ---
        print(f"PersonaDrivenProcessor: Loading Sentence Transformer model from: {MINILM_MODEL_PATH}")
        try:
            self.model = SentenceTransformer(MINILM_MODEL_PATH)
            print("PersonaDrivenProcessor: Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Sentence Transformer model from {MINILM_MODEL_PATH}: {e}")
            print("Please ensure the path is correct and the model files are complete.")
            raise e # Crucial to raise if model fails to load

        # --- NEW: Create a combined query embedding for persona and job ---
        self.query_text = f"Persona: {self.persona}. Job to be done: {self.job_to_be_done}."
        self.query_embedding = self.model.encode(self.query_text, convert_to_tensor=True)
        print(f"PersonaDrivenProcessor: Generated query embedding for: '{self.query_text[:70]}...'")

        self._load_documents()
        
    def _load_documents(self):
        """Load all PDF documents from the directory."""
        for filename in os.listdir(self.documents_dir):
            if filename.lower().endswith('.pdf'):
                self.documents.append(filename)
                
    def _extract_text_with_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF while maintaining structure information,
        leveraging DocumentProcessor for heading and keyword extraction.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict containing structured text data
        """
        structured_content = {
            'title': "Untitled Document",
            'sections': [],
            'pages': {}
        }
        
        try:
            # --- NEW: Instantiate DocumentProcessor with the correct model_path ---
            processor = DocumentProcessor(pdf_path, model_path=MINILM_MODEL_PATH) 
            doc_outline = processor.process() # This returns {'title': ..., 'outline': [...]}

            structured_content['title'] = doc_outline.get('title', "Untitled Document")

            # Store full page text
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                structured_content['pages'][page_num + 1] = page.get_text("text").strip()
            doc.close()

            # Group content into sections based on DocumentProcessor's outline
            current_section_title = None
            current_section_page = None
            current_section_content_lines = []
            
            # Use a dummy heading at the end to ensure the last section is processed
            outline_with_end_marker = doc_outline['outline'] + [{'text': 'END_OF_DOCUMENT_MARKER', 'page': len(structured_content['pages']) + 1, 'level': 'H1', 'keywords': []}]

            outline_idx = 0
            for line_data in self._get_all_lines_data(pdf_path): # Helper to get all lines
                is_outline_heading = False
                if outline_idx < len(outline_with_end_marker) and \
                   line_data['text'] == outline_with_end_marker[outline_idx]['text'] and \
                   line_data['page'] == outline_with_end_marker[outline_idx]['page']:
                    is_outline_heading = True
                
                if is_outline_heading:
                    if current_section_title is not None:
                        structured_content['sections'].append({
                            'title': current_section_title,
                            'page': current_section_page,
                            'content': " ".join(current_section_content_lines).strip(),
                            # Use section.get('keywords', []) for safe access
                            'font_size': outline_with_end_marker[outline_idx-1]['size'] if outline_idx > 0 and 'size' in outline_with_end_marker[outline_idx-1] else line_data['font_size'], 
                            'keywords': outline_with_end_marker[outline_idx-1]['keywords'] if outline_idx > 0 and 'keywords' in outline_with_end_marker[outline_idx-1] else []
                        })
                    
                    if outline_with_end_marker[outline_idx]['text'] != 'END_OF_DOCUMENT_MARKER':
                        current_section_title = outline_with_end_marker[outline_idx]['text']
                        current_section_page = outline_with_end_marker[outline_idx]['page']
                        current_section_content_lines = []
                        outline_idx += 1
                    else: # Reached the end marker
                        current_section_title = None # Reset for next document
                        current_section_page = None
                        current_section_content_lines = []
                        break # Exit the loop after processing the last section
                else:
                    current_section_content_lines.append(line_data['text'])

        except Exception as e:
            print(f"Failed to process {pdf_path} with DocumentProcessor: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: if DocumentProcessor fails, try basic text extraction
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    structured_content['pages'][page_num + 1] = page_text.strip()
                    full_text += page_text + "\n"
                doc.close()
                structured_content['sections'].append({
                    'title': structured_content['title'],
                    'page': 1,
                    'content': full_text.strip(),
                    'font_size': 12, # Default
                    'keywords': []
                })
            except Exception as e_fallback:
                print(f"Fallback text extraction also failed for {pdf_path}: {e_fallback}")
                pass # Return empty sections/pages
        
        return structured_content

    def _get_all_lines_data(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Helper to get all lines with their page, text, and font size."""
        all_lines_data = []
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        if line['spans']:
                            all_lines_data.append({
                                'text': "".join(s['text'] for s in line['spans']).strip(),
                                'page': page_num + 1,
                                'font_size': round(line['spans'][0]['size'])
                            })
        doc.close()
        return all_lines_data
        
    def _calculate_relevance_score(self, text: str, section_title: str = "", section_keywords: List[str] = []) -> float:
        parts = []

        if section_title:
            parts.append(f"Title: {section_title}")
        if text:
            parts.append(f"Content: {text}")
        if section_keywords:
            parts.append(f"Keywords: {', '.join(section_keywords)}")

        combined = " ".join(parts)

        try:
            content_embedding = self.model.encode(combined, convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(self.query_embedding, content_embedding).item()
            normalized_score = (cosine_similarity + 1) / 2
            return max(0.01, normalized_score)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

        
    def _extract_subsections(self, content: str, target_word_count: int = 100) -> List[str]:
    
        if not content.strip():
            return []

        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        subsections = []

        current_chunk = []
        current_word_count = 0
        min_words = int(target_word_count * 0.8)
        max_words = int(target_word_count * 1.3)

        for sentence in sentences:
            word_count = len(sentence.split())
            current_chunk.append(sentence)
            current_word_count += word_count

            if current_word_count >= target_word_count or (current_word_count >= min_words and word_count < 10):
                # Finalize chunk
                subsection = " ".join(current_chunk).strip()
                if len(subsection.split()) >= min_words:
                    subsections.append(subsection)
                current_chunk = []
                current_word_count = 0

        # Add any remaining sentences if not already included
        if current_chunk:
            remaining_subsection = " ".join(current_chunk).strip()
            if len(remaining_subsection.split()) >= min_words:
                subsections.append(remaining_subsection)
            elif subsections:
                # Append remaining small chunk to the last subsection
                subsections[-1] += " " + remaining_subsection
            else:
                subsections.append(remaining_subsection)

        return subsections
        
    def process(self) -> Dict[str, Any]:
        """
        Main processing function that orchestrates the entire workflow.
        
        Returns:
            Dict containing the structured output as per requirements
        """
        print(f"\nProcessing {len(self.documents)} documents...")
        
        all_sections_raw = []
        all_subsections_raw = []
        
        for doc_filename in self.documents:
            doc_path = os.path.join(self.documents_dir, doc_filename)
            print(f"Processing {doc_filename}...")
            
            try:
                structured_content = self._extract_text_with_structure(doc_path)
                self.document_contents[doc_filename] = structured_content
                
                # Process sections extracted by structure (now much more accurate due to DocumentProcessor)
                for section in structured_content['sections']:
                    cleaned_section_content = section['content'].strip()
                    if len(cleaned_section_content) < 50: # Still filter very short sections
                        continue
                        
                    relevance_score = self._calculate_relevance_score(
                        cleaned_section_content, 
                        section['title'],
                        section.get('keywords', []) # Pass keywords from DocumentProcessor as additional context
                    )
                    
                    if relevance_score > 0.1: # Adjust threshold as needed for semantic similarity
                        section_data = {
                            'document': doc_filename,
                            'section_title': section['title'],
                            'importance_rank': relevance_score, 
                            'page_number': section['page']
                        }
                        all_sections_raw.append(section_data)
                        
                        # Extract subsections from this section's content
                        subsections = self._extract_subsections(cleaned_section_content)
                        for subsection_text in subsections:
                            subsection_score = self._calculate_relevance_score(subsection_text, section['title'], section.get('keywords', []))
                            
                            if subsection_score > 0.1: # Adjust threshold
                                subsection_data = {
                                    'document': doc_filename,
                                    'refined_text': subsection_text, 
                                    'page_number': section['page'], 
                                    'importance_rank': subsection_score 
                                }
                                all_subsections_raw.append(subsection_data)
                                
                # If no structured sections were found, or the document is very short/unstructured,
                # process entire pages as a fallback.
                if not structured_content['sections'] and structured_content['pages']:
                    print(f"No structured sections found or document too short for {doc_filename}. Processing pages as fallback.")
                    for page_num, page_content in structured_content['pages'].items():
                        if len(page_content.strip()) > 100:
                            page_relevance = self._calculate_relevance_score(page_content, structured_content['title'])
                            if page_relevance > 0.1:
                                all_sections_raw.append({
                                    'document': doc_filename,
                                    'section_title': f"Page {page_num} Content from {structured_content['title']}", 
                                    'importance_rank': page_relevance, 
                                    'page_number': page_num
                                })
                                page_subsections = self._extract_subsections(page_content)
                                for p_sub_text in page_subsections:
                                    p_sub_score = self._calculate_relevance_score(p_sub_text, structured_content['title'])
                                    if p_sub_score > 0.1:
                                        all_subsections_raw.append({
                                            'document': doc_filename,
                                            'refined_text': p_sub_text,
                                            'page_number': page_num,
                                            'importance_rank': p_sub_score
                                        })

            except Exception as e:
                print(f"Error processing {doc_filename}: {e}")
                import traceback
                traceback.print_exc() 
                continue
        
        # Sort by the decimal importance_rank first
        all_sections_raw.sort(key=lambda x: x['importance_rank'], reverse=True)
        top_sections = []
        for i, section in enumerate(all_sections_raw):
            if i >= 5: break 
            section['importance_rank'] = i + 1  # Convert to integer rank here
            top_sections.append(section)

        # Sort by the decimal importance_rank first
        all_subsections_raw.sort(key=lambda x: x['importance_rank'], reverse=True)
        top_subsections = []
        for i, subsection in enumerate(all_subsections_raw):
            if i >= 5: break 
            subsection['importance_rank'] = i + 1  # Convert to integer rank here
            top_subsections.append(subsection)
            
        result = {
            "metadata": {
                "input_documents": sorted(self.documents), 
                "persona": self.persona, 
                "job_to_be_done": self.job_to_be_done, 
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": top_sections,
            "subsection_analysis": top_subsections 
        }
        
        return result


def main():
    """
    Main function to process documents based on persona and job requirements.
    """
    # These directories are relative to the Docker container's root or wherever '/app' is mapped.
    INPUT_DIR = "/app/input"
    OUTPUT_DIR = "/app/output"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    config_path = os.path.join(INPUT_DIR, "config.json")
    
    default_persona_str = "Travel Planner"
    default_job_str = "Plan a trip of 4 days for a group of 10 college friends."

    persona_to_pass = default_persona_str
    job_to_pass = default_job_str

    if not os.path.exists(config_path):
        print("Warning: config.json not found in input directory. Using default persona and job-to-be-done.")
    else:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                persona_config = config.get("persona")
                if isinstance(persona_config, dict) and "role" in persona_config:
                    persona_to_pass = persona_config["role"]
                elif isinstance(persona_config, str):
                    persona_to_pass = persona_config
                else:
                    print(f"Warning: Invalid 'persona' format in config.json. Expected string or {{'role': '...'}}. Using default: {default_persona_str}")
                    persona_to_pass = default_persona_str

                job_config = config.get("job_to_be_done")
                if isinstance(job_config, dict) and "task" in job_config:
                    job_to_pass = job_config["task"]
                elif isinstance(job_config, str):
                    job_to_pass = job_config
                else:
                    print(f"Warning: Invalid 'job_to_be_done' format in config.json. Expected string or {{'task': '...'}}. Using default: {default_job_str}")
                    job_to_pass = default_job_str

        except json.JSONDecodeError as e:
            print(f"Error decoding config.json: {e}. Using default persona and job-to-be-done.")
        except Exception as e:
            print(f"An unexpected error occurred while reading config.json: {e}. Using default persona and job-to-be-done.")
            
    print(f"\nPersona: {persona_to_pass}")
    print(f"Job to be done: {job_to_pass}")
    
    try:
        processor = PersonaDrivenProcessor(INPUT_DIR, persona_to_pass, job_to_pass)
        result = processor.process()
        
        output_path = os.path.join(OUTPUT_DIR, "challenge1b_output.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"\nSuccessfully created analysis at {output_path}")
        print(f"Processed {len(result['metadata']['input_documents'])} documents")
        print(f"Extracted {len(result['extracted_sections'])} relevant sections")
        print(f"Identified {len(result['subsection_analysis'])} relevant subsections")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        error_output = {
            "error": "Processing failed",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(OUTPUT_DIR, "error.json"), 'w', encoding='utf-8') as f:
            json.dump(error_output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()