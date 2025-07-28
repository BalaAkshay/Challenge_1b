import fitz  # PyMuPDF
import re
from collections import Counter
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer



class DocumentProcessor:
    """
    A class to process a single PDF document, extracting its title,
    heading hierarchy, and key phrases using KeyBERT.
    """
    model_name = "all-MiniLM-L6-v2"
    local_model_path = "./sentence_transformer_models/" + model_name
    def __init__(self, pdf_path, model_path=local_model_path):
        self.pdf_path = pdf_path
        try:
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open or read PDF: {pdf_path}")
            raise e

        # Load offline model for KeyBERT
        self.bert_model = SentenceTransformer(model_path)
        self.keyword_extractor = KeyBERT(model=self.bert_model)

    def _get_document_title(self):
        if self.doc.metadata and self.doc.metadata.get('title'):
            title = self.doc.metadata['title'].strip()
            if title:
                return title

        max_size = 0
        title_text = ""
        if len(self.doc) > 0:
            page = self.doc[0]
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            if span['size'] > max_size:
                                max_size = span['size']
                                title_text = span['text'].strip()
        return title_text if title_text else "Untitled Document"

    def _analyze_text_styles(self):
        sizes = []
        for page in self.doc:
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            sizes.append(round(span['size']))
        if not sizes:
            return 12.0
        count = Counter(sizes)
        return float(count.most_common(1)[0][0])

    def _classify_headings(self, body_font_size):
        headings = []
        for page_num, page in enumerate(self.doc):
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        if not line['spans']:
                            continue
                        first_span = line['spans'][0]
                        line_text = "".join(s['text'] for s in line['spans']).strip()
                        font_size = round(first_span['size'])
                        font_name = first_span['font'].lower()
                        
                        if not line_text:
                            continue

                        is_heading = False
                        if font_size > body_font_size * 1.15: is_heading = True
                        if 'bold' in font_name: is_heading = True
                        if len(line_text) < 100 and not line_text.endswith('.') and font_size > body_font_size: is_heading = True
                        if re.match(r'^\d+(\.\d+)*\s|\b[A-Z]\.\s', line_text): is_heading = True
                        if len(line_text.split()) > 25: is_heading = False
                        if not is_heading: continue

                        # Extract keyphrases for the heading
                        keywords = self.keyword_extractor.extract_keywords(
                            line_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3
                        )
                        keywords_only = [kw for kw, _ in keywords]

                        headings.append({
                            'text': line_text,
                            'page': page_num + 1,
                            'size': font_size,
                            'keywords': keywords_only
                        })
        return headings

    def _determine_hierarchy(self, classified_headings):
        if not classified_headings:
            return []

        unique_sizes = sorted(list(set(h['size'] for h in classified_headings)), reverse=True)
        size_to_level = {}
        if len(unique_sizes) > 0: size_to_level[unique_sizes[0]] = "H1"
        if len(unique_sizes) > 1: size_to_level[unique_sizes[1]] = "H2"
        if len(unique_sizes) > 2: size_to_level[unique_sizes[2]] = "H3"

        outline = []
        for heading in classified_headings:
            level = size_to_level.get(heading['size'])
            if level:
                outline.append({
                    'level': level,
                    'text': heading['text'],
                    'page': heading['page'],
                    'keywords': heading['keywords']
                })
        return outline

    def process(self):
        title = self._get_document_title()
        body_font_size = self._analyze_text_styles()
        classified_headings = self._classify_headings(body_font_size)
        outline = self._determine_hierarchy(classified_headings)

        self.doc.close()
        return {
            "title": title,
            "outline": outline
        }
