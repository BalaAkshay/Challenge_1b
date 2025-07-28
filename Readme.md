## 🧠 Models and Libraries Used

- PyMuPDF: For PDF parsing and layout extraction.
- collections.Counter: For computing dominant styles (e.g., body font size).
- re and json: For heading identification and structured output formatting.
- (Optional): An alternative ML-powered version (document_processor_with_MLModel.py) uses:
  - KeyBERT
  - SentenceTransformers for offline keyword extraction.

⚠️ The main implementation used in main.py is based on rule-based heuristics only and does *not* require a machine learning model.

## ⚙️ How to Build and Run

This solution is fully containerized with Docker. You can build and run it easily using the commands below:


### 🧪 Build the Docker Image

```
docker build -t pdf-outline-extractor .
```

### 🚀 Run the Container

```
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output pdf-outline-extractor
```

- Mounts the local input/ and output/ directories to the container.
- All .pdf files in input/ will be processed.
- Results are saved as .json files in output/.
