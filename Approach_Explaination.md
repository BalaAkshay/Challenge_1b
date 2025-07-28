**Core Architecture Understanding**

The `DocumentProcessor` class represents the first stage of a multi-phase pipeline. Think of it as a document archaeologist that carefully excavates the structural bones of a PDF before the real intelligence work begins. The class focuses on three fundamental tasks: identifying the document's title, mapping its hierarchical structure, and extracting semantic keywords from headings.

**The Foundation: Text Analysis and Structure Detection**

The approach begins with a crucial insight about document processing - you need to understand the visual hierarchy before you can understand the semantic hierarchy. The `_analyze_text_styles()` method acts like a typography detective, examining all font sizes throughout the document to determine what constitutes "normal" body text. This baseline becomes the reference point for identifying headings, much like how you'd identify mountains by first understanding sea level.

The heading classification logic in `_classify_headings()` employs multiple heuristics working together. It doesn't rely on just one signal but combines font size analysis, font weight detection, text length patterns, and formatting cues. This multi-signal approach is essential because PDF documents can be inconsistent - some might use bold text for headings, others might use larger fonts, and some might use both or neither.

**Semantic Enhancement Through KeyBERT**

Here's where the approach becomes particularly clever for the persona-driven challenge. Instead of just extracting structural headings, the code uses KeyBERT with a sentence transformer model to extract meaningful keyphrases from each heading. This creates a semantic fingerprint for each section that can later be matched against persona interests and job-to-be-done requirements.

The choice of the "all-MiniLM-L6-v2" model is strategic - it's lightweight enough to meet the 1GB constraint while providing robust semantic understanding. By running this locally without internet access, the system maintains the required offline capability.

**Preparing for Persona-Driven Intelligence**

While this code doesn't yet implement the persona matching logic, it's clearly designed as the foundation for that intelligence. The structured output with hierarchical levels, page numbers, and extracted keywords creates a rich index that a higher-level system can use to rank and filter sections based on persona needs.

The approach anticipates that different personas will have different ways of consuming information - a researcher might care about methodology sections, while a student might focus on fundamental concepts. By extracting both structural and semantic information, this foundation enables sophisticated matching algorithms to identify the most relevant sections for any given persona and task.

This preprocessing stage essentially transforms unstructured PDFs into structured, semantically-rich data that can be intelligently filtered and prioritized based on user context.
