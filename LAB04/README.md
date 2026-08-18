# LAB04
 
# Project Structure
 
```text
RAG-Project/
├── data/                                # Directory for raw and processed datasets
│   ├── golden_set.json                  # Ground truth dataset used for system evaluation
│   └── qa_looseweight.txt               # Source document containing Q&A data
├── evaluation/                          # Scripts and modules for evaluating the RAG system
│   ├── build_golden_set.py              # Script to build the evaluation dataset
│   ├── eval_generation.py               # Evaluates the generation (LLM) component
│   ├── eval_retrieval.py                # Evaluates the retrieval component
│   └── metrics.py                       # Defines evaluation metrics for performance testing
├── labs/                                # Step-by-step lab scripts for learning
│   ├── lab01_extract_text.py            # Extract text from source documents
│   ├── lab02_chunking.py                # Split text into manageable chunks
│   ├── lab03_create_embeddings.py       # Generate vector embeddings
│   ├── lab04_create_vector_db.py        # Build the vector database
│   ├── lab05_query_embedding.py         # Create embeddings for user queries
│   ├── lab06_similarity_search.py       # Retrieve relevant chunks via similarity
│   └── lab07_complete_retrieval.py      # Complete retrieval pipeline integration
├── outputs/                             # Stores outputs generated during processing or evaluation
│   ├── chunks.json                      # Processed text chunks
│   ├── embeddings.npy                   # Saved embedding vectors
│   ├── eval_retrieval.json              # Results from retrieval evaluation
│   ├── extracted_text.json              # Extracted text from source documents
│   └── retrieval_results.json           # Sample results from retrieval queries
├── src/                                 # Core modules of the RAG system
│   ├── document_loader.py               # Handles loading and parsing source documents
│   ├── embedding_model.py               # Manages the generation of text embeddings
│   ├── generator.py                     # Manages LLM interaction and answer generation
│   ├── hybrid_retriever.py              # Combines dense and sparse retrieval methods
│   ├── index_meta.py                    # Manages metadata for the vector index
│   ├── memory.py                        # Handles conversational memory (chat history)
│   ├── prompt_templates.py              # Contains templates for formatting LLM prompts
│   ├── query_transform.py               # Transforms or expands user queries for better retrieval
│   ├── rag_pipeline.py                  # Orchestrates the entire end-to-end RAG workflow
│   ├── rerankers.py                     # Re-ranks retrieved documents for better relevance
│   ├── retriever.py                     # Handles semantic retrieval from the vector store
│   ├── text_splitter.py                 # Splits loaded documents into optimal chunks
│   └── vector_store.py                  # Manages interactions with the vector database
├── vector_db/                           # Directory containing the built indices and stores
│   ├── bm25_index.pkl                   # Sparse index (BM25) for keyword-based search
│   ├── chunk_store.json                 # Stores chunk texts mapped to their IDs
│   ├── document.index                   # Dense vector index (e.g., FAISS) for similarity search
│   └── index_meta.json                  # Metadata configuration for the vector database
├── build_index.py                       # Script to process documents and build vector/BM25 indices
├── config.py                            # Central configuration file for the system
├── create_memory_ppt.py                 # Utility script to generate a presentation on memory
├── create_redesigned_ppt.py             # Utility script to generate a redesigned presentation
├── create_updated_ppt.py                # Utility script to generate an updated presentation
├── main.py                              # Application entry point to interact with the RAG system
└── requirements.txt                     # Lists required Python dependencies
```

### Key Components of the RAG Workflow

*   **Data & Configuration**: The raw source data is maintained in the `data/` folder. The `config.py` file contains the centralized configuration for paths, model parameters, and other system settings.
*   **Initialization (Indexing)**: Before querying, the knowledge base must be built. The `build_index.py` script leverages modules in `src/` (like `document_loader.py`, `text_splitter.py`, and `embedding_model.py`) to process `qa_looseweight.txt` and populate indices within the `vector_db/` directory.
*   **Core RAG Logic (`src/`)**:
    *   **Retrieval**: Handled by `vector_store.py` (database connection), `retriever.py` (dense similarity search), and `hybrid_retriever.py` (combining dense and BM25 search).
    *   **Augmentation & Generation**: The `rag_pipeline.py` fetches the most relevant context using retrievers and optionally `rerankers.py`. It then constructs a prompt using `prompt_templates.py` and feeds it to the LLM via `generator.py` to get the final answer.
*   **Application Entry**: The main user-facing application logic starts from `main.py`, orchestrating the queries and displaying answers.
*   **Evaluation**: The `evaluation/` directory provides scripts to validate both the retrieval and generation phases of the workflow against a golden dataset (`data/golden_set.json`).
 
# Summary
RAG (Retrieval-Augmented Generation) คือเทคนิคที่ให้ LLM ค้นข้อมูลจากฐานความรู้ก่อนตอบ แทนที่จะตอบจากความจำอย่างเดียว ในกรณีนี้คือฐานข้อมูล Q&A ภาษาไทยเรื่องอาหารเพื่อสุขภาพ/ลดน้ำหนัก แบ่งเป็นหมวดๆ พร้อมแท็กกำกับเพื่อให้ retrieve ได้แม่นยำ เมื่อผู้ใช้ถาม ระบบจะค้นคำถาม-คำตอบที่ใกล้เคียงที่สุดมาแนบเป็น context ให้โมเดลใช้ตอบอย่างมีหลักฐานอ้างอิงแทนการเดา 
