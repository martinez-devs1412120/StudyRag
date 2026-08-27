# StudyRAG

RAG-based study assistant for your BSCS course materials (PDFs, PPTX).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env template and add your Groq API key (free at console.groq.com)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

### 1. Add your course materials

Place PDF and PPTX files in `data/documents/`

### 2. Ingest documents

```bash
python main.py ingest
```

### 3. Ask questions

```bash
# Single question
python main.py ask "What did the reviewer say about normalization in DBMS?"

# Interactive chat
python main.py chat
```

### 4. Other commands

```bash
python main.py stats   # Show document count
python main.py clear   # Clear vector store
```

## Configuration

Edit `config.yaml` to adjust:
- Chunk size/overlap
- Embedding model
- LLM provider (Groq or Ollama)
- Number of retrieved chunks (top_k)

## Local LLM (Ollama)

To use Ollama instead of Groq:

1. Install Ollama: `https://ollama.ai`
2. Pull a model: `ollama pull llama3.1:8b`
3. Change `LLM_PROVIDER: "ollama"` in config.yaml
4. Run `ollama serve` in background