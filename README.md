# Bot5e - D&D 5e Assistant

A sophisticated chatbot application designed to answer Dungeons & Dragons 5th Edition rules questions using Retrieval-Augmented Generation (RAG) and local LLM inference via Ollama.

## Features

- **Intelligent Query Classification**: Automatically routes queries to the appropriate agent (general chat or rules lookup)
- **RAG-Powered Rules Search**: Searches through the D&D 5e System Reference Document using vector embeddings
- **Local LLM**: Uses Ollama for privacy-focused, local language model inference
- **Response Formatting**: Formats D&D content with proper markdown rendering
- **Conversation History**: Maintains context across multiple questions
- **Streamlit UI**: Clean, intuitive web interface

## Architecture

```
┌─────────────┐
│   Streamlit │  User Interface
│   (app.py)  │
└──────┬──────┘
       │
       ├──► classifier.py (Query Classification)
       │
       ├──► ChatbotAgent (General Conversation)
       │
       └──► PDFQAAgent (Rules Lookup)
            ├──► text_read_split.py (PDF Processing)
            ├──► embedder.py (Generate Embeddings)
            ├──► vector_store.py (FAISS Search)
            └──► formatter.py (Response Formatting)
```

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: Install from [ollama.ai](https://ollama.ai)
3. **D&D 5e SRD PDF**: Place `SRD-OGL_V5.1.pdf` in the project root

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Bot5e
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama with llama3.1:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1
ollama serve  # Keep this running in a separate terminal
```

4. Place the D&D 5e SRD PDF in the project root directory as `SRD-OGL_V5.1.pdf`

## Configuration

All configuration is managed through [config.yaml](config.yaml). Key settings:

```yaml
# LLM Settings
llm:
  model: "llama3.1"
  temperature: 0.7
  ollama_base_url: "http://localhost:11434"

# Document Processing
documents:
  pdf_path: "SRD-OGL_V5.1.pdf"
  embeddings_path: "embeddings.pkl"
  index_path: "index.faiss"

# Vector Store
vector_store:
  top_k: 7  # Number of relevant chunks to retrieve

# Classification
classifier:
  confidence_threshold: 0.6
```

See [config.yaml](config.yaml) for all available options.

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Start asking questions:
   - **General**: "What's a good class for beginners?"
   - **Rules**: "How does divine smite work?"
   - **Mechanics**: "Explain advantage and disadvantage"

## Project Structure

```
Bot5e/
├── app.py                 # Streamlit UI and main application
├── agents.py              # ChatbotAgent and PDFQAAgent classes
├── classifier.py          # Query classification logic
├── config.py              # Configuration management
├── config.yaml            # Configuration file
├── logger.py              # Logging setup
├── embedder.py            # Embedding generation
├── vector_store.py        # FAISS vector store operations
├── text_read_split.py     # PDF text extraction and chunking
├── formatter.py           # Response formatting for D&D content
├── requirements.txt       # Python dependencies
├── SRD-OGL_V5.1.pdf      # D&D 5e System Reference Document
├── embeddings.pkl         # Cached embeddings (generated)
├── index.faiss           # FAISS index (generated)
└── bot5e.log             # Application logs (generated)
```

## How It Works

### 1. Query Classification
When you ask a question, the classifier determines whether it's:
- **General conversation** (greetings, casual D&D discussion)
- **Rules-based question** (specific mechanics, spells, classes)

### 2. Agent Routing
- **ChatbotAgent**: Handles general conversation using the LLM
- **PDFQAAgent**: Searches the SRD and provides rules-based answers

### 3. RAG Pipeline (for rules questions)
1. Your question is converted to a vector embedding
2. FAISS finds the 7 most relevant chunks from the SRD
3. The LLM receives these chunks as context
4. The response is formatted with proper D&D styling

### 4. Response Formatting
The formatter enhances responses with:
- Proper markdown tables
- Stat blocks
- Spell descriptions
- Ability score formatting

## Caching

On first run, the application:
1. Extracts text from the PDF (~30 seconds)
2. Generates embeddings (~2 minutes)
3. Creates FAISS index (~5 seconds)

These are cached in `embeddings.pkl` and `index.faiss` for instant subsequent startups.

## Logging

Logs are written to:
- **Console**: INFO level and above
- **bot5e.log**: All logs (configurable in config.yaml)

Log level can be changed in [config.yaml](config.yaml):
```yaml
logging:
  level: "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Error Handling

The application includes comprehensive error handling:
- **PDF not found**: Clear error message with file path
- **Ollama not running**: Graceful failure with instructions
- **Empty queries**: Input validation
- **Model failures**: Fallback error messages
- **Classification errors**: Defaults to rules agent (safer)

## Troubleshooting

### "Connection refused" error
- Ensure Ollama is running: `ollama serve`
- Check if the correct port is configured in config.yaml

### "PDF file not found"
- Verify `SRD-OGL_V5.1.pdf` is in the project root
- Check the `pdf_path` setting in config.yaml

### Slow responses
- Ollama models are large - first response may be slower
- Consider using a smaller model in config.yaml
- Ensure adequate RAM (8GB+ recommended)

### "Failed to load classifier model"
- Check internet connection (first run downloads model)
- Model is ~1.5GB - may take time to download

## Performance

- **First run**: ~2-3 minutes (PDF processing + embedding generation)
- **Subsequent runs**: ~5 seconds (loads from cache)
- **Query response time**: 3-10 seconds (depends on LLM and hardware)

## Dependencies

Key dependencies:
- **streamlit**: Web UI framework
- **langchain**: LLM orchestration
- **ollama**: Local LLM inference
- **sentence-transformers**: Text embeddings
- **faiss-cpu**: Vector similarity search
- **transformers**: Zero-shot classification
- **PyMuPDF**: PDF text extraction

See [requirements.txt](requirements.txt) for complete list.

## License

This project uses the D&D 5e System Reference Document (SRD) which is available under the Open Gaming License (OGL) v1.0a.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Add tests for new features
- Update documentation
- Run linting before submitting

## Acknowledgments

- Built with [Ollama](https://ollama.ai) for local LLM inference
- Uses [FAISS](https://github.com/facebookresearch/faiss) for vector search
- D&D 5e SRD provided under the OGL by Wizards of the Coast

---

**Note**: This is an unofficial fan tool and is not affiliated with or endorsed by Wizards of the Coast.
