# RAG App

A Retrieval-Augmented Generation (RAG) application that enables question-answering over PDF documents using LangChain, OpenAI embeddings, and FAISS vector store.

## Overview

This application processes PDF documents, creates vector embeddings, and builds a RAG (Retrieval-Augmented Generation) chain that allows you to query the document's content using natural language questions. The system uses:

- **LangChain Classic** for building the RAG chain
- **OpenAI** for embeddings and LLM inference
- **FAISS** for efficient vector similarity search
- **PyPDF** for PDF document loading

## Features

- 📄 Load and process PDF documents
- 🔍 Automatic text chunking with overlap for better context
- 🧠 Generate embeddings using OpenAI's embedding models
- 💾 Persistent vector store using FAISS
- 💬 Interactive question-answering over document content
- 🔗 Retrieval-augmented generation for accurate answers

## Prerequisites

- Python 3.14 or higher
- OpenAI API key
- UV package manager (recommended) or pip

## Installation

1. **Clone the repository** (if applicable):

   ```bash
   git clone <repository-url>
   cd RagApp
   ```

2. **Install dependencies using UV**:

   ```bash
   uv sync
   ```

   Or using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Project Structure

```
RagApp/
├── main.py                  # Main application code
├── pyproject.toml          # Project dependencies
├── reAct.pdf              # Example PDF document
├── faiss_index_reaAct/    # Generated FAISS vector store
│   ├── index.faiss
│   └── index.pkl
└── README.md              # This file
```

## Usage

1. **Activate the virtual environment** (if using UV):

   ```bash
   source .venv/bin/activate
   ```

2. **Place your PDF file** in the project root directory (or update the `pdf_path` variable in `main.py`)

3. **Run the application**:
   ```bash
   python main.py
   ```

## How It Works

1. **Document Loading**: The PDF is loaded and parsed into text documents
2. **Text Splitting**: Documents are split into chunks of 1000 characters with 30-character overlap
3. **Embedding Generation**: Each chunk is converted to vector embeddings using OpenAI's embedding model
4. **Vector Store Creation**: Embeddings are stored in a FAISS index for fast similarity search
5. **RAG Chain Building**: A retrieval chain is created that:
   - Retrieves relevant document chunks based on the query
   - Combines them with the user's question
   - Generates an answer using GPT-4o-mini
6. **Query Processing**: The user's question is processed through the RAG chain to generate contextually relevant answers

## Example Output

When you run the application, you'll see output like this:

```
Hello from ragapp!
ReAct is a novel prompt-based paradigm that integrates reasoning and acting in language models to enhance task-solving capabilities across various domains, such as question answering and decision making. It allows models to interact with external sources, like APIs, to gather information, which improves interpretability and reduces issues like hallucination and error propagation. Empirical evaluations show that ReAct significantly outperforms traditional methods in both few-shot learning setups and interactive decision-making tasks.
```

## Dependencies

- `faiss-cpu>=1.12.0` - Vector similarity search
- `langchain>=1.0.3` - LangChain core framework
- `langchain-community>=0.4.1` - Community integrations
- `langchain-openai>=1.0.2` - OpenAI integrations
- `langchainhub>=0.1.21` - LangChain Hub for prompts
- `langsmith>=0.4.40` - LangSmith observability
- `pypdf>=6.1.3` - PDF document loading
- `python-dotenv>=1.2.1` - Environment variable management

## Customization

### Change the PDF Document

Update the `pdf_path` variable in `main.py`:

```python
pdf_path = "your_document.pdf"
```

### Modify the Question

Change the question in the `invoke()` call:

```python
res = rag_chain.invoke({"input": "Your question here?"})
```

### Adjust Chunking Parameters

Modify the `CharacterTextSplitter` parameters:

```python
text_splitter = CharacterTextSplitter(
    chunk_size=1000,      # Size of each chunk
    chunk_overlap=30,     # Overlap between chunks
    separator="\n"        # Separator for splitting
)
```

### Change the LLM Model

Update the model in the `ChatOpenAI` initialization:

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

## Notes

- The FAISS index is saved locally and will be reused if it already exists
- The first run will create the index, subsequent runs will load it
- Make sure your `.env` file is in `.gitignore` to avoid committing API keys
- The application uses `allow_dangerous_deserialization=True` when loading FAISS indices - only load indices from trusted sources

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Author

[Add your name/contact information here]
