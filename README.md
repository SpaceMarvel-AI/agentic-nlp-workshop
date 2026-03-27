# Agentic NLP Workshop

A hands-on workshop exploring agentic NLP patterns using LangChain, RAG, and AI agents.

## Project Structure

```
agentic-nlp-workshop/
├── demo1_rag/          # Retrieval-Augmented Generation demo
├── demo2_agent/        # AI Agent with tool use demo
├── demo3_capstone/     # Capstone project combining RAG + Agents
├── demo4_usecases/     # Real-world use case examples
├── utils/              # Shared utility functions
├── data/               # Sample data files (PDFs, etc.)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```

## Demos

- **Demo 1 - RAG**: Build a retrieval-augmented generation pipeline over PDF documents using FAISS and LangChain.
- **Demo 2 - Agent**: Create an AI agent with custom tools and reasoning capabilities.
- **Demo 3 - Capstone**: Combine RAG and agent patterns into a complete agentic application.
- **Demo 4 - Use Cases**: Explore real-world applications including summarization, Q&A, and structured extraction.
