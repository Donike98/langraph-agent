# LangGraph Helper Agent

Hey! This is a helper agent I built to answer questions about LangGraph and LangChain. It's designed to give you quick, code-first answers without all the fluff you'd normally get from an LLM.

<p align="center">
  <img src="architecture/arch.png" alt="Architecture Diagram" width="800">
</p>

## What it does?

The agent works in two modes depending on what you need:

**Offline mode** uses BM25 search on locally downloaded docs with Ollama (llama3.2). Great when you're on a plane or just want fast, consistent answers without burning through API credits.

**Online mode** taps into Tavily's web search and Google's Gemini API. Use this when you need the latest info or when the docs don't have what you're looking for.

Behind the scenes, it's a 3-node LangGraph workflow that classifies your question, retrieves relevant context, and synthesizes a clean answer. No hallucination, no citing sources you can't verify - just straight code examples and clear explanations.

## Getting started

Install dependencies:

```bash
pip install -r requirements.txt
```

Want to use online mode? Create a `.env` file with your API keys:

```bash
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

Both are free tier:
- Gemini: Get your key from [Google AI Studio](https://aistudio.google.com/apikey). Note that this is AI Studio, not Google Cloud or Vertex AI. You get 1500 requests per day.
- Tavily: Sign up at [tavily.com](https://tavily.com) for 1000 searches per month.

Then run with the `--online` flag:

```bash
python main.py --online "What are the latest LangGraph features?"
```

## Setting up offline mode

If you want to use offline mode, you'll need to download the docs and build a search index:

```bash
python scripts/pull_docs.py
python scripts/build_index.py
```

You'll also need Ollama installed. Get it from [ollama.com](https://ollama.com/download) (check the [docs](https://github.com/ollama/ollama) for platform-specific installation).

Make sure Ollama is running with llama3.2:

```bash
ollama serve
ollama pull llama3.2:latest
```

You can also use the helper script:

```bash
./start_ollama.sh
```

## How it works

The agent automatically figures out what kind of question you're asking:

- **How-to questions** get a working code example first, then a brief explanation
- **Comparison questions** get a clear difference statement plus code for both options
- **Troubleshooting** gets the fixed code upfront, then why it was broken
- **Conceptual questions** get a short explanation followed by a minimal example

Responses are capped at 40 lines to keep things scannable. No bold formatting, no source citations, no filler text. Just natural language and working code.

## Usage examples

```bash
# Ask anything about LangGraph/LangChain
python main.py --offline "StateGraph vs MessageGraph?"
python main.py --online "Best practices for state management 2025?"

# Or set the mode via environment variable
export AGENT_MODE=offline
python main.py "How to handle errors in nodes?"
```

## Project structure

```
├── main.py              # CLI entry point
├── agent.py             # Core LangGraph workflow
├── retriever_offline.py # BM25 search
├── retriever_online.py  # Tavily web search
├── data/
│   ├── raw/             # Downloaded docs
│   └── processed/       # BM25 index
├── scripts/
│   ├── pull_docs.py     # Download documentation
│   └── build_index.py   # Build search index
└── requirements.txt     # Python dependencies
```

## Troubleshooting

**Offline mode not working?**

Check that Ollama is running (`ollama serve`) and that you've pulled the model (`ollama pull llama3.2:latest`). Also verify the index exists at `data/processed/bm25_index.pkl`.

**Online mode not working?**

Double-check your API keys in `.env`. Make sure you're using the Google AI Studio key, not a Google Cloud key. You can verify your quotas in the Gemini and Tavily dashboards.

**Getting "command not found"?**

Make sure you're in the project directory and try `python3 main.py` if `python` doesn't work.

## Why I built this?

Documentation search is usually either too broad (LLMs hallucinate) or too narrow (keyword search misses things). This agent tries to hit the sweet spot: accurate retrieval with intelligent synthesis, giving you just enough context to solve your problem without drowning you in docs.

The LangGraph workflow makes it easy to swap retrievers, adjust prompts, or add new query types without touching the core logic. And running everything locally means no vendor lock-in or API dependencies if you don't want them.

## Resources to explore

Here are some great resources to expand your understanding:

If you're new to LangGraph, start with the [official docs](https://langchain-ai.github.io/langgraph/) and [LangChain Academy](https://academy.langchain.com/) has free courses that are actually pretty good. The [LangGraph repo](https://github.com/langchain-ai/langgraph) has solid examples in the examples folder if you learn better from code.

For RAG systems, check out the [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) video series. If you want to understand why BM25 works, [this Elastic post](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) breaks it down well. For more sophisticated retrieval, [ChromaDB](https://www.trychroma.com/) and [FAISS](https://github.com/facebookresearch/faiss) are worth exploring once you outgrow BM25.

Running models locally is easier than you think. [Ollama](https://github.com/ollama/ollama/tree/main/docs) is what I use here.

If you need better search, [Tavily](https://tavily.com/) is optimized for LLM retrieval. 
