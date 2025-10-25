import pickle
from pathlib import Path

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_FILE = PROCESSED_DIR / "bm25_index.pkl"


def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Search index not found at {INDEX_FILE}\n"
            "Run 'python scripts/build_index.py' to create it"
        )
    
    with open(INDEX_FILE, 'rb') as f:
        return pickle.load(f)


def retrieve_offline(query, top_k=3):
    index_data = load_index()
    bm25 = index_data['bm25']
    chunks = index_data['chunks']
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = [chunks[i] for i in top_indices]
    
    return "\n\n".join(results)


if __name__ == "__main__":
    query = "How do I add persistence in LangGraph?"
    print(f"Query: {query}\n")
    
    try:
        context = retrieve_offline(query)
        print("Retrieved context:")
        print(context[:500] + "..." if len(context) > 500 else context)
    except FileNotFoundError as e:
        print(f"Error: {e}")

