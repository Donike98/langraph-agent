import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_FILE = PROCESSED_DIR / "merged.txt"
INDEX_FILE = PROCESSED_DIR / "bm25_index.pkl"


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def build_index():
    print("Building BM25 search index...")
    
    if not MERGED_FILE.exists():
        print(f"Error: {MERGED_FILE} not found!")
        print("Run 'python scripts/pull_docs.py' first")
        return
    
    with open(MERGED_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Chunking documentation...")
    chunks = chunk_text(content)
    print(f"Created {len(chunks)} chunks")
    
    print("Building BM25 index...")
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    index_data = {
        'bm25': bm25,
        'chunks': chunks
    }
    
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"Index saved to {INDEX_FILE}")
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    build_index()

