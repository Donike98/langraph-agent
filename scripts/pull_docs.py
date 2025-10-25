import os
import json
import hashlib
import time
import requests
from pathlib import Path

SOURCES = [
    {
        "name": "langgraph-llms.txt",
        "url": "https://langchain-ai.github.io/langgraph/llms.txt"
    },
    {
        "name": "langgraph-llms-full.txt",
        "url": "https://langchain-ai.github.io/langgraph/llms-full.txt"
    },
    {
        "name": "langchain-llms.txt",
        "url": "https://python.langchain.com/llms.txt"
    }
]

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = DATA_DIR / "metadata.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def load_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def download_doc(source):
    print(f"Downloading {source['name']}...")
    try:
        response = requests.get(source['url'], timeout=30)
        response.raise_for_status()
        content = response.text
        
        file_path = RAW_DIR / source['name']
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "url": source['url'],
            "sha256": compute_sha256(content),
            "size": len(content),
            "last_fetched": int(time.time())
        }
    except Exception as e:
        print(f"Error downloading {source['name']}: {e}")
        return None


def merge_docs():
    print("Merging documentation files...")
    merged_content = []
    
    for source in SOURCES:
        file_path = RAW_DIR / source['name']
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                merged_content.append(f"\n\n{source['name']}\n\n")
                merged_content.append(content)
    
    merged_file = PROCESSED_DIR / "merged.txt"
    with open(merged_file, 'w', encoding='utf-8') as f:
        f.write(''.join(merged_content))
    
    print(f"Merged documentation saved to {merged_file}")


def main():
    print("Starting documentation refresh...")
    metadata = load_metadata()
    
    for source in SOURCES:
        doc_metadata = download_doc(source)
        if doc_metadata:
            old_hash = metadata.get(source['name'], {}).get('sha256')
            new_hash = doc_metadata['sha256']
            
            if old_hash != new_hash:
                print(f"{source['name']} updated (hash: {new_hash[:8]})")
            else:
                print(f"{source['name']} unchanged")
            
            metadata[source['name']] = doc_metadata
    
    save_metadata(metadata)
    merge_docs()
    print("\nDocumentation refresh complete!")
    print("Next step: Run 'python scripts/build_index.py' to rebuild search index")


if __name__ == "__main__":
    main()

