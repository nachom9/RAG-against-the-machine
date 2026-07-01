import bm25s
from src.models import MinimalSearchResults, MinimalSource
import json
from pathlib import Path

def get_search_results(query: str, k: int = 10)-> str:
    index_path = "data/processed/bm25_index"
    chunks_path = "data/processed/chunks/all_chunks.json"
    sources = []

    retriever = bm25s.BM25.load(index_path, load_corpus=True)

    with open(chunks_path, 'r', encoding="utf-8") as f:
        chunks = json.load(f)

    query_tokens = bm25s.tokenize(query, stopwords="english")
    results = retriever.retrieve(query_tokens, k=min(k, len(chunks)), return_as="documents")
    for r in results[0]:
        chunk_data = chunks[r['id']]
        validate_source = MinimalSource(
                file_path=chunk_data['file_path'],
                first_character_index=chunk_data['first_character_index'],
                last_character_index=chunk_data['last_character_index'],
                text=chunk_data['text']
        )
        sources.append(validate_source)

    search_result = MinimalSearchResults(
            question_id="single_query",
            question=query,
            retrieved_sources=sources,
    )

    dict_result = search_result.model_dump()
    json_result = json.dumps(dict_result, indent=4, ensure_ascii=False)
    return json_result

def create_dir(path: str):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error. Permission denied while creating the directory '{path}'")
        exit(1)
    except OSError as e:
        print(f"Error. Failed to create the directory '{path}': {e}")
        exit(1)
