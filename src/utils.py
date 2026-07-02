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
    return dict_result

def create_dir(path: str):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error. Permission denied while creating the directory '{path}'")
        exit(1)
    except OSError as e:
        print(f"Error. Failed to create the directory '{path}': {e}")
        exit(1)

def	get_prompt(sources, context: str, question: str) -> str:

    for i, source in enumerate(sources, 1):
        context += (
            f"### Source {i}\n"
            f"File: {source['file_path']}\n"
            f"{source['text']}\n\n"
        )

    prompt = f"""<|im_start|>system
You are a technical assistant answering questions about the vLLM repository.

Use only the provided context.
Do not use external knowledge.
Ignore sources that are not directly relevant to the question.
Do not infer that something is required unless the provided context explicitly says it.

Give exactly one answer.
Use at most 2 sentences.
Do not repeat the answer.
Do not provide alternative phrasings.
Do not include a separate source list.
Do not explain what each source contains.

Your output must be exactly one short answer.
The answer must end with one or more citations in this exact format: [Source N].
Do not write an answer without a citation.
Do not include a separate source list.
If the context does not contain the answer, write exactly: The provided context is insufficient.
<|im_end|>
<|im_start|>user

Context:
{context}

Question:
{question}

Required answer format:
<answer text>. [Source N]

/no_think
<|im_end|>
<|im_start|>assistant
"""
    return prompt
