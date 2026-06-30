from src.ingest import Parser
import fire
import bm25s
import json
from .models import MinimalSearchResults, MinimalSource

class RAGAplication:

    def index(self, max_chunk_size: int = 2000):
        parser = Parser(max_chunk_size)
        print("Ingestion on process...")
        parser.get_chunks('vllm-0.10.1')
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10)-> str:
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
        print(json_result)


    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass


def main() -> None:
    print("\n========= Hello from rag-against-the-machine! =========\n")
    fire.Fire(RAGAplication)
    print("\n========= Program ended =========\n")
