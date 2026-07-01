from src.ingest import Parser
import fire
import bm25s
import json
from .models import MinimalSearchResults, MinimalSource, StudentSearchResults
from src.utils import get_search_results
from transformers import AutoTokenizer, AutoModelForCausalLM

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


    def search_dataset(self, dataset_path: str, output_path: str, k: int = 10):

        index_path = "data/processed/bm25_index"
        chunks_path = "data/processed/chunks/all_chunks.json"
        search_results_list = []
        retriever = bm25s.BM25.load(index_path, load_corpus=True)
        with open(chunks_path, 'r', encoding="utf-8") as f:
            chunks = json.load(f)

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        for question in dataset['rag_questions']:
            sources = []
            query = question['question']
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
                question_id=question['question_id'],
                question=query,
                retrieved_sources=sources,
                )
            search_results_list.append(search_result)

        searchs = StudentSearchResults(
            search_results=search_results_list,
            k=k
        )
        dict_searchs = searchs.model_dump()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dict_searchs, f, indent=4, ensure_ascii=False)

    def answer(self, query: str, k: int = 10):
        search = get_search_results(query, k)
        context = f"Question: {search['question']}\n\n"

        for i, source in enumerate(search["retrieved_sources"], 1):
            context += (
                f"### Source {i}\n"
                f"File: {source['file_path']}\n"
                f"{source['text']}\n\n"
            )

        prompt = f"""You are a precise technical assistant for the vLLM inference engine.
        Your task is to answer the user's question using ONLY the provided codebase context.

        [CONSTRAINTS]
        1. Answer the question using ONLY the information provided in the [CONTEXT] section.
        2. If the context does not contain the answer, reply exactly with: "I do not have enough information in the context to answer."
        3. Do not make up or hallucinate any functions, code blocks, parameters, or explanations.
        4. Keep your answer technical, concise, and straight to the point.

        [CONTEXT]
        {context}

        [USER QUESTION]
        {query}

        [ANSWER]"""
        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
            )
        prompt_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][prompt_length:]
        answer_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(answer_text)





    def answer_dataset(self):
        pass

    def evaluate(self):
        pass


def main() -> None:
    print("\n========= Hello from rag-against-the-machine! =========\n")
    fire.Fire(RAGAplication)
    print("\n========= Program ended =========\n")
