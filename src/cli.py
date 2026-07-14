from src.ingest import Parser
from pathlib import Path
import fire
import bm25s
import json
from .models import (MinimalSearchResults, MinimalSource,
                     StudentSearchResults, MinimalAnswer,
                     StudentSearchResultsAndAnswer)
from src.utils import get_search_results, get_prompt, create_dir, get_answer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from typing import cast, Any


class RAGApplication:

    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size < 150:
            print("Error. Chunk size must be at least 150")
            exit(1)
        parser = Parser(max_chunk_size)
        print("Ingestion on process...")
        parser.get_chunks('data/raw/vllm-0.10.1')
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        if k < 1 or k > 10:
            print("Error. k value must be between 1 and 10")
            exit(1)
        index_path = "data/processed/bm25_index"
        chunks_path = "data/processed/chunks/all_chunks.json"
        sources = []

        retriever = bm25s.BM25.load(index_path, load_corpus=True)

        try:
            with open(chunks_path, 'r', encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"Error. Couldn't open file '{chunks_path}': {e}")
            exit(1)

        query_tokens = bm25s.tokenize(query, stopwords="english")
        results = retriever.retrieve(query_tokens, k=min(k, len(chunks)),
                                     return_as="documents")
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
        for s in dict_result['retrieved_sources']:
            print(f"{s['file_path']} [{s['first_character_index']}"
                  f":{s['last_character_index']}]")

    def search_dataset(self, dataset_path: str,
                       save_directory: str, k: int = 10) -> None:
        if k < 1 or k > 10:
            print("Error. k value must be between 1 and 10")
            exit(1)
        index_path = "data/processed/bm25_index"
        chunks_path = "data/processed/chunks/all_chunks.json"
        search_results_list = []
        retriever = bm25s.BM25.load(index_path, load_corpus=True)

        try:
            with open(chunks_path, 'r', encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"Error. Couldn't open file '{chunks_path}': {e}")
            exit(1)
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Error. Couldn't open file '{dataset_path}': {e}")
            exit(1)

        if len(dataset['rag_questions']) < 1:
            print("Error. There is no questions")
            exit(1)

        try:
            for question in dataset['rag_questions']:
                sources = []
                query = question['question']
                query_tokens = bm25s.tokenize(query, stopwords="english")
                results = retriever.retrieve(
                          query_tokens,
                          k=min(k, len(chunks)),
                          return_as="documents"
                )
                for r in results[0]:
                    chunk_data = chunks[r['id']]
                    validate_source = MinimalSource(
                            file_path=chunk_data['file_path'],
                            first_character_index=(
                                chunk_data['first_character_index']
                            ),
                            last_character_index=(
                                chunk_data['last_character_index']
                            ),
                            text=chunk_data['text']
                    )
                    sources.append(validate_source)
                search_result = MinimalSearchResults(
                    question_id=question['question_id'],
                    question=query,
                    retrieved_sources=sources,
                    )
                search_results_list.append(search_result)
        except KeyError:
            print("Error. Wrong JSON format")
            exit(1)

        searchs = StudentSearchResults(
            search_results=search_results_list,
            k=k
        )
        dict_searchs = searchs.model_dump()
        create_dir(save_directory)
        output_file_path = Path(save_directory) / Path(dataset_path).name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(dict_searchs, f, indent=4, ensure_ascii=False)
        print(f"Saved search_results to {output_file_path}")

    def answer(self, query: str, k: int = 10) -> None:
        if k < 1 or k > 10:
            print("Error. k value must be between 1 and 10")
            exit(1)
        search = get_search_results(query, k)
        sources = search["retrieved_sources"]
        context = f"Question: {search['question']}\n\n"
        question = search['question']
        prompt = get_prompt(sources, context, question)

        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=20,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
            )

        prompt_length = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][prompt_length:]

        answer_text = cast(
                           str,
                           tokenizer.decode(output_ids,
                                            skip_special_tokens=True),
                           )
        answer_text = answer_text.replace("<think>\n</think>", "")
        answer_text = answer_text.replace("<think>", "")
        answer_text = answer_text.replace("</think>", "")
        answer_text = answer_text.strip()

        minimal_answer = MinimalAnswer(
            question=question,
            question_id='single_query',
            retrieved_sources=sources,
            answer=answer_text
        )
        result = StudentSearchResultsAndAnswer(
            k=k,
            search_results=[minimal_answer]
        )

        print(result.model_dump_json(indent=4))

    def answer_dataset(self,
                       search_results_path: str,
                       save_directory: str) -> None:
        answers = []
        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            )
        with open(search_results_path, 'r', encoding='utf-8') as f:
            search_results = json.load(f)
            for search_result in tqdm(search_results['search_results'],
                                      desc="Generating answers"):
                answer = get_answer(model, tokenizer, search_result)
                answers.append(answer)

        final_output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.get("k", 10)
            )
        create_dir(save_directory)
        output_file_path = (Path(save_directory) /
                            Path(search_results_path).name
                            )
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_output.model_dump(), f,
                      indent=4, ensure_ascii=False)

    def answer_dataset_test(self,
                            search_results_path: str,
                            save_directory: str) -> None:
        answers = []

        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
        )

        with open(search_results_path, "r", encoding="utf-8") as f:
            search_results = json.load(f)

        for search_result in tqdm(search_results["search_results"],
                                  desc="Generating answers"):
            answer = get_answer(model, tokenizer, search_result)

            answers.append({
                "question": answer.question,
                "answer": answer.answer
            })

        create_dir(save_directory)
        output_file_path = (Path(save_directory) /
                            Path(search_results_path).name)

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)

    def evaluate(self,
                 search_results_path: str,
                 dataset_path: str) -> None:
        k_list = [1, 3, 5, 10]
        results = []

        with open(search_results_path, 'r', encoding='utf-8') as f:
            search_results = json.load(f)
        with open(dataset_path, 'r', encoding='utf-8') as d:
            dataset = json.load(d)

        for k in k_list:
            correct_sources = 0
            i = 0
            for search_result in search_results['search_results']:
                commun_chars = 0
                iuo_metric = 0
                data_file_path = (dataset['rag_questions'][i]
                                  ['sources'][0]['file_path'])
                for source in search_result['retrieved_sources'][:k]:
                    file_path = source['file_path']
                    if data_file_path == file_path:
                        first_char = source['first_character_index']
                        last_char = source['last_character_index']
                        data = dataset['rag_questions'][i]['sources'][0]
                        data_first_char = data['first_character_index']
                        data_last_char = data['last_character_index']
                        commun_chars = max(0, (min(last_char, data_last_char)
                                           - max(first_char, data_first_char)))
                        total_size = ((last_char - first_char)
                                      + (data_last_char - data_first_char)
                                      - commun_chars)
                        iuo_metric = commun_chars / total_size
                        if iuo_metric >= 0.05:
                            correct_sources += 1
                            break
                i += 1
            results.append(correct_sources / i)
        print(f"Recall@1: {results[0]:.3f} "
              f"Recall@3: {results[1]:.3f} "
              f"Recall@5: {results[2]:.3f} "
              f"Recall@10: {results[3]:.3f}")


def main() -> None:
    print("\n========= Hello from rag-against-the-machine! =========\n")
    fire.Fire(RAGApplication)
    print("\n========= Program ended =========\n")
