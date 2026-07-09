from src.ingest import Parser
from pathlib import Path
import fire
import bm25s
import json
from .models import MinimalSearchResults, MinimalSource, StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer
from src.utils import get_search_results, get_prompt, create_dir, get_answer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm


class RAGApplication:

    def index(self, max_chunk_size: int = 2000):
        parser = Parser(max_chunk_size)
        print("Ingestion on process...")
        parser.get_chunks('data/raw/vllm-0.10.1')
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> str:
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

    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 10):

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
        create_dir(save_directory)
        output_file_path = Path(save_directory) / Path(dataset_path).name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(dict_searchs, f, indent=4, ensure_ascii=False)

    def answer(self, query: str, k: int = 10):
        search = get_search_results(query, k)
        sources = search["retrieved_sources"]
        context = f"Question: {search['question']}\n\n"
        question = search['question']
        prompt = get_prompt(sources, context, question)

        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
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

        answer_text = tokenizer.decode(output_ids, skip_special_tokens=True)
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

    def answer_dataset(self, search_results_path: str, save_directory: str):
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
            for search_result in tqdm(search_results['search_results'], desc="Generating answers"):
                answer = get_answer(model, tokenizer, search_result)
                answers.append(answer)

        final_output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.get("k", 10)
            )
        create_dir(save_directory)
        output_file_path = Path(save_directory) / Path(search_results_path).name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_output.model_dump(), f, indent=4, ensure_ascii=False)

    def evaluate(self, search_results_path: str, dataset_path: str):
        correct_sources = 0
        i = 0
        with open(search_results_path, 'r', encoding='utf-8') as f:
            search_results = json.load(f)
        with open(dataset_path, 'r', encoding='utf-8') as d:
            dataset = json.load(d)

        for search_result in search_results['search_results']:
            commun_chars = 0
            iuo_metric = 0
            data_file_path = dataset['rag_questions'][i]['sources'][0]['file_path']
            for source in search_result['retrieved_sources']:
                file_path = source['file_path']
                if data_file_path == file_path:
                    first_char = source['first_character_index']
                    last_char = source['last_character_index']
                    data_first_char = dataset['rag_questions'][i]['sources'][0]['first_character_index']
                    data_last_char = dataset['rag_questions'][i]['sources'][0]['last_character_index']
                    commun_chars = max(0, min(last_char, data_last_char) - max(first_char, data_first_char))
                    total_size = (last_char - first_char) + (data_last_char - data_first_char) - commun_chars
                    iuo_metric = commun_chars / total_size
                    if iuo_metric >= 0.05:
                        correct_sources += 1
                        break
            i += 1
        print(correct_sources / i)


def main() -> None:
    print("\n========= Hello from rag-against-the-machine! =========\n")
    fire.Fire(RAGApplication)
    print("\n========= Program ended =========\n")
