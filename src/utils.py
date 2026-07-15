import bm25s
from src.models import MinimalSearchResults, MinimalSource, MinimalAnswer
import json
from pathlib import Path
import re
from typing import Any, cast
from transformers import PreTrainedTokenizerBase


def get_search_results(query: str, k: int = 10) -> dict[str, Any]:
    """Retrieves the most relevant document chunks for a given query.

    The function loads the previously generated BM25 index and the chunk
    metadata, performs a retrieval using the provided query, and validates the
    retrieved sources before packaging them into a structured search result.

    Args:
        query: Text query used to retrieve relevant document chunks.
        k: Maximum number of document chunks to retrieve.

    Returns:
        A dictionary containing the question identifier, the original query,
        and a list of the retrieved sources. Each source includes its file
        path, character offsets, and chunk text.
    """

    index_path = "data/processed/bm25_index"
    chunks_path = "data/processed/chunks/all_chunks.json"
    sources = []
    try:
        retriever = bm25s.BM25.load(index_path, load_corpus=True)
    except Exception:
        print("Error. There is no index.")
        exit(1)

    with open(chunks_path, 'r', encoding="utf-8") as f:
        chunks = json.load(f)

    query_tokens = bm25s.tokenize(query, stopwords="english")
    results = retriever.retrieve(query_tokens,
                                 k=min(k, len(chunks)),
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
    return dict_result


def create_dir(path: str) -> None:
    """Creates a directory if it does not already exist.

    The function creates the specified directory, including any missing parent
    directories. If the directory cannot be created due to insufficient
    permissions or another operating system error, an error message is printed
    and the program terminates.

    Args:
        path: Path of the directory to create.

    Output:
        Creates the requested directory on the filesystem if it does not
        already exist. Prints an error message and exits if the directory
        cannot be created.
    """

    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error. Permission denied while "
              f"creating the directory '{path}'")
        exit(1)
    except OSError as e:
        print(f"Error. Failed to create the directory '{path}': {e}")
        exit(1)


def get_prompt(sources: list[dict[str, Any]],
               context: str,
               question: str) -> str:
    """Builds the prompt used for answer generation.

    The function appends the retrieved document sources to the provided
    context and constructs the complete prompt that will be sent to the
    language model. The generated prompt contains the system instructions,
    the retrieved context, the user question, and the required output format.

    Args:
        sources: List of retrieved document sources used as context for the
            language model.
        context: Initial context string to which the retrieved sources are
            appended.
        question: User question that the model should answer.

    Returns:
        A formatted prompt string ready to be passed to the language model for
        answer generation.
    """

    for i, source in enumerate(sources, 1):
        context += (
            f"### Source {i}\n"
            f"File: {source['file_path']}\n"
            f"{source['text']}\n\n"
        )

    prompt = """<|im_start|>system
You are a technical assistant answering questions about the vLLM repository.

Use only the provided context.
Do not use external knowledge.
Ignore sources that are not directly relevant to the question.
Do not infer that something is required unless """ + \
        """the provided context explicitly says it.

Give exactly one answer.
Use at most 2 sentences.
Do not repeat the answer.
Do not provide alternative phrasings.
Do not include a separate source list.
Do not explain what each source contains.

Your output must be exactly one short answer.
The answer must end with one or more citations """ + \
        """in this exact format: [Source N].
Do not write an answer without a citation.
Do not include a separate source list.
If the context does not contain the answer, write exactly: """ + \
        """The provided context is insufficient.
<|im_end|>
<|im_start|>user
"""

    prompt += "Context:\n" + context + "\n"
    prompt += "Question:\n" + question + "\n"
    prompt += "Required answer format:\n<answer text>. [Source N]"
    prompt += "\n\n/no_think\n"
    prompt += "<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def get_answer(model: Any,
               tokenizer: PreTrainedTokenizerBase,
               search: dict[str, Any],
               k: int = 10) -> MinimalAnswer:
    """Generates an answer from the retrieved context using the language model.

    The function constructs a prompt from the retrieved sources, performs text
    generation with the provided language model, cleans the generated output,
    validates that the answer follows the expected format, and packages the
    result into a validated answer object.

    Args:
        model: Language model used to generate the answer.
        tokenizer: Tokenizer associated with the language model.
        search: Dictionary containing the question and its retrieved sources.
        k: Maximum number of retrieved sources considered. This parameter is
            included for consistency with the application interface.

    Returns:
        A ``MinimalAnswer`` object containing the original question, its
        identifier, the retrieved sources, and the generated answer.
    """

    sources = search["retrieved_sources"]
    context = f"Question: {search['question']}\n\n"
    question = search['question']
    question_id = search['question_id']
    prompt = get_prompt(sources, context, question)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=96,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][prompt_length:]

    answer_text = cast(
                       str,
                       tokenizer.decode(output_ids, skip_special_tokens=True),
                       )
    answer_text = answer_text.replace("<think>\n</think>", "")
    answer_text = answer_text.replace("<think>", "")
    answer_text = answer_text.replace("</think>", "")
    answer_text = answer_text.replace("<answer>", "")
    answer_text = answer_text.replace("</answer>", "")
    answer_text = answer_text.replace("<answer text>", "")
    answer_text = answer_text.replace("</answer text>", "")
    answer_text = answer_text.replace("<answer", "")
    answer_text = answer_text.strip()

    clean = answer_text.strip()
    clean = clean.replace(".", "")
    clean = clean.replace(",", "")

    if (
        "context is insufficient" in clean
        or "Source N" in clean
        or clean.endswith("[Source")
        or clean.endswith("[")
        or clean.endswith(",")
        or re.fullmatch(r'(\s*\[Source\s+\d+\]\s*)+', clean)
        or clean.startswith('.')
        or answer_text.startswith('[Source')
    ):
        answer_text = "The provided context is insufficient."

    if answer_text != "The provided context is insufficient.":
        answer_text = re.sub(r'(?<!\s)\[Source', ' [Source', answer_text)

    minimal_answer = MinimalAnswer(
        question=question,
        question_id=question_id,
        retrieved_sources=sources,
        answer=answer_text
    )

    return (minimal_answer)
