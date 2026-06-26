from pathlib import Path
from .models import MinimalSource
import json

def get_chunks(path: str):
    files = [str(f) for f in Path(path).rglob("*") if f.is_file()]
    create_dir("data/processed/chunks")
    for file in files:
        end_check = False
        last_char = 0
        first_char = 0
        if file.endswith(".py"):
            with open(file, 'r') as f:
                content = f.read()
                while not end_check and last_char < len(content):
                    chunk_limit = min(first_char + 1999, len(content))
                    last_char = (
                        max(content[first_char:chunk_limit].rfind("class"),
                            content[first_char:chunk_limit].rfind("def"))
                    )
                    if last_char <= 0:
                        last_char = chunk_limit
                    else:
                        last_char += first_char
                    if last_char == len(content):
                        end_check = True
                    chunk = content[first_char:last_char]
                    save_chunk(chunk, file, first_char, last_char)
                    first_char = last_char
        elif file.endswith(".md"):
            with open(file, 'r') as f:
                content = f.read()
                while not end_check and last_char < len(content):
                    chunk_limit = min(first_char + 1999, len(content))
                    last_char = content[first_char:chunk_limit].rfind("\n# ")
                    if last_char <= 0:
                        last_char = content[first_char:chunk_limit].rfind("\n## ")
                    if last_char <= 0:
                        last_char = chunk_limit
                    else:
                        last_char += first_char
                    if last_char == len(content):
                        end_check = True
                    chunk = content[first_char:last_char]
                    save_chunk(chunk, file, first_char, last_char)
                    first_char = last_char


def save_chunk(content: str, path: str, first_char: int, last_char: int):

    chunk = MinimalSource(
        file_path = path,
        first_character_index = first_char,
        last_character_index = last_char,
        text = content
    )
    json_test = chunk.model_dump_json(indent=4)
    print(json_test)


def create_dir(path: str):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print("Error. Permission denied while creating the directory '{path}'")
        exit(1)
    except OSError as e:
        print(f"Error. Failed to create the directory '{path}': {e}")
        exit(1)
