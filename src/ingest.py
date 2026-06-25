from pathlib import Path
from .models import MinimalSource
import json

def get_chunks(path: str):
    files = [str(f) for f in Path(path).rglob("*") if f.is_file()]
    create_dir("data/processed/chunks")
    for file in files:
        end_check = False
        last_char = 0
        if file.endswith(".py"):
            with open(file, 'r') as f:
                content = f.read()
                first_char = 0
                while not end_check and last_char < len(content):
                    chunk_limit = first_char + 1999
                    chunk = content[first_char:chunk_limit]
                    last_char = (
                        max(chunk[first_char:chunk_limit].rfind("class"),
                            chunk[first_char:chunk_limit].rfind("def"))
                    )
                    if last_char < 0:
                        last_char = chunk_limit
                    if last_char == len(content):
                        end_check = True
                    chunk = content[first_char:last_char]
                    save_chunk(chunk, path, first_char, last_char)
                    first_char = last_char + 1
        elif file.endswith(".md"):
            with open(file, 'r') as f:
                content = f.read()
                first_char = 0
                while not end_check and last_char < len(content):
                    chunk_limit = first_char + 1999
                    chunk = content[first_char:chunk_limit]
                    last_char = chunk[first_char:chunk_limit].rfind("\n#")
                    if last_char == -1:
                        last_char = chunk[first_char:chunk_limit].rfind("\n##")
                    if last_char < 0:
                        last_char = chunk_limit
                    if last_char == len(content):
                        end_check = True
                    chunk = content[first_char:last_char]
                    save_chunk(chunk, path, first_char, last_char)
                    first_char = last_char + 1


def save_chunk(content: str, path: str, first_char: int, last_char: int):

    chunk = MinimalSource(
        file_path = path,
        first_character_index = first_char,
        last_character_index = last_char,
        text = content
    )


def create_dir(path: str):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print("Error. Permission denied while creating the directory '{path}'")
        exit(1)
    except OSError as e:
        print(f"Error. Failed to create the directory '{path}': {e}")
        exit(1)
