from pathlib import Path
import bm25s
from .models import MinimalSource
import json
from src.utils import create_dir


class Parser:

    def __init__(self, max_chunk_size: int):
        self.text_chunks: list = []
        self.max_chunk_size = max_chunk_size
        self.sources: list = []
        self.path: str = "data/processed/chunks"

    def get_chunks(self, path: str) -> None:
        files = [str(f) for f in Path(path).rglob("*") if f.is_file()]
        ignore = ["Zone.Identifier", ".git", "/."]
        if len(files) < 2500:
            print(len(files))
            print("Error. Files are missing")
            exit(1)

        for file in files:
            if any(word in file for word in ignore):
                continue
            end_check = False
            last_char = 0
            first_char = 0
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error. Couldn't open file '{file}': {e}")
                continue

            if file.endswith(".py"):
                while not end_check and last_char < len(content):
                    chunk_limit = min(first_char + (self.max_chunk_size - 1),
                                      len(content))
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
                    self.save_chunk(chunk, file, first_char, last_char)
                    first_char = last_char

            elif file.endswith(".md"):
                while not end_check and last_char < len(content):
                    chunk_limit = min(first_char + (self.max_chunk_size - 1),
                                      len(content))
                    last_char = content[first_char:chunk_limit].rfind("\n# ")
                    if last_char <= 0:
                        last_char = (content[first_char:chunk_limit]
                                     .rfind("\n## "))
                    if last_char <= 0:
                        last_char = chunk_limit
                    else:
                        last_char += first_char
                    if last_char == len(content):
                        end_check = True
                    chunk = content[first_char:last_char]
                    self.save_chunk(chunk, file, first_char, last_char)
                    first_char = last_char

            else:
                while not end_check and last_char < len(content):
                    chunk_limit = min(first_char + (self.max_chunk_size - 1),
                                      len(content))
                    last_char = chunk_limit
                    if last_char == len(content):
                        end_check = True
                    chunk = content[first_char:last_char]
                    self.save_chunk(chunk, file, first_char, last_char)
                    first_char = last_char

        self.generate_json()

    def save_chunk(self,
                   content: str,
                   path: str,
                   first_char: int,
                   last_char: int) -> None:

        chunk = MinimalSource(
            file_path=path,
            first_character_index=first_char,
            last_character_index=last_char,
            text=content
        )
        self.text_chunks.append(content)
        self.sources.append(chunk)

    def generate_json(self) -> None:
        create_dir(self.path)

        dict_sources = [chunk.model_dump() for chunk in self.sources]
        path = f"{self.path}/all_chunks.json"
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(dict_sources, f, indent=4, ensure_ascii=False)
        chunk_tokens = bm25s.tokenize(self.text_chunks, stopwords="english")
        chunks_range = [str(i) for i in range(len(self.text_chunks))]
        retriever = bm25s.BM25(corpus=chunks_range)
        retriever.index(chunk_tokens)
        index_path = "data/processed/bm25_index"
        try:
            retriever.save(index_path, corpus=chunks_range)
        except Exception as e:
            print(f"Error. {e}")
            exit(1)
