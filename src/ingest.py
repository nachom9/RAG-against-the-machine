from pathlib import Path
import bm25s
from .models import MinimalSource
import json
from src.utils import create_dir


class Parser:
    """Parses source files and creates searchable document chunks.

    This class is responsible for reading source files from a directory,
    splitting their contents into smaller chunks according to the configured
    maximum size, storing metadata about each chunk, and generating the JSON
    files and BM25 index required by the retrieval system.
    """
    def __init__(self, max_chunk_size: int):
        """Initializes the parser configuration.

        Args:
            max_chunk_size: Maximum number of characters allowed in each
                generated text chunk.
        """
        self.text_chunks: list = []
        self.max_chunk_size = max_chunk_size
        self.sources: list = []
        self.path: str = "data/processed/chunks"

    def get_chunks(self, path: str) -> None:
        """Extracts and stores chunks from all supported files in a directory.

        The method recursively searches for files inside the provided path,
        ignores unsupported or unwanted files, splits the content according to
        the file type, and stores each generated chunk together with its source
        metadata. Python files are split preferentially at class or function
        definitions, Markdown files at heading boundaries, and other files by
        simple character limits.

        Args:
            path: Directory containing the source files to parse.

        Output:
            Generates an ``all_chunks.json`` file containing the extracted
            chunks and their metadata, and creates a BM25 index used for later
            document retrieval. Prints error messages if files cannot be
            processed.
        """
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
        """Stores a generated text chunk and its metadata.

        The method creates a validated source object containing information
        about the chunk location and content, then stores both the raw text and
        metadata internally for later JSON generation and indexing.

        Args:
            content: Text content contained in the generated chunk.
            path: Original file path where the chunk was extracted from.
            first_char: Character index where the chunk starts in the original
                file.
            last_char: Character index where the chunk ends in the original
                file.

        Output:
            Adds the chunk content and its metadata to the parser internal
            collections.
        """

        chunk = MinimalSource(
            file_path=path,
            first_character_index=first_char,
            last_character_index=last_char,
            text=content
        )
        self.text_chunks.append(content)
        self.sources.append(chunk)

    def generate_json(self) -> None:
        """Generates the chunk metadata file and creates the BM25 index.

        The method saves all extracted chunks as a JSON file containing the
        source metadata, tokenizes the chunk contents, builds a BM25 retrieval
        index, and stores the index on disk for future searches.

        Output:
            Creates an ``all_chunks.json`` file containing all generated
            chunks and a BM25 index directory used by the retrieval system.
            Prints an error message and exits if the index cannot be saved.
        """
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
