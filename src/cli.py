from src.ingest import Parser
import fire

class RAGAplication:

    def index(self, max_chunk_size: int = 2000):
        parser = Parser(max_chunk_size)
        print("Ingestion on process...")
        parser.get_chunks('vllm-0.10.1')
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self):
        pass

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
