from src.ingest import Parser

class RAGAplication:

    def index():
        parser = Parser()
        print("Ingestion on process...")
        parser.get_chunks('vllm-0.10.1')
        print("Ingestion complete! Indices saved under data/processed/")

def main() -> None:
    print("\n========= Hello from rag-against-the-machine! =========\n")
    RAGAplication.index()
    print("\n========= Program ended =========\n")
