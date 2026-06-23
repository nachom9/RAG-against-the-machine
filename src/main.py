import ingest

def main():
    print("\n========= Hello from rag-against-the-machine! =========\n")
    ingest.get_chunks('data')
    print("\n========= Program ended =========\n")


if __name__ == "__main__":
    main()
