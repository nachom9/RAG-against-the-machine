import fire
from src.cli import RAGApplication


def main() -> None:
    """Starts the RAG application command-line interface.

    The function initializes the Fire command-line interface using the
    ``RAGApplication`` class, exposing its methods as executable commands from
    the terminal.

    Output:
        Starts the command-line interface and allows users to execute the
        available RAG application operations through command-line arguments.
    """
    fire.Fire(RAGApplication)


if __name__ == "__main__":
    main()
