import fire
from src.cli import RAGApplication


def main() -> None:
    fire.Fire(RAGApplication)


if __name__ == "__main__":
    main()
