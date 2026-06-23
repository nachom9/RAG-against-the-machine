from pathlib import Path
from os import listdir
from os.path import isfile, join


def get_chunks(path: str):
    files = [str(f) for f in Path(path).rglob("*") if f.is_file()]
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            print(content)



def create_dir(path: str):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print("Error. Permission denied while creating the directory '{path}'")
        exit(1)
    except OSError as e:
        print(f"Error. Failed to create the directory '{path}': {e}")
        exit(1)
