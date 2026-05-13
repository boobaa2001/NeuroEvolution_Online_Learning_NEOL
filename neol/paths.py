import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def result_dir(benchmark_name: str) -> str:
    path = os.path.join(OUTPUT_DIR, benchmark_name)
    os.makedirs(path, exist_ok=True)
    return path

