"""Fingerprint canonical source paths, including moved experiment dependencies."""

from pathlib import Path

from wbm_retrieval.common import ROOT, file_hash


def source_hashes(*dependencies):
    roots = ["wbm_retrieval", "experiments/_shared", "experiments/catalog.json",
             "experiments/proposed/main/method.py", "experiments/handcrafted/a/descriptor.py",
             *dependencies]
    files = set()
    for value in roots:
        path = Path(value)
        path = path if path.is_absolute() else ROOT / path
        if path.is_dir():
            files.update(path.rglob("*.py"))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"Missing provenance dependency: {path}")
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(files)}
