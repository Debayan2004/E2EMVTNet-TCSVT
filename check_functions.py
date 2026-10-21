import ast
from pathlib import Path

names = {
    "build_detection_targets",
    "encode_transcript",
    "select_recognition_sample",
    "spotting_collate_fn",
    "get_gt_centers",
    "build_proposal_targets",
}

found = {name: [] for name in names}

for path in Path(".").rglob("*.py"):
    if ".git" in path.parts or "__pycache__" in path.parts:
        continue

    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"PARSE ERROR {path}: {exc}")
        continue

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                found[node.name].append(f"{path}:{node.lineno}")

        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in names:
                    found[alias.name].append(
                        f"{path}:{node.lineno} (import from {node.module})"
                    )

for name in sorted(names):
    if found[name]:
        for location in found[name]:
            print(f"FOUND {name}: {location}")
    else:
        print(f"MISSING {name}")
