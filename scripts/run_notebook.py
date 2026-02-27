from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    nb_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/summary_pipeline.ipynb")
    nb_path = nb_arg if nb_arg.is_absolute() else Path(nb_arg)
    if not nb_path.exists() and nb_path.suffix == "":
        alt = nb_path.with_suffix(".ipynb")
        if alt.exists():
            nb_path = alt
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    try:
        from nbclient import NotebookClient
        from nbformat import read, write
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("nbclient/nbformat are required for notebook smoke test. Install them in environment.") from exc

    with nb_path.open("r", encoding="utf-8") as f:
        nb = read(f, as_version=4)

    client = NotebookClient(nb, timeout=1200, kernel_name="python3")
    client.execute(cwd=str(repo_root))

    out = Path("outputs") / "summary_pipeline" / "latest" / "data" / "executed_notebook.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        write(nb, f)
    print(json.dumps({"executed": str(nb_path), "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
