"""publish tasks.jsonl and tasks.parquet to hugging face hub.

usage:
    python scripts/publish_to_hf.py \
        --repo  your-hf-username/london-dynamic-routing \
        --jsonl data/tasks.jsonl \
        --parquet data/tasks.parquet

authentication: set HF_TOKEN env var or run `huggingface-cli login` first.

the repo is created as a dataset repo (private by default) if it doesn't exist.
"""
import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",    required=True,
                    help="hf repo id, e.g. your-username/london-dynamic-routing")
    ap.add_argument("--jsonl",   default="data/tasks.jsonl")
    ap.add_argument("--parquet", default="data/tasks.parquet")
    ap.add_argument("--card",    default="README_dataset.md",
                    help="dataset card to upload as README.md")
    ap.add_argument("--public",  action="store_true",
                    help="make the repo public (default: private)")
    ap.add_argument("--commit-message", default="add london-dynamic-routing dataset")
    args = ap.parse_args()

    jsonl_path   = Path(args.jsonl)
    parquet_path = Path(args.parquet)
    card_path    = Path(args.card)

    for p in (jsonl_path, parquet_path):
        if not p.exists():
            print(f"error: {p} not found. run generate_tasks.py first.", file=sys.stderr)
            sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("error: huggingface_hub not installed. run: pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)

    print(f"creating/verifying repo: {args.repo}")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
            token=token,
        )
    except Exception as e:
        print(f"error creating repo: {e}", file=sys.stderr)
        sys.exit(1)

    upload_plan = [
        (jsonl_path,   "data/tasks.jsonl"),
        (parquet_path, "data/tasks.parquet"),
    ]
    if card_path.exists():
        upload_plan.append((card_path, "README.md"))
    else:
        print(f"warning: {card_path} not found; skipping dataset card upload",
              file=sys.stderr)

    for local_path, repo_path in upload_plan:
        print(f"uploading {local_path} → {args.repo}/{repo_path} ...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=args.repo,
                repo_type="dataset",
                commit_message=args.commit_message,
                token=token,
            )
            print(f"  ok: {repo_path}")
        except Exception as e:
            print(f"  error uploading {local_path}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\ndone. dataset at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
