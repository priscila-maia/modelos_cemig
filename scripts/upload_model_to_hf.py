"""Upload local model folders to a Hugging Face model repo."""

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi

from src.core.config import env_str, load_project_env


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp_v2_40k"
    / "models"
    / "qwen3_embedding_0_6b_ft_v2"
)
ENV_FILE = PROJECT_ROOT / ".env"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Upload a local model folder to a Hugging Face repo subdirectory"
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_DIR),
        help="Local model directory. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--repo-id",
        default="CemigP/modelos_cemig",
        help="Target Hugging Face model repo id.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Subdirectory inside the repo. Defaults to the local folder name.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch or revision inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom commit message for the upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or local hf login.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public when it does not exist yet.",
    )
    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="Skip local validation for sentence-transformers folders before upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the model folder and print the resolved upload target without uploading.",
    )
    return parser.parse_args(argv)


def resolve_model_path(raw_path: str) -> Path:
    model_path = Path(raw_path).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve()


def validate_model_dir(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")


def ensure_sentence_transformer_layout(model_path: Path) -> None:
    modules_file = model_path / "modules.json"
    if not modules_file.exists():
        return

    pooling_config = model_path / "1_Pooling" / "config.json"
    if not pooling_config.exists():
        raise FileNotFoundError(
            "Sentence-transformers model is missing 1_Pooling/config.json"
        )

    # Normalize modules in sentence-transformers do not persist files, but keeping the
    # directory locally makes the saved folder match modules.json more closely.
    (model_path / "2_Normalize").mkdir(exist_ok=True)


def validate_sentence_transformer_load(model_path: Path) -> None:
    if not (model_path / "modules.json").exists():
        return

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device="cpu", trust_remote_code=True)
    embeddings = model.encode(["teste curto"], show_progress_bar=False)
    shape = getattr(embeddings, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != 1:
        raise RuntimeError(
            f"Unexpected embedding output while validating model at {model_path}: {shape}"
        )


def build_commit_message(
    model_path: Path, path_in_repo: str, custom_message: Optional[str]
) -> str:
    if custom_message:
        return custom_message
    return f"Upload model '{model_path.name}' to '{path_in_repo}'"


def upload_model_folder(args) -> str:
    load_project_env(ENV_FILE)

    model_path = resolve_model_path(args.model_path)
    path_in_repo = args.path_in_repo or model_path.name
    token = args.token or env_str("HF_TOKEN", "") or None

    validate_model_dir(model_path)
    ensure_sentence_transformer_layout(model_path)

    if not args.skip_load_check:
        validate_sentence_transformer_load(model_path)

    if args.dry_run:
        return "dry-run completed"

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )

    commit_message = build_commit_message(model_path, path_in_repo, args.commit_message)
    commit_info = api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        revision=args.revision,
        commit_message=commit_message,
        ignore_patterns=[".DS_Store"],
    )

    return getattr(commit_info, "commit_url", None) or getattr(
        commit_info, "oid", "upload completed"
    )


def main(argv=None):
    args = parse_args(argv)
    commit_ref = upload_model_folder(args)
    print(f"Repo: {args.repo_id}")
    print(f"Model path: {resolve_model_path(args.model_path)}")
    print(f"Path in repo: {args.path_in_repo or Path(args.model_path).name}")
    print(f"Result: {commit_ref}")


if __name__ == "__main__":
    main()
