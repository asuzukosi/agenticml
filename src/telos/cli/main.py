from __future__ import annotations

import sys
from collections.abc import Callable

from telos.cli.commands import (
    data_clean_push,
    data_synthetic_gen,
    data_telos_to_chatml,
    eval_format_validity,
    initialize_chatml_embeddings,
    initialize_telos_embeddings,
    verify_telos_embeddings,
    train_chatml_lora,
    train_telos_lora,
)

COMMANDS: dict[str, Callable[[list[str] | None], None]] = {
    "train-telos-lora": train_telos_lora.main,
    "train-chatml-lora": train_chatml_lora.main,
    "eval-format-validity": eval_format_validity.main,
    "init-telos-embeddings": initialize_telos_embeddings.main,
    "verify-telos-embeddings": verify_telos_embeddings.main,
    "init-chatml-embeddings": initialize_chatml_embeddings.main,
    "data-clean-push": data_clean_push.main,
    "data-telos-to-chatml": data_telos_to_chatml.main,
    "data-synthetic-gen": data_synthetic_gen.main,
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: telos <command> [args...]")
        print("commands:", ", ".join(sorted(COMMANDS)))
        return
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"unknown command: {name}")
        print("commands:", ", ".join(sorted(COMMANDS)))
        sys.exit(1)
    COMMANDS[name](rest)
