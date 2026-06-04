from __future__ import annotations

import argparse
import sys

from telos.constants import DEFAULT_BASE_MODEL
from telos.model_init.verify_telos_embeddings import run_verify_telos_embeddings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="verify telos reserved-token embeddings match seed-word means",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help="checkpoint to verify (default: telos init hub model)",
    )
    args = parser.parse_args(argv)

    try:
        run_verify_telos_embeddings(args.model)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
