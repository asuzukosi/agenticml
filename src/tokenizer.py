"""
Tokenizer adapter for Telos v1.

Wraps a Llama-3.1 tokenizer and renames a contiguous range of its
reserved special tokens to Telos frame markers. The rename is real -
the underlying tokenizer's added_tokens_decoder is modified, so when
the tokenizer is saved the rename persists.
 
Token mapping (Telos v1):
 
    <|goal|>     <- <|reserved_special_token_0|>   id=128002  runtime
    <|mission|>  <- <|reserved_special_token_1|>   id=128003  runtime
    <|obs|>      <- <|reserved_special_token_2|>   id=128005  runtime
    <|belief|>   <- <|reserved_special_token_3|>   id=128011  model
    <|plan|>     <- <|reserved_special_token_4|>   id=128012  model
    <|think|>    <- <|reserved_special_token_5|>   id=128013  model
    <|action|>   <- <|reserved_special_token_6|>   id=128014  model
    <|end|>      <- <|reserved_special_token_7|>   id=128015  model
    <|result|>   <- <|reserved_special_token_8|>   id=128016  runtime
    <|feedback|> <- <|reserved_special_token_9|>   id=128017  runtime
    <|reward|>   <- <|reserved_special_token_10|>  id=128018  runtime
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from tokenizers import Tokenizer, AddedToken
from transformers import AutoTokenizer, PreTrainedTokenizerBase

TELOS_TOKEN_MAP: tuple[tuple[str, int], ...] = (
    ("<|goal|>",     0),
    ("<|mission|>",  1),
    ("<|obs|>",      2),
    ("<|belief|>",   3),
    ("<|plan|>",     4),
    ("<|think|>",    5),
    ("<|action|>",   6),
    ("<|end|>",      7),
    ("<|result|>",   8),
    ("<|feedback|>", 9),
    ("<|reward|>",   10),
)

TELOS_OWNERS: dict[str, str] = {
    "<|goal|>":     "runtime",
    "<|mission|>":  "runtime",
    "<|obs|>":      "runtime",
    "<|belief|>":   "model",
    "<|plan|>":     "model",
    "<|think|>":    "model",
    "<|action|>":   "model",
    "<|end|>":      "model",
    "<|result|>":   "runtime",
    "<|feedback|>": "runtime",
    "<|reward|>":   "runtime",
}
 
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B"

@dataclass
class TelosToken:
    """Resolved Telos marker: surface form, underlying ID, owner."""
    name: str
    token_id: int
    owner: str

class TelosTokenizer:
    """A thin wrapper around the Llama-3.1 tokenizer that renames a range
    of reserved special tokens to Telos frame markers.
 
    The rename modifies the underlying tokenizer in-place. After
    construction, the tokenizer treats e.g. ``"<|goal|>"`` as the same
    single token previously named ``"<|reserved_special_token_0|>"``.
 
    The original reserved names are not preserved - if downstream code
    needs them it should load a fresh tokenizer.
    """
 
    def __init__(self, base_tokenizer: PreTrainedTokenizerBase):
        self._tok = base_tokenizer
        self._telos_tokens: dict[str, TelosToken] = {}
        self._apply_rename()
 
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
 
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_BASE_MODEL,
        **kwargs,
    ) -> "TelosTokenizer":
        """Load a base tokenizer and apply the Telos rename."""
        base = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        return cls(base)
 
    def _apply_rename(self) -> None:
        """
        Rewrite a contiguous range of reserved-token slots to Telos names.
 
        Strategy: look up each reserved slot's current token ID, then
        register an AddedToken with the new Telos name at that same ID.
        The HuggingFace fast tokenizer accepts this and the ID is reused.
        """
        rename_specs: list[tuple[str, int]] = []
        for telos_name, slot in TELOS_TOKEN_MAP:
            source_name = f"<|reserved_special_token_{slot}|>"
            ids = self._tok.encode(source_name, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"reserved slot {slot} did not encode as a single token "
                    f"(got {ids}); base tokenizer may not be Llama-3.1"
                )
            rename_specs.append((telos_name, ids[0]))
 
        added_tokens = [
            AddedToken(
                name,
                lstrip=False,
                rstrip=False,
                single_word=False,
                normalized=False,
                special=True,
            )
            for name, _ in rename_specs
        ]
        # add_tokens reuses existing IDs when the surface form replaces an
        # existing added token - which is what we want, since the reserved
        # slots are themselves added tokens.
        self._tok.add_tokens(added_tokens, special_tokens=True)
 
        for (name, token_id) in rename_specs:
            new_id = self._tok.convert_tokens_to_ids(name)
            if new_id != token_id:
                raise RuntimeError(
                    f"rename of {name} did not preserve ID "
                    f"(expected {token_id}, got {new_id})"
                )
            self._telos_tokens[name] = TelosToken(
                name=name,
                token_id=token_id,
                owner=TELOS_OWNERS[name],
            )

    @property
    def hf(self) -> PreTrainedTokenizerBase:
        """Underlying HuggingFace tokenizer, for use with model.generate etc."""
        return self._tok
 
    @property
    def vocab_size(self) -> int:
        return len(self._tok)
 
    def token(self, name: str) -> TelosToken:
        """Return the resolved metadata for a Telos marker."""
        if name not in self._telos_tokens:
            raise KeyError(f"not a Telos marker: {name!r}")
        return self._telos_tokens[name]
 
    def id_of(self, name: str) -> int:
        """Return the token ID for a Telos marker."""
        return self.token(name).token_id
 
    def telos_token_ids(self) -> set[int]:
        """The set of all Telos marker IDs."""
        return {t.token_id for t in self._telos_tokens.values()}
 
    @property
    def end_id(self) -> int:
        """Convenience: ID of the <|end|> stop token."""
        return self.id_of("<|end|>")

    @property
    def goal_id(self) -> int:
        """Convenience: ID of the <|goal|> token."""
        return self.id_of("<|goal|>")

    @property
    def mission_id(self) -> int:
        """Convenience: ID of the <|mission|> token."""
        return self.id_of("<|mission|>")
    
    @property
    def obs_id(self) -> int:
        """Convenience: ID of the <|obs|> token."""
        return self.id_of("<|obs|>")

    @property
    def belief_id(self) -> int:
        """Convenience: ID of the <|belief|> token."""
        return self.id_of("<|belief|>")

    @property
    def plan_id(self) -> int:
        """Convenience: ID of the <|plan|> token."""
        return self.id_of("<|plan|>")

    @property
    def think_id(self) -> int:
        """Convenience: ID of the <|think|> token."""
        return self.id_of("<|think|>")

    @property
    def action_id(self) -> int:
        """Convenience: ID of the <|action|> token."""
        return self.id_of("<|action|>")

    @property
    def result_id(self) -> int:
        """Convenience: ID of the <|result|> token."""
        return self.id_of("<|result|>")

    @property
    def feedback_id(self) -> int:
        """Convenience: ID of the <|feedback|> token."""
        return self.id_of("<|feedback|>")

    @property
    def reward_id(self) -> int:
        """Convenience: ID of the <|reward|> token."""
        return self.id_of("<|reward|>")
 
    def encode(
        self,
        text: str,
        *,
    ) -> list[int]:
        """
        Encode a Telos-formatted string to token IDs.
 
        Telos markers in ``text`` are recognized as their single
        underlying token IDs.
        trajectories are self-delimiting via frame markers.
        """
        return self._tok.encode(text, add_special_tokens=False)
 
    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs back to a string.
 
        With ``skip_special_tokens=False`` (default), Telos markers appear
        as their Telos names (e.g. ``"<|goal|>"``) in the output.
        """
        return self._tok.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
 
    def describe(self) -> str:
        """Return a human-readable summary of the Telos token mapping."""
        lines = ["Telos token mapping:"]
        for telos_name, slot in TELOS_TOKEN_MAP:
            tok = self._telos_tokens[telos_name]
            lines.append(
                f"  {telos_name:<14} <- reserved_special_token_{slot:<3} "
                f"id={tok.token_id:<7} owner={tok.owner}"
            )
        return "\n".join(lines)