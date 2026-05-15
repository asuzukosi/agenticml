"""
tokenizer adapter for telos v1.

reserved special tokens to telos frame markers. The rename is real -
the underlying tokenizer's added_tokens_decoder is modified, so when
the tokenizer is saved the rename persists.
 
token mapping (telos v1):
 
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
from typing import Any, Iterable, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from telos.constants import TELOS_TOKEN_MAP, TELOS_OWNERS, DEFAULT_BASE_MODEL
from telos.constants import FrameType, FrameOwner
from telos.frames import render
from telos.trajectory import FrameLike, Trajectory

def _reserved_name(slot: int) -> str:
    return f"<|reserved_special_token_{slot}|>"
 

@dataclass
class TelosToken:
    """resolved Telos frame type: name, underlying ID, owner."""
    name: FrameType
    token_id: int
    owner: FrameOwner

class TelosTokenizer:
    """wraps a llama-3.1 tokenizer and aliases telos frame markers to its
    reserved special tokens.
 
    The underlying huggingface tokenizer is not modified. Aliasing is
    performed at the string level: ``encode`` substitutes telos names
    for their underlying reserved names before tokenization, and
    ``decode`` substitutes them back.
    """
 
    def __init__(self, base_tokenizer: PreTrainedTokenizerBase):
        self._tok = base_tokenizer
        self._telos_tokens: dict[str, TelosToken] = {}
        self._encode_subs: list[tuple[FrameType, str]] = []  # (frame_type, reserved_name)
        self._decode_subs: list[tuple[str, FrameType]] = []  # (reserved_name, frame_type)
        self._build_alias_table()
 
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = DEFAULT_BASE_MODEL,
        **kwargs,
    ) -> "TelosTokenizer":
        """load a base tokenizer and apply telos aliasing."""
        base = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        return cls(base)
 
    def _build_alias_table(self) -> None:
        """resolve each telos marker to a reserved token ID and record substitutions."""
        for telos_name, slot in TELOS_TOKEN_MAP:
            reserved_name = _reserved_name(slot)
            ids = self._tok.encode(reserved_name, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"reserved slot {slot} did not encode as a single token "
                    f"(got {ids}); base tokenizer may not be Llama-3.1-compatible"
                )
            token_id = ids[0]
            self._telos_tokens[telos_name] = TelosToken(
                name=telos_name,
                token_id=token_id,
                owner=TELOS_OWNERS[telos_name],
            )
            self._encode_subs.append((FrameType(telos_name), reserved_name))
            self._decode_subs.append((reserved_name, FrameType(telos_name)))
 
    @property
    def hf(self) -> PreTrainedTokenizerBase:
        """underlying huggingface tokenizer."""
        return self._tok
 
    @property
    def vocab_size(self) -> int:
        return len(self._tok)
 
    def token(self, name: str) -> TelosToken:
        """get the TelosToken for a given frame type name."""
        try:
            return self._telos_tokens[FrameType(name)]
        except KeyError:
            raise KeyError(f"not a Telos frame type: {name!r}")
        except ValueError:
            raise KeyError(f"not a Telos frame type: {name!r}")
 
    def id_of(self, name: str) -> int:
        return self.token(name).token_id
 
    def telos_token_ids(self) -> set[int]:
        return {t.token_id for t in self._telos_tokens.values()}
 

    @property
    def goal_id(self) -> int:
        """id of the <|goal|> start token, for use as a generation start."""
        return self.id_of(FrameType.GOAL)
    
    @property
    def mission_id(self) -> int:
        """id of the <|mission|> start token, for use as a generation start."""
        return self.id_of(FrameType.MISSION)
    
    @property
    def obs_id(self) -> int:
        """id of the <|obs|> start token, for use as a generation start."""
        return self.id_of(FrameType.OBS)
    
    @property
    def belief_id(self) -> int:
        """id of the <|belief|> start token, for use as a generation start."""
        return self.id_of(FrameType.BELIEF)
    
    @property
    def plan_id(self) -> int:
        """id of the <|plan|> start token, for use as a generation start."""
        return self.id_of(FrameType.PLAN)
    
    @property
    def think_id(self) -> int:
        """id of the <|think|> start token, for use as a generation start."""
        return self.id_of(FrameType.THINK)
    
    @property
    def action_id(self) -> int:
        """id of the <|action|> start token, for use as a generation start."""
        return self.id_of(FrameType.ACTION)
    
    @property
    def end_id(self) -> int:
        """id of the <|end|> stop token, for use as a generation stop."""
        return self.id_of(FrameType.END)

    @property
    def result_id(self) -> int:
        """id of the <|result|> start token, for use as a generation start."""
        return self.id_of(FrameType.RESULT)
    
    @property
    def feedback_id(self) -> int:
        """id of the <|feedback|> start token, for use as a generation start."""
        return self.id_of(FrameType.FEEDBACK)
    
    @property
    def reward_id(self) -> int:
        """id of the <|reward|> start token, for use as a generation start."""
        return self.id_of(FrameType.REWARD)

 
    def encode(self, text: str) -> list[int]:
        """encode a telos-formatted string to token IDs.
 
        telos markers in ``text`` are rewritten to their underlying
        reserved-token names before tokenization, so they encode as
        single token IDs. BOS is never added - telos trajectories are
        self-delimiting via frame markers.
        """
        rewritten = text
        for telos_name, reserved_name in self._encode_subs:
            if telos_name in rewritten:
                rewritten = rewritten.replace(telos_name, reserved_name)
        return self._tok.encode(rewritten, add_special_tokens=False)
 
    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """decode token IDs back to a telos-formatted string.
        reserved-token names in the decoded output are rewritten to
        their telos aliases.
        """
        text = self._tok.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
        if skip_special_tokens:
            return text
        for reserved_name, telos_name in self._decode_subs:
            if reserved_name in text:
                text = text.replace(reserved_name, telos_name)
        return text
 
    def describe(self) -> str:
        lines = ["Telos token aliases:"]
        for telos_name, slot in TELOS_TOKEN_MAP:
            tok = self._telos_tokens[telos_name]
            lines.append(
                f"  {telos_name.value:<14} -> reserved_special_token_{slot:<3} "
                f"id={tok.token_id:<7} owner={tok.owner}"
            )
        return "\n".join(lines)

    def apply_trajectory_template(
        self,
        trajectory: Union[Trajectory, Iterable[FrameLike]],
        *,
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs: dict[str, Any],
    ) -> Union[str, list[int], Any]:
        """render a trajectory to telos wire text, then optionally tokenize."""
        if not isinstance(trajectory, Trajectory):
            trajectory = Trajectory(trajectory)
        text = render(trajectory.to_frames())
        if not tokenize:
            if return_tensors is not None:
                raise ValueError("return_tensors is only valid when tokenize=True")
            return text
        ids = self.encode(text)
        if return_tensors is None:
            return ids
        if return_tensors == "pt":
            import torch

            return torch.tensor([ids])
        if return_tensors == "np":
            import numpy as np

            return np.array([ids])
        raise ValueError(f"unsupported return_tensors: {return_tensors!r}")