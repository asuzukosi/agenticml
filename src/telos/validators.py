from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
from telos.constants import TERMINAL_TOOLS, FrameType, FrameOwner
from telos.frames import Frame


@dataclass
class Violation:
    """a single trajectory-level rule violation."""
    rule: str
    frame_index: int
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] frame {self.frame_index}: {self.message}"


def _action_tool(f: Frame) -> Optional[str]:
    if f.type is not FrameType.ACTION:
        return None
    return (f.content or {}).get("tool")


def _action_is_terminal(f: Frame) -> bool:
    t = _action_tool(f)
    return t is not None and t in TERMINAL_TOOLS


# ---- structural rules (pure, frame-list scans) ----

def _violations_goal_must_be_first(frames: list[Frame]) -> list[Violation]:
    if frames[0].type is not FrameType.GOAL:
        return [Violation(
            rule="missing_goal",
            frame_index=0,
            message=f"trajectory must begin with <|goal|>, got {frames[0].type.value}",
        )]
    return []


def _violations_opening_prelude(frames: list[Frame]) -> list[Violation]:
    """before the first model-owned frame, only goal / mission / obs are allowed."""
    allowed = {FrameType.GOAL, FrameType.MISSION, FrameType.OBS}
    out: list[Violation] = []
    for i, f in enumerate(frames):
        if f.owner == FrameOwner.MODEL:
            break
        if f.type not in allowed:
            out.append(Violation(
                rule="opening_prelude",
                frame_index=i,
                message=(
                    f"unexpected {f.type.value} before any model frame "
                    f"(expected goal/mission/obs)"
                ),
            ))
    return out

@dataclass
class _ActionResultWalk:
    """tracks non-terminal <|action|> vs <|result|> pairing while scanning."""
    frames: list[Frame]
    pending_non_terminal: int = 0
    first_pending_action_idx: Optional[int] = None
    saw_any_action: bool = False
    in_model_block: bool = False
    violations: list[Violation] = field(default_factory=list)

    def _set_model_block(self, f: Frame) -> None:
        if f.owner == FrameOwner.MODEL:
            if not self.in_model_block:
                self.in_model_block = True
        else:
            self.in_model_block = False

    def _on_action(self, i: int, f: Frame) -> None:
        if self.pending_non_terminal > 0 and not self.in_model_block:
            self.violations.append(Violation(
                rule="orphan_action",
                frame_index=self.first_pending_action_idx or i,
                message="non-terminal action emitted while previous batch unresolved",
            ))
        if _action_is_terminal(f):
            self.saw_any_action = True
            return
        self.saw_any_action = True
        if self.pending_non_terminal == 0:
            self.first_pending_action_idx = i
        self.pending_non_terminal += 1

    def _on_result(self, i: int, f: Frame) -> None:
        if self.pending_non_terminal > 0:
            self.pending_non_terminal -= 1
            if self.pending_non_terminal == 0:
                self.first_pending_action_idx = None
            return
        prev = self.frames[i - 1] if i > 0 else None
        if (
            prev is not None
            and prev.type is FrameType.ACTION
            and _action_is_terminal(prev)
        ):
            return
        self.violations.append(Violation(
            rule="orphan_result",
            frame_index=i,
            message="result with no preceding unresolved non-terminal action",
        ))

    def _on_feedback_or_reward(self, i: int, f: Frame) -> None:
        if not self.saw_any_action:
            self.violations.append(Violation(
                rule="premature_runtime_frame",
                frame_index=i,
                message=f"{f.type.value} appears before any action",
            ))

    def _on_model_step(self, i: int, f: Frame) -> None:
        if self.pending_non_terminal > 0 and f.type in (
            FrameType.BELIEF,
            FrameType.PLAN,
            FrameType.THINK,
        ):
            self.violations.append(Violation(
                rule="unresolved_action",
                frame_index=self.first_pending_action_idx or i,
                message=(
                    "non-terminal action(s) not followed by <|result|> before "
                    f"next {f.type.value}"
                ),
            ))

    def run(self, *, allow_unresolved_actions_at_end: bool = False) -> list[Violation]:
        for i, f in enumerate(self.frames):
            self._set_model_block(f)
            if f.type is FrameType.ACTION:
                self._on_action(i, f)
            elif f.type is FrameType.RESULT:
                self._on_result(i, f)
            elif f.type in (FrameType.FEEDBACK, FrameType.REWARD):
                self._on_feedback_or_reward(i, f)
            elif f.type in (FrameType.BELIEF, FrameType.PLAN, FrameType.THINK):
                self._on_model_step(i, f)
        if self.pending_non_terminal > 0 and not allow_unresolved_actions_at_end:
            self.violations.append(Violation(
                rule="unresolved_action",
                frame_index=self.first_pending_action_idx or (len(self.frames) - 1),
                message=(
                    f"{self.pending_non_terminal} non-terminal action(s) without "
                    f"matching result(s) by end of trajectory"
                ),
            ))
        return self.violations


# extend validation by adding callables: (frames) -> list[Violation]
StructuralRule = Callable[[list[Frame]], list[Violation]]

STRUCTURAL_RULES: tuple[StructuralRule, ...] = (
    _violations_goal_must_be_first,
    _violations_opening_prelude,
)


def validate_for_model_generation(frames: list[Frame]) -> list[Violation]:
    """validate prelude + model-generated suffix (format-validity eval).

    wire training pattern: <|action|>{...}<|end|> then <|result|>{...} from runtime.
    the model learns through <|end|>; results are not generated. trailing
    non-terminal actions without a <|result|> frame in the list are allowed.
    """
    return validate(frames, allow_unresolved_actions_at_end=True)


def validate(
    frames: list[Frame],
    *,
    allow_unresolved_actions_at_end: bool = False,
) -> list[Violation]:
    """check a trajectory against telos sequence rules.

    on the wire, <|end|> closes a model turn; <|result|> is runtime-owned and
    usually appears in the next segment (see render() in frames.py). parsed
    frame lists are action then result with no end frame between them.

    non-terminal actions need matching <|result|> frame(s) before the next model
    step (belief/plan/think/action) or trajectory end. terminal actions (answer /
    fail) do not require a result.

    allow_unresolved_actions_at_end: for eval on partial trajectories where the
    model just emitted action<|end|> and runtime has not appended results yet.
    use validate_for_model_generation() instead of passing the flag directly.
    """
    if not frames:
        return []

    violations: list[Violation] = []
    for rule in STRUCTURAL_RULES:
        violations.extend(rule(frames))

    violations.extend(
        _ActionResultWalk(frames).run(
            allow_unresolved_actions_at_end=allow_unresolved_actions_at_end,
        )
    )
    return violations


def is_valid(
    frames: list[Frame],
    *,
    allow_unresolved_actions_at_end: bool = False,
) -> bool:
    """convenience: true iff validate returns no violations. used for testing."""
    return not validate(
        frames, allow_unresolved_actions_at_end=allow_unresolved_actions_at_end
    )
