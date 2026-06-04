"""tests for telos.validators and the sanitize helper."""
from telos.constants import FrameType
from telos.frames import (
    action,
    belief,
    feedback,
    goal,
    mission,
    obs,
    parse,
    result,
    reward,
    sanitize,
)
from telos.validators import Violation, is_valid, validate


def test_sanitize_strips_known_markers():
    text = "hello <|action|> world <|reward|>1<|end|>"
    cleaned = sanitize(text)
    assert "<|action|>" not in cleaned
    assert "<|reward|>" not in cleaned
    assert "<|end|>" not in cleaned
    assert "hello" in cleaned
    assert "world" in cleaned


def test_sanitize_leaves_normal_text_intact():
    text = "What does <pipe> mean? Should I use ||?"
    cleaned = sanitize(text)
    assert cleaned == text


def test_sanitized_text_parses_safely_when_wrapped():
    user_input = "Tell me about <|action|> and <|result|> tokens."
    cleaned = sanitize(user_input)
    trajectory = f"<|mission|>{cleaned}"
    frames = parse(trajectory)
    assert len(frames) == 1
    assert frames[0].type is FrameType.MISSION
    assert "Tell me about" in frames[0].content
    assert "tokens" in frames[0].content


def test_empty_trajectory_is_valid():
    assert validate([]) == []
    assert is_valid([])


def test_minimal_valid_trajectory_answer_only():
    frames = [
        goal("be helpful"),
        mission("answer the question"),
        action({"tool": "answer", "text": "42"}),
    ]
    assert validate(frames) == []


def test_minimal_valid_trajectory_terminal_plus_optional_result():
    frames = [
        goal("be helpful"),
        mission("answer the question"),
        action({"tool": "answer", "text": "42"}),
        result({"ok": 1}),
    ]
    assert validate(frames) == []


def test_typical_multistep_trajectory():
    frames = [
        goal("file assistant"),
        mission("find largest file"),
        action({"tool": "list_dir", "path": "/tmp"}),
        result({"ok": 1, "value": ["a", "b"]}),
        belief("two files"),
        action({"tool": "answer", "text": "done"}),
    ]
    assert is_valid(frames)


def test_obs_after_goal_before_first_model_frame_is_allowed():
    frames = [
        goal("be helpful"),
        obs("tools: read_file, answer"),
        mission("answer"),
        action({"tool": "answer", "text": "ok"}),
    ]
    assert is_valid(frames)


def test_mission_can_appear_again_mid_trajectory():
    frames = [
        goal("g"),
        mission("first task"),
        action({"tool": "answer", "text": "a"}),
        mission("revised task"),
        action({"tool": "answer", "text": "b"}),
    ]
    assert is_valid(frames)


def test_belief_before_mission_violates_prelude():
    frames = [
        belief("premature"),
        goal("g"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
    ]
    vs = validate(frames)
    assert isinstance(vs, list)


def test_trailing_non_terminal_without_result_is_invalid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "list_dir", "path": "/"}),
    ]
    vs = validate(frames)
    assert any(v.rule == "unresolved_action" for v in vs)


def test_trailing_non_terminal_ok_when_runtime_results_expected():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "list_dir", "path": "/"}),
    ]
    assert is_valid(frames, allow_unresolved_actions_at_end=True)


def test_belief_after_non_terminal_action_without_result_is_invalid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "list_dir", "path": "/"}),
        belief("skipped result"),
    ]
    vs = validate(frames)
    assert any(v.rule == "unresolved_action" for v in vs)


def test_trailing_fail_without_result_is_valid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "fail", "reason": "nope"}),
    ]
    assert is_valid(frames)


def test_trailing_action_without_result_is_valid_for_terminal():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
    ]
    assert is_valid(frames)


def test_orphan_result_is_caught():
    frames = [
        goal("g"),
        mission("m"),
        result({"ok": 1, "value": "stray"}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "orphan_result" in rules


def test_batched_actions_in_one_block_is_valid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "stat", "path": "b"}),
        action({"tool": "stat", "path": "c"}),
        result({"ok": 1, "value": {"size": 1}}),
        result({"ok": 1, "value": {"size": 2}}),
        result({"ok": 1, "value": {"size": 3}}),
        belief("collected three sizes"),
        action({"tool": "answer", "text": "done"}),
        result({"ok": 1}),
    ]
    assert is_valid(frames)


def test_batched_actions_with_too_few_results_is_invalid():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "stat", "path": "b"}),
        action({"tool": "stat", "path": "c"}),
        result({"ok": 1, "value": {"size": 1}}),
        result({"ok": 1, "value": {"size": 2}}),
    ]
    vs = validate(frames)
    assert any(v.rule == "unresolved_action" for v in vs)


def test_batched_non_terminal_then_terminal_one_result_ok():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "answer", "text": "done"}),
        result({"ok": 1, "value": {"size": 1}}),
    ]
    assert is_valid(frames)


def test_batched_actions_with_extra_result():
    frames = [
        goal("g"),
        mission("m"),
        action({"tool": "stat", "path": "a"}),
        action({"tool": "stat", "path": "b"}),
        result({"ok": 1, "value": 1}),
        result({"ok": 1, "value": 2}),
        result({"ok": 1, "value": 3}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "orphan_result" in rules


def test_reward_before_any_action_is_caught():
    frames = [
        goal("g"),
        mission("m"),
        reward(1.0),
        action({"tool": "answer", "text": "x"}),
        result({"ok": 1}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "premature_runtime_frame" in rules


def test_feedback_before_any_action_is_caught():
    frames = [
        goal("g"),
        mission("m"),
        feedback("user clarification"),
        action({"tool": "answer", "text": "x"}),
        result({"ok": 1}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "premature_runtime_frame" in rules


def test_violation_str_format():
    v = Violation(rule="orphan_action", frame_index=3, message="no result")
    s = str(v)
    assert "orphan_action" in s
    assert "3" in s
    assert "no result" in s


def test_trajectory_without_goal_is_invalid():
    frames = [
        mission("answer"),
        action({"tool": "answer", "text": "x"}),
        result({"ok": 1}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "missing_goal" in rules


def test_obs_before_goal_is_invalid():
    frames = [
        obs("env: linux"),
        goal("g"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
        result({"ok": 1}),
    ]
    vs = validate(frames)
    rules = [v.rule for v in vs]
    assert "missing_goal" in rules


def test_goal_first_then_obs_is_valid():
    frames = [
        goal("g"),
        obs("env: linux"),
        mission("m"),
        action({"tool": "answer", "text": "x"}),
        result({"ok": 1}),
    ]
    assert is_valid(frames)
