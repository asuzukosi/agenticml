"""tests for telos.trajectory."""
import pytest
 
from telos.constants import FrameType
from telos.frames import Frame, action, belief, end, goal, mission
from telos.trajectory import Trajectory

def test_empty_trajectory():
    t = Trajectory()
    assert len(t) == 0
    assert list(t) == []
    assert not t # __bool__ falsy when empty

def test_construct_from_list_of_frames():
    frames = [goal("g"), mission("m")]
    t = Trajectory(frames)
    assert len(t) == 2
    assert t[0].type is FrameType.GOAL
    assert t[1].type is FrameType.MISSION


def test_construct_from_list_of_dicts():
    t = Trajectory([
        {"type": "goal", "content": "g"},
        {"type": "mission", "content": "m"},
    ])
    assert len(t) == 2
    assert t[0].type is FrameType.GOAL
    assert t[0].content == "g"

def test_construct_from_mixed_frames_and_dicts():
    t = Trajectory([
        goal("g"),
        {"type": "mission", "content": "m"},
        action({"tool": "x"}),
    ])
    assert len(t) == 3
    assert t[0].type is FrameType.GOAL
    assert t[1].type is FrameType.MISSION
    assert t[2].type is FrameType.ACTION
 
 
def test_construct_from_another_trajectory():
    a = Trajectory([goal("g")])
    b = Trajectory(a)
    assert len(b) == 1
    assert b == a
    # b is a copy, not the same list.
    b.append(mission("m"))
    assert len(a) == 1
    assert len(b) == 2

def test_construct_accepts_full_marker_names():
    t = Trajectory([{"type": "<|goal|>", "content": "g"}])
    assert t[0].type is FrameType.GOAL

def test_construct_rejects_unknown_frame_type():
    with pytest.raises(ValueError, match="unknown frame type"):
        Trajectory([{"type": "nope", "content": "x"}])

def test_to_dict_strips_marker_wrappers():
    t = Trajectory([goal("g"), action({"tool": "x"})])
    out = t.to_dict()
    assert out == [
        {"type": "goal", "content": "g"},
        {"type": "action", "content": {"tool": "x"}},
    ]

def test_to_frames_returns_fresh_list():
    t = Trajectory([goal("g")])
    a = t.to_frames()
    b = t.to_frames()
    assert a == b
    assert a is not b

def test_round_trip_dict_to_dict():
    original = [
        {"type": "goal", "content": "g"},
        {"type": "action", "content": {"tool": "x"}},
        {"type": "end", "content": None},
    ]
    t = Trajectory(original)
    assert t.to_dict() == original

def test_len_and_iteration():
    t = Trajectory([goal("g"), mission("m"), end()])
    assert len(t) == 3
    types = [f.type for f in t]
    assert types == [FrameType.GOAL, FrameType.MISSION, FrameType.END]
 
 
def test_indexing_returns_frame():
    t = Trajectory([goal("g"), mission("m")])
    assert isinstance(t[0], Frame)
    assert t[0].type is FrameType.GOAL
 
 
def test_slice_returns_trajectory():
    t = Trajectory([goal("g"), mission("m"), action({"tool": "x"}), end()])
    sub = t[1:3]
    assert isinstance(sub, Trajectory)
    assert len(sub) == 2
    assert sub[0].type is FrameType.MISSION

def test_in_membership():
    g = goal("g")
    t = Trajectory([g, mission("m")])
    assert g in t

def test_truthiness():
    assert not Trajectory()
    assert Trajectory([goal("g")])

def test_append_frame():
    t = Trajectory()
    t.append(goal("g"))
    assert len(t) == 1

def test_append_dict():
    t = Trajectory()
    t.append({"type": "mission", "content": "m"})
    assert len(t) == 1
    assert t[0].type is FrameType.MISSION

def test_extend():
    t = Trajectory()
    t.extend([goal("g"), {"type": "mission", "content": "m"}])
    assert len(t) == 2

def test_add_two_trajectories():
    a = Trajectory([goal("g")])
    b = Trajectory([mission("m")])
    c = a + b
    assert isinstance(c, Trajectory)
    assert len(c) == 2
    # Originals unchanged.
    assert len(a) == 1
    assert len(b) == 1

def test_add_trajectory_and_frame():
    t = Trajectory([goal("g")])
    new = t + mission("m")
    assert isinstance(new, Trajectory)
    assert len(new) == 2
    assert new[1].type is FrameType.MISSION

def test_add_trajectory_and_dict():
    t = Trajectory([goal("g")])
    new = t + {"type": "mission", "content": "m"}
    assert len(new) == 2
    assert new[1].type is FrameType.MISSION

def test_add_trajectory_and_list():
    t = Trajectory([goal("g")])
    new = t + [mission("m"), end()]
    assert len(new) == 3

def test_radd_frame_plus_trajectory():
    """Frame + Trajectory should also work."""
    t = Trajectory([mission("m")])
    new = goal("g") + t
    assert len(new) == 2
    assert new[0].type is FrameType.GOAL

def test_iadd_in_place():
    t = Trajectory([goal("g")])
    t += mission("m")
    assert len(t) == 2

def test_iadd_with_trajectory():
    t = Trajectory([goal("g")])
    t += Trajectory([mission("m"), end()])
    assert len(t) == 3

def test_equality():
    a = Trajectory([goal("g"), mission("m")])
    b = Trajectory([goal("g"), mission("m")])
    assert a == b

def test_inequality_with_non_trajectory():
    t = Trajectory([goal("g")])
    assert (t == [goal("g")]) is False

def test_repr_contains_trajectory():
    t = Trajectory([goal("g")])
    assert "Trajectory" in repr(t)