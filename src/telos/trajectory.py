"""
trajectory:: an ordered, mutable container of telos frames
"""
from __future__ import annotations
from typing import Any, Iterable, Iterator, Union
from telos.constants import FrameType
from telos.frames import Frame

FrameLike = Union[Frame, dict[str, Any]]

def _frame_from_dict(d: dict[str, Any]) -> Frame:
    """ convert a public-api dict a frame object. """
    if "type" not in d:
        raise ValueError(f"frame dict missing 'type' key: {d}")
    type_str = d["type"]
    if not type_str.startswith("<|"):
        type_str = "<|" + type_str + "|>"
    try:
        ft = FrameType(type_str)
    except ValueError as e:
        raise ValueError(f"unknown frame type: {type_str}") from e
    return Frame(type=ft, content=d.get("content", None))

def _frame_to_dict(f: Frame) -> dict:
    """Convert a Frame to a public-API dict.
 
    The full marker (``<|goal|>``) is stripped to the short name
    (``goal``) for friendlier external representations.
    """
    short_name = f.type.value[2:-2]
    return {"type": short_name, "content": f.content}

def _coerce_frame(item: FrameLike) -> Frame:
    """convert a frame-like object to a frame object."""
    if isinstance(item, Frame):
        return item
    return _frame_from_dict(item)


class Trajectory:
    """an ordered, mutable sequence of Telos frames."""
 
    __slots__ = ("_frames",)
 
    def __init__(
        self,
        frames: Union["Trajectory", Iterable[FrameLike], None] = None,
    ):
        if frames is None:
            self._frames: list[Frame] = []
        elif isinstance(frames, Trajectory):
            self._frames = list(frames._frames)
        else:
            self._frames = [_coerce_frame(f) for f in frames]

    def to_dict(self) -> list[dict]:
        """a list of frame dicts suitable for JSON serialization."""
        return [_frame_to_dict(f) for f in self._frames]
 
    def to_frames(self) -> list[Frame]:
        """a fresh list of the underlying Frame objects."""
        return list(self._frames)

    def __len__(self) -> int:
        """the number of frames in the trajectory."""
        return len(self._frames)

    def __getitem__(self, index: Union[int, slice]) -> Union[Frame, Trajectory]:
        """the frame at the given index."""
        if isinstance(index, slice):
            return Trajectory(self._frames[index])
        return self._frames[index]

    def __contains__(self, item) -> bool:
        return item in self._frames
 
    def __bool__(self) -> bool:
        return bool(self._frames)

    def append(self, item: FrameLike) -> None:
        """append a Frame or a frame-dict to the trajectory."""
        self._frames.append(_coerce_frame(item))
 
    def extend(self, items: Iterable[FrameLike]) -> None:
        """append multiple frames or frame-dicts."""
        for item in items:
            self.append(item)

    def __add__(self, other) -> "Trajectory":
        """concatenate with another Trajectory, list, or single frame.
        returns a new Trajectory; does not mutate self.
        """
        new = Trajectory(self)
        if isinstance(other, Trajectory):
            new._frames.extend(other._frames)
        elif isinstance(other, Frame) or isinstance(other, dict):
            new._frames.append(_coerce_frame(other))
        elif isinstance(other, list):
            new._frames.extend(_coerce_frame(f) for f in other)
        else:
            return NotImplemented
        return new
 
    def __radd__(self, other) -> "Trajectory":
        """allow `list_of_frames + trajectory` and `frame + trajectory`."""
        if isinstance(other, Frame) or isinstance(other, dict):
            return Trajectory([other]) + self
        if isinstance(other, list):
            return Trajectory(other) + self
        return NotImplemented
 
    def __iadd__(self, other) -> "Trajectory":
        """in-place addition via `trajectory += other`."""
        if isinstance(other, Trajectory):
            self._frames.extend(other._frames)
        elif isinstance(other, Frame) or isinstance(other, dict):
            self._frames.append(_coerce_frame(other))
        elif isinstance(other, list):
            self._frames.extend(_coerce_frame(f) for f in other)
        else:
            return NotImplemented
        return self
    
    def __eq__(self, other) -> bool:
        """equality comparison."""
        if isinstance(other, Trajectory):
            return self._frames == other._frames
        return NotImplemented
 
    def __repr__(self) -> str:
        """a string representation of the trajectory."""
        return f"Trajectory({self._frames!r})"