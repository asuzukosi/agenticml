from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterator

class ToolError(Exception):
    pass

@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    schema: dict[str, Any] # openai-style function definitions


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool: {name}")

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema for name in sorted(self._tools.keys())]

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """ execute a tool by name """
        tool = self.get(name)
        try:
            return tool.fn(**args)
        except ToolError:
            raise
        except TypeError as e:
            raise ToolError(f"bad arguments to {name!r}: {e}") from e
        except Exception as e:
            raise ToolError(f"tool {name!r} raised: {type(e).__name__}: {e}") from e

    def __call__(self, name: str, args: dict[str, Any]) -> Any:
        return self.call(name, args=args)

    def __getitem__(self, name: str) -> Tool:
        return self.get(name)

    def __len__(self) -> int:
        return len(self._tools)