"""Utilities for grouping mutable instance state behind a stable public API."""

from types import SimpleNamespace
from typing import ClassVar, FrozenSet


class StateProxy:
    """Proxy selected public attributes to a compact internal state object."""

    _STATE_ATTRS: ClassVar[FrozenSet[str]] = frozenset()

    @classmethod
    def _state_attrs(cls) -> FrozenSet[str]:
        attrs = set()
        for klass in cls.__mro__:
            attrs.update(getattr(klass, "_STATE_ATTRS", ()))
        return frozenset(attrs)

    def _init_state_proxy(self) -> None:
        object.__setattr__(self, "_state", SimpleNamespace())

    def __getattr__(self, name):
        if name in self._state_attrs():
            state = object.__getattribute__(self, "_state")
            try:
                return getattr(state, name)
            except AttributeError as exc:
                raise AttributeError(name) from exc
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._state_attrs() and "_state" in self.__dict__:
            setattr(self._state, name, value)
            return
        super().__setattr__(name, value)
