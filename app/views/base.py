"""Shared interface every dashboard page implements."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Page(ABC):
    """One sidebar entry. Subclasses implement `render()`."""

    #: Shown in the sidebar radio and used as the dict key in `PAGES`.
    title: str = ""

    @abstractmethod
    def render(self) -> None:
        """Draw this page's Streamlit UI. Called once per rerun."""
        raise NotImplementedError
