"""Every page lives in its own module; this file wires them into one dict.

Add a new page by writing a `Page` subclass in its own module and adding one
line below — nothing else in the app needs to change.
"""

from __future__ import annotations

from app.views.analyst import AnalystPage
from app.views.base import Page
from app.views.dashboard import DashboardPage
from app.views.monitoring import MonitoringPage
from app.views.prediction import PredictionPage
from app.views.registry import RegistryPage

PAGES: dict[str, Page] = {
    page.title: page
    for page in (
        DashboardPage(),
        AnalystPage(),
        PredictionPage(),
        RegistryPage(),
        MonitoringPage(),
    )
}

__all__ = ["PAGES", "Page"]
