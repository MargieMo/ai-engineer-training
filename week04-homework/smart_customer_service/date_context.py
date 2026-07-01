"""Shared helpers for the current calendar date."""
from datetime import datetime


def today():
    """Return today's date in the local timezone."""
    return datetime.now().date()


def today_str() -> str:
    """Return today as YYYY-MM-DD."""
    return today().strftime("%Y-%m-%d")
