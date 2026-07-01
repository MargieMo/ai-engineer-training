from datetime import date, timedelta

from langchain.tools import tool

from ..date_context import today

# Weekday labels used when the user asks "what day is it?" style questions.
_WEEKDAY_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

_WEEKDAY_MAP = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _format_with_weekday(target: date) -> str:
    """Return YYYY-MM-DD plus weekday names in English and Chinese."""
    wd = target.weekday()
    return f"{target.strftime('%Y-%m-%d')} ({_WEEKDAY_EN[wd]}, {_WEEKDAY_CN[wd]})"


def _find_weekday_index(text: str) -> int:
    for name, index in _WEEKDAY_MAP.items():
        if name in text:
            return index
    return -1


def resolve_relative_time_text(relative_time_str: str, anchor: date | None = None) -> str:
    """Resolve relative time phrases to a date string using *anchor* (defaults to today).

    Supported expressions include:
    - 昨天 / yesterday
    - 前天 / the day before yesterday
    - 今天 / today
    - 明天 / tomorrow
    - 后天 / the day after tomorrow
    - 今天星期几 / what day is today  -> date + weekday name
    - 上周三 / last Wednesday
    - 这周三 / this Wednesday
    """
    anchor_date = anchor or today()
    text = relative_time_str.lower()

    # "What day is today?" style questions.
    if any(kw in text for kw in ["星期几", "周几", "what day", "which day"]):
        return _format_with_weekday(anchor_date)

    if "昨天" in text or "yesterday" in text:
        target_date = anchor_date - timedelta(days=1)
    elif "前天" in text or "day before yesterday" in text:
        target_date = anchor_date - timedelta(days=2)
    elif "明天" in text or "tomorrow" in text:
        target_date = anchor_date + timedelta(days=1)
    elif "后天" in text or "day after tomorrow" in text:
        target_date = anchor_date + timedelta(days=2)
    elif "今天" in text or "today" in text:
        target_date = anchor_date
    elif "上周" in text or "last week" in text:
        target_weekday = _find_weekday_index(text)
        if target_weekday == -1:
            return "Could not recognize the weekday information."
        days_ago = (anchor_date.weekday() - target_weekday + 7) % 7 + 7
        target_date = anchor_date - timedelta(days=days_ago)
    elif "这周" in text or "this week" in text:
        target_weekday = _find_weekday_index(text)
        if target_weekday == -1:
            return "Could not recognize the weekday information."
        days_ahead = (target_weekday - anchor_date.weekday()) % 7
        target_date = anchor_date + timedelta(days=days_ahead)
    else:
        return "Could not parse the relative time. Please use a clearer description."

    return target_date.strftime("%Y-%m-%d")


@tool
def get_date_for_relative_time(relative_time_str: str) -> str:
    """Convert a relative time description into a concrete date (YYYY-MM-DD).

    Supports: 昨天/yesterday, 前天, 今天/today, 明天/tomorrow, 后天,
    今天星期几/what day is today (returns date + weekday),
    上周X/last Wednesday, 这X/this Wednesday.
    Uses the current system date as "today".
    """
    print(f"--- [Tool Call] Parsing relative time: {relative_time_str} ---")
    return resolve_relative_time_text(relative_time_str)
