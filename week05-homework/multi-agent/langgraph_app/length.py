def length_bounds(
    target: int,
    *,
    min_ratio: float = 0.85,
    max_ratio: float = 1.15,
    floor: int = 100,
) -> tuple[int, int]:
    """Return inclusive min/max character bounds for a target length."""
    min_chars = max(floor, int(target * min_ratio))
    max_chars = max(min_chars, int(target * max_ratio))
    return min_chars, max_chars


def validate_length(
    text: str,
    target: int,
    *,
    stage: str,
    min_ratio: float = 0.85,
    max_ratio: float = 1.15,
) -> None:
    """Raise ValueError when text is outside the allowed character range."""
    count = len(text)
    min_chars, max_chars = length_bounds(
        target,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )
    if count < min_chars:
        raise ValueError(
            f"{stage}代理返回内容过短（{count} 字符，目标 {target}，"
            f"至少需要 {min_chars}）。"
        )
    if count > max_chars:
        raise ValueError(
            f"{stage}代理返回内容过长（{count} 字符，目标 {target}，"
            f"最多 {max_chars}）。"
        )
