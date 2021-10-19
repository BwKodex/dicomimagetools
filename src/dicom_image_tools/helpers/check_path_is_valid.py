from pathlib import Path
from typing import Optional, Union


def check_path_is_valid_path(path_to_check: Union[str, Path]) -> Optional[Path]:
    """Checks a path given as a string or Path object to see if it is a valid and existing path

    Args:
        path_to_check: Path to validate

    Returns:
        The filepath as a Path-instance if valid file else None

    Raises:
        TypeError: If filepath is neither a string nor a Path
    """
    if isinstance(path_to_check, str):
        path_to_check = Path(path_to_check)

    if not isinstance(path_to_check, Path) or not path_to_check.exists():
        raise TypeError("Invalid path")

    return path_to_check
