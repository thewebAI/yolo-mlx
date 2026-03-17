# Copyright (c) 2026 webAI, Inc.
"""Project runtime directory bootstrap helpers.

Creates commonly used project directories on demand so scripts can run in a
fresh checkout without tracked placeholder files.
"""

from pathlib import Path

RUNTIME_DIR_NAMES = ("datasets", "images", "models", "results")


def ensure_runtime_dirs(project_dir: Path) -> dict[str, Path]:
    """Ensure standard runtime directories exist under the project root.

    Args:
        project_dir: Project root directory path.

    Returns:
        Mapping from directory name to resolved directory path.
    """
    created: dict[str, Path] = {}
    for name in RUNTIME_DIR_NAMES:
        path = (project_dir / name).resolve()
        path.mkdir(parents=True, exist_ok=True)
        created[name] = path
    return created
