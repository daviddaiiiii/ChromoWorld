"""Locate the release root and resolve CLI paths consistently across components."""
from pathlib import Path


def find_release_root(source: str | Path) -> Path:
    path = Path(source).resolve()
    for candidate in path.parents:
        if (candidate / "Code").is_dir() and (candidate / "Code" / "Analyze").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate the release root containing Code/Analyze from {path}")


def resolve_release_path(value: str | Path, source: str | Path) -> Path:
    """Resolve relative paths from release, with aliases for historical commands."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "release":
        parts = parts[1:]
    if parts and parts[0] in {"Analyze", "ChromoPix", "ChromoPhy"}:
        parts = ("Code", *parts)
    if parts and parts[0] == "Artifact":
        parts = ("Artifacts", *parts[1:])
    return find_release_root(source).joinpath(*parts)
