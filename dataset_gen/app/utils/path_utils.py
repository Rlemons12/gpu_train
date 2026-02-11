from pathlib import PureWindowsPath, PurePosixPath

def normalize_to_wsl_path(path: str) -> str:
    """
    Convert a Windows path (C:\\...) to a WSL path (/mnt/c/...).
    If already a POSIX path, return as-is.
    """
    path = path.strip()

    # Already WSL / POSIX
    if path.startswith("/mnt/"):
        return path

    # Windows drive path
    if ":" in path and "\\" in path:
        win = PureWindowsPath(path)
        drive = win.drive.replace(":", "").lower()
        return str(PurePosixPath("/mnt", drive, *win.parts[1:]))

    return path

