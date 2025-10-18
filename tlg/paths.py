from pathlib import Path


def get_config_path(config_file_name: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "conf" / "data" / config_file_name