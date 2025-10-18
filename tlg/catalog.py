"""Dataset catalog helpers for Stage A2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class CatalogError(RuntimeError):
    """Raised when catalog operations fail."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "catalog" / "datasets.yml"
GARDEN_PROJECT_PATH = PROJECT_ROOT / "garden_project.yml"


@dataclass
class CatalogEntry:
    name: str
    path: str
    manifest_hash: str
    created_utc: str


def load_slug() -> str:
    if not GARDEN_PROJECT_PATH.exists():
        return "unknown"
    data = yaml.safe_load(GARDEN_PROJECT_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("slug"), str):
        return data["slug"]
    return "unknown"


def compute_manifest_hash(manifest_path: Path) -> str:
    if not manifest_path.exists():
        raise CatalogError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = json.dumps(_canonicalize(manifest), sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or DEFAULT_CATALOG_PATH
    if not catalog_path.exists():
        return {"slug": load_slug(), "datasets": []}
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CatalogError("Catalog file must contain a mapping")
    if "datasets" not in data:
        data["datasets"] = []
    if not isinstance(data["datasets"], list):
        raise CatalogError("catalog.datasets must be a list")
    if "slug" not in data:
        data["slug"] = load_slug()
    return data


def save_catalog(data: dict[str, Any], path: Path | None = None) -> None:
    catalog_path = path or DEFAULT_CATALOG_PATH
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def add_dataset_to_catalog(dataset_dir: Path, catalog_path: Path | None = None) -> CatalogEntry:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_name = manifest.get("dataset_name", dataset_dir.name)
    manifest_hash = compute_manifest_hash(manifest_path)

    catalog = load_catalog(catalog_path)
    created = datetime.now(UTC).isoformat()
    entry = CatalogEntry(
        name=str(dataset_name),
        path=str(dataset_dir),
        manifest_hash=manifest_hash,
        created_utc=created,
    )

    datasets: list[dict[str, Any]] = []
    for item in catalog.get("datasets", []):
        if not isinstance(item, dict):
            continue
        if item.get("path") == entry.path:
            item.update(
                {
                    "name": entry.name,
                    "manifest_hash": entry.manifest_hash,
                    "created_utc": entry.created_utc,
                }
            )
            datasets.append(item)
        else:
            datasets.append(item)
    if all(item.get("path") != entry.path for item in datasets if isinstance(item, dict)):
        datasets.append(entry.__dict__)
    catalog["datasets"] = datasets
    if "slug" not in catalog:
        catalog["slug"] = load_slug()

    save_catalog(catalog, catalog_path)
    return entry


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


__all__ = [
    "CatalogError",
    "CatalogEntry",
    "DEFAULT_CATALOG_PATH",
    "add_dataset_to_catalog",
    "load_catalog",
    "compute_manifest_hash",
]
