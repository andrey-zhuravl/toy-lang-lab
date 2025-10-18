"""Typer CLI wrapping Hydra-style configs for Toy Lang Lab."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from tlg.config import ALLOWED_FORMATS, load_config
from tlg.generator import generate_dataset
from tlg.grammar import load_grammar
from tlg.mlflow_logger import log_run
from tlg.stats import compute_stats, dataset_stats_from_dir
from tlg.writer import write_dataset

app = typer.Typer(help="Toy Lang Lab CLI (Stage A1)")


def _timestamp_dir(base: Path = Path("out")) -> Path:
    now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return base / now


def _parse_override(value: str, key: str) -> str:
    if value.startswith(f"{key}="):
        return value.split("=", 1)[1]
    return value


# ConfArg = Annotated[str, typer.Argument()]
#
# # ИСПРАВЛЕНИЕ: Удалено 'None' из typer.Argument, так как значение по умолчанию
# # должно устанавливаться только в сигнатуре функции (т.е., ' = None' в build_data)
# OverridesArg = Annotated[list[str] | None, typer.Argument(help="Hydra-style overrides")]
#
# # ИСПРАВЛЕНИЕ: Удален аргумент 'None' из typer.Option. Typer теперь корректно
# # выведет имя опции '--mlflow' из имени параметра 'mlflow'.
# MlflowOpt = Annotated[bool | None, typer.Option(help="Force MLflow logging on/off")]

ConfArg = Annotated[str, typer.Argument(..., help="Config path or conf=...")]
OverridesArg = Annotated[list[str] | None, typer.Argument(None, help="Hydra-style overrides")]
MlflowOpt = Annotated[bool | None, typer.Option(None, help="Force MLflow logging on/off")]

@app.command("build-data")
def build_data(
    conf: ConfArg,
    overrides: OverridesArg = None,
    mlflow: MlflowOpt = None,
) -> None:
    overrides = overrides or []
    override_map: dict[str, str] = {}
    for item in overrides:
        if "=" not in item:
            raise typer.BadParameter(f"Override '{item}' must be key=value")
        key, value = item.split("=", 1)
        override_map[key] = value

    config_value = _parse_override(conf, "conf")
    config_path = Path(config_value)
    cfg = load_config(config_path)

    format_override = override_map.get("format")
    out_override = override_map.get("out")
    seed_override = override_map.get("seed")
    if mlflow is None and "mlflow" in override_map:
        mlflow = override_map["mlflow"].lower() in {"1", "true", "yes"}

    if format_override:
        if format_override not in ALLOWED_FORMATS:
            raise typer.BadParameter(f"format must be one of {ALLOWED_FORMATS}")
        dataset_cfg = replace(cfg.dataset, format=format_override)
        cfg = replace(cfg, dataset=dataset_cfg)

    if seed_override is not None:
        try:
            new_seed = int(seed_override)
        except ValueError as exc:  # pragma: no cover - defensive
            raise typer.BadParameter("seed override must be an integer") from exc
        cfg = replace(cfg, seed=new_seed)

    grammar = load_grammar(cfg.grammar_yaml)
    dataset, grammar_hash = generate_dataset(cfg, grammar)
    stats = compute_stats(dataset)

    output_dir = Path(out_override) if out_override else _timestamp_dir()
    result = write_dataset(cfg, dataset, grammar_hash, stats, output_dir)

    typer.echo(json.dumps(result.manifest, indent=2))

    params = {
        "dataset_name": cfg.dataset.name,
        "format": cfg.dataset.format,
        "splits.train": cfg.dataset.splits.train,
        "splits.valid": cfg.dataset.splits.valid,
        "splits.test": cfg.dataset.splits.test,
        "seed": cfg.seed,
        "config_hash": result.manifest["config_hash"],
        "code_hash": result.manifest["code_hash"],
    }
    tags = {"stage": "A1", "component": "tlg-data-gen"}
    log_run(params=params, tags=tags, manifest=result.manifest, stats=stats, enabled=mlflow)


StatsArg = Annotated[str, typer.Argument(..., help="Output directory or path=...")]


@app.command()
def stats(path: StatsArg) -> None:
    output_dir = Path(_parse_override(path, "path"))
    stats_data = dataset_stats_from_dir(output_dir)
    typer.echo(json.dumps(stats_data, indent=2))

@app.command("hello")
def hello(name: str = typer.Argument("world", help="Who to greet")) -> None:
    """Print a friendly greeting."""
    typer.echo(f"Hello, {name}!")


def run() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
