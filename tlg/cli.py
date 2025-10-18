"""Typer CLI wrapping Hydra-style configs for Toy Lang Lab."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from tlg.catalog import add_dataset_to_catalog
from tlg.config import ALLOWED_FORMATS, load_config
from tlg.dictionaries import build_dictionary, load_dictionary_config
from tlg.embeddings import build_embeddings, load_embeddings_config
from tlg.generator import generate_dataset
from tlg.grammar import load_grammar
from tlg.mlflow_logger import log_run
from tlg.stats import compute_stats, dataset_stats_from_dir
from tlg.tokenizers_learned import (
    apply_tokenizer_to_dataset,
    load_bpe_training_config,
    load_learned_tokenizer,
    save_trained_tokenizer,
    train_bpe,
)
from tlg.validate import validate_dataset
from tlg.writer import write_dataset

app = typer.Typer(help="Toy Lang Lab CLI (Stage A2)")
tok_app = typer.Typer(help="Tokenizer commands")
app.add_typer(tok_app, name="tok")
dict_app = typer.Typer(help="Dictionary commands")
app.add_typer(dict_app, name="dict")
embed_app = typer.Typer(help="Embeddings commands")
app.add_typer(embed_app, name="embed")
catalog_app = typer.Typer(help="Catalog commands")
app.add_typer(catalog_app, name="catalog")


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
OverridesArg = Annotated[list[str] | None, typer.Argument(help="Hydra-style overrides")]
MlflowOpt = Annotated[bool | None, typer.Option(help="Force MLflow logging on/off")]


def _parse_override_map(overrides: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not overrides:
        return mapping
    for item in overrides:
        if "=" not in item:
            raise typer.BadParameter(f"Override '{item}' must be key=value")
        key, value = item.split("=", 1)
        mapping[key] = value
    return mapping


def _update_nested(data: dict[str, object], path: list[str], value: object) -> None:
    current: dict[str, object] = data
    for key in path[:-1]:
        entry = current.setdefault(key, {})
        if not isinstance(entry, dict):
            entry = {}
            current[key] = entry
        current = entry  # type: ignore[assignment]
    current[path[-1]] = value


def _canonicalize_raw(raw: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(raw))

@app.command("build-data")
def build_data(
    conf: ConfArg,
    overrides: OverridesArg = None,
    mlflow: MlflowOpt = None,
) -> None:
    override_map = _parse_override_map(overrides)

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

    extra_manifest: dict[str, object] = {}
    if cfg.tokenizer.type == "learned":
        if cfg.tokenizer.model_path is None:
            raise typer.BadParameter("learned tokenizer requires model_path")
        tokenizer = load_learned_tokenizer(cfg.tokenizer.model_path)
        tokenization_meta = apply_tokenizer_to_dataset(
            dataset,
            cfg.tokenizer.apply_to,
            tokenizer,
            cfg.tokenizer.add_ids,
        )
        extra_manifest.update(
            {
                "tokenizer_name": tokenization_meta["tokenizer_name"],
                "tokenizer_hash": tokenization_meta["tokenizer_hash"],
                "tokenizer_config": cfg.raw_dict.get("tokenizer", {}),
            }
        )

    stats = compute_stats(dataset)

    output_dir = Path(out_override) if out_override else _timestamp_dir()
    result = write_dataset(
        cfg,
        dataset,
        grammar_hash,
        stats,
        output_dir,
        extra_manifest=extra_manifest,
    )

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
        "parallel_workers": cfg.performance.parallel_workers,
        "streaming_writer": cfg.performance.streaming_writer,
    }
    if cfg.tokenizer.type == "learned":
        params["tokenizer_name"] = extra_manifest.get("tokenizer_name", "")
        params["tokenizer_hash"] = extra_manifest.get("tokenizer_hash", "")
    tags = {"stage": "A2", "component": "tlg-data-gen"}
    log_run(params=params, tags=tags, manifest=result.manifest, stats=stats, enabled=mlflow)


PathArg = Annotated[str, typer.Argument(..., help="Output directory or path=...")]


@app.command()
def stats(path: PathArg) -> None:
    output_dir = Path(_parse_override(path, "path"))
    stats_data = dataset_stats_from_dir(output_dir)
    typer.echo(json.dumps(stats_data, indent=2))


@app.command()
def validate(path: PathArg) -> None:
    output_dir = Path(_parse_override(path, "path"))
    result = validate_dataset(output_dir)
    payload = {
        "ok": result.ok,
        "errors": result.errors,
        "split_hashes": result.split_hashes,
        "dataset_hash": result.dataset_hash,
    }
    typer.echo(json.dumps(payload, indent=2))
    if not result.ok:
        raise typer.Exit(code=1)


@tok_app.command("train")
def tok_train(conf: ConfArg, overrides: OverridesArg = None) -> None:
    override_map = _parse_override_map(overrides)
    config_value = _parse_override(conf, "conf")
    config_path = Path(config_value)
    training_cfg = load_bpe_training_config(config_path)

    raw_config = json.loads(json.dumps(training_cfg.raw))

    if "data" in override_map:
        input_override = Path(override_map["data"])
        training_cfg = replace(training_cfg, input_path=input_override.resolve())
        _update_nested(raw_config, ["train", "input"], str(input_override))
    if "out" in override_map:
        out_override = Path(override_map["out"])
        training_cfg = replace(training_cfg, output_dir=out_override.resolve())
        _update_nested(raw_config, ["output", "dir"], str(out_override))
    if "limit" in override_map:
        try:
            limit_value = int(override_map["limit"])
        except ValueError as exc:  # pragma: no cover - defensive
            raise typer.BadParameter("limit override must be an integer") from exc
        training_cfg = replace(training_cfg, limit=limit_value)
        _update_nested(raw_config, ["train", "limit"], limit_value)
    if "seed" in override_map:
        try:
            seed_value = int(override_map["seed"])
        except ValueError as exc:  # pragma: no cover - defensive
            raise typer.BadParameter("seed override must be an integer") from exc
        training_cfg = replace(training_cfg, seed=seed_value)
        _update_nested(raw_config, ["tokenizer", "seed"], seed_value)

    training_cfg = replace(training_cfg, raw=_canonicalize_raw(raw_config))

    model = train_bpe(training_cfg)
    manifest = save_trained_tokenizer(model, training_cfg.output_dir)
    typer.echo(
        json.dumps(
            {
                "name": manifest.name,
                "tokenizer_hash": manifest.tokenizer_hash,
                "vocab": str(manifest.vocab_path),
                "merges": str(manifest.merges_path),
                "manifest": str(manifest.manifest_path),
            },
            indent=2,
        )
    )


@tok_app.command("apply")
def tok_apply(overrides: OverridesArg = None) -> None:
    override_map = _parse_override_map(overrides)
    required = {"model", "in", "out", "field"}
    missing = sorted(required.difference(override_map))
    if missing:
        raise typer.BadParameter(f"Missing required overrides: {', '.join(missing)}")

    model_path = Path(override_map["model"]).resolve()
    input_path = Path(override_map["in"]).resolve()
    output_path = Path(override_map["out"]).resolve()
    field = override_map["field"]
    add_ids = override_map.get("ids", "false").lower() in {"1", "true", "yes"}

    tokenizer = load_learned_tokenizer(model_path)

    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {input_path}")

    processed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as in_handle, output_path.open(
        "w", encoding="utf-8"
    ) as out_handle:
        for line in in_handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(field)
            if isinstance(value, str):
                tokens, ids = tokenizer.encode_with_ids(value)
                record[f"{field}_tokens"] = tokens
                if add_ids:
                    record[f"{field}_ids"] = ids
            out_handle.write(json.dumps(record, ensure_ascii=False))
            out_handle.write("\n")
            processed += 1

    typer.echo(
        json.dumps(
            {
                "records": processed,
                "tokenizer_hash": tokenizer.tokenizer_hash,
                "field": field,
            },
            indent=2,
        )
    )


@dict_app.command("build")
def dict_build(conf: ConfArg, overrides: OverridesArg = None) -> None:
    override_map = _parse_override_map(overrides)
    config_value = _parse_override(conf, "conf")
    config_path = Path(config_value)
    dict_cfg = load_dictionary_config(config_path)

    raw_config = json.loads(json.dumps(dict_cfg.raw))

    if "out" in override_map:
        out_override = Path(override_map["out"])
        dict_cfg = replace(dict_cfg, output_path=out_override.resolve())
        _update_nested(raw_config, ["output", "path"], str(out_override))
    if "seed" in override_map:
        try:
            seed_value = int(override_map["seed"])
        except ValueError as exc:  # pragma: no cover - defensive
            raise typer.BadParameter("seed override must be an integer") from exc
        dict_cfg = replace(dict_cfg, seed=seed_value)
        _update_nested(raw_config, ["seed"], seed_value)

    dict_cfg = replace(dict_cfg, raw=_canonicalize_raw(raw_config))

    manifest = build_dictionary(dict_cfg)
    typer.echo(json.dumps(manifest, indent=2))


@embed_app.command("build")
def embed_build(conf: ConfArg, overrides: OverridesArg = None) -> None:
    override_map = _parse_override_map(overrides)
    config_value = _parse_override(conf, "conf")
    config_path = Path(config_value)
    emb_cfg = load_embeddings_config(config_path)

    raw_config = json.loads(json.dumps(emb_cfg.raw))

    if "out" in override_map:
        out_override = Path(override_map["out"])
        emb_cfg = replace(emb_cfg, output_path=out_override.resolve())
        _update_nested(raw_config, ["output", "path"], str(out_override))
    if "seed" in override_map:
        try:
            seed_value = int(override_map["seed"])
        except ValueError as exc:  # pragma: no cover - defensive
            raise typer.BadParameter("seed override must be an integer") from exc
        emb_cfg = replace(emb_cfg, seed=seed_value)
        _update_nested(raw_config, ["embeddings", "seed"], seed_value)

    emb_cfg = replace(emb_cfg, raw=_canonicalize_raw(raw_config))

    manifest = build_embeddings(emb_cfg)
    typer.echo(json.dumps(manifest, indent=2))


@catalog_app.command("add")
def catalog_add(path: PathArg, overrides: OverridesArg = None) -> None:
    override_map = _parse_override_map(overrides)
    dataset_path = Path(_parse_override(path, "path"))
    catalog_override = override_map.get("catalog")
    catalog_path = Path(catalog_override).resolve() if catalog_override else None
    entry = add_dataset_to_catalog(dataset_path, catalog_path)
    typer.echo(json.dumps(entry.__dict__, indent=2))


@app.command("hello")
def hello(name: str = typer.Argument("world", help="Who to greet")) -> None:
    """Print a friendly greeting."""
    typer.echo(f"Hello, {name}!")


def run() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
