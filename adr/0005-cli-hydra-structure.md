# ADR 0005: CLI + Hydra Structure

## Status
Accepted

## Context
The CLI must load Hydra-style YAML configs and expose commands for building datasets and computing
stats.

## Decision
Use Typer for command definition with explicit `build-data` and `stats` commands. Configs are loaded
through `tlg.config.load_config`, which mirrors Hydra structure (datasets, tasks, noise, tokenizer).
Overrides for format and seed are available via CLI options.

## Consequences
The CLI is simple to extend and integrates with the rest of the package. Hydra users can reuse YAML
configs directly, and Typer ensures a friendly UX.
