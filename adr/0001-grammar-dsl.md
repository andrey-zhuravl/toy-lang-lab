# ADR 0001: Grammar DSL Structure & Validation

## Status
Accepted

## Context
We require a human-readable DSL for defining template-driven synthetic worlds. The DSL must be
simple, YAML-based, and validated for reproducibility as described in the Stage A1 spec.

## Decision
Use YAML files with top-level sections `entities`, `templates`, optional `weights`, and optional
`noise`. Validation is performed by `tlg.grammar.load_grammar`, ensuring entities are non-empty,
templates exist, and weights sum to 1.

## Consequences
The DSL is easy to edit by hand, integrates with Hydra configs, and remains deterministic thanks to
strict validation. Additional sections can be added in later stages without breaking compatibility.
