# Toy Lang Lab (A1)

Toy Lang Lab generates small synthetic datasets for experimentation with simple grammars. Stage A1
focuses on deterministic data builds from YAML grammars, reproducibility metadata, and a minimal CLI
experience.

## Quickstart

```bash
pip install -e .[dev]
tlg build-data conf=conf/data/toy_small.yaml out=out/dev
```

Inspect dataset statistics:

```bash
tlg stats path=out/dev
```

## Reproducibility

All dataset builds record a manifest with configuration, git revision, and deterministic hashes for
config and data splits. Re-running `tlg build-data` with the same configuration produces identical
hashes and output files.

See `spec/toy-lang-lab.A1.spec.v2.json` for the full scope and acceptance criteria of Stage A1.
