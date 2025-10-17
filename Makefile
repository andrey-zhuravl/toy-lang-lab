PYTHON = python3
PACKAGE = tlg
CONF ?= conf/data/toy_small.yaml
OUT ?= out/dev

.PHONY: build-data fmt lint typecheck test smoke release clean

build-data:
$(PYTHON) -m $(PACKAGE).cli build-data conf=$(CONF) out=$(OUT)

fmt:
ruff --fix
black .

lint:
ruff .
black --check .

typecheck:
mypy --strict $(PACKAGE)

test:
pytest -q

smoke:
pytest -q tests/e2e/test_smoke.py

release:
$(PYTHON) -m build
@echo tag v0.1.0

clean:
rm -rf out dist *.egg-info .mypy_cache .pytest_cache
