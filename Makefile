CONFIG ?= config/default.yaml
PY ?= python

.PHONY: install all simulate features train threshold production monitor report test lint clean

install:
	$(PY) -m pip install -e ".[dev]"

all:
	$(PY) -m pfloat all --config $(CONFIG)

simulate:
	$(PY) -m pfloat simulate --config $(CONFIG)
features:
	$(PY) -m pfloat features --config $(CONFIG)
train:
	$(PY) -m pfloat train --config $(CONFIG)
threshold:
	$(PY) -m pfloat select-threshold --config $(CONFIG)
production:
	$(PY) -m pfloat production --config $(CONFIG)
monitor:
	$(PY) -m pfloat monitor --config $(CONFIG)
report:
	$(PY) -m pfloat report --config $(CONFIG)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

clean:
	rm -rf data data_ci artifacts artifacts_ci reports_ci
