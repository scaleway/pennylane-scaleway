SHELL := /bin/bash

.PHONY: install-pennylane-scaleway
install-pennylane-scaleway: install-dependencies
	pip install "..[all]"

.PHONY: install-dependencies
install-dependencies:
	pip install --upgrade pip
	pip install -Ur ../requirements.txt

.PHONY: install-tests-dependencies
install-tests-dependencies:
	pip install --upgrade pip
	pip install -Ur requirements.txt

.PHONY: test-all
test-all: install-tests-dependencies install-pennylane-scaleway print-env
	pytest -v -s .

.PHONY: test-scaleway
test-scaleway: install-tests-dependencies install-pennylane-scaleway print-env
	pytest -v -s test_scaleway_devices.py

.PHONY: test-aer
test-aer: install-tests-dependencies install-pennylane-scaleway print-env
	pytest -v -s test_aer_device.py

.PHONY: test-aqt
test-aqt: install-tests-dependencies install-pennylane-scaleway print-env
	pytest -v -s test_aqt_device.py

.PHONY: test-iqm
test-iqm: install-tests-dependencies install-pennylane-scaleway print-env
	pytest -v -s test_iqm_device.py

.PHONY: test-randomized-circuits
test-randomized-circuits: install-pennylane-scaleway print-env
	python randomized_circuits_test.py

.PHONY: print-env
print-env:
	@echo 'Print env vars for testing...'
	@echo SCW_API_URL=${SCW_API_URL}
	@echo SCW_SECRET_KEY=${SCW_SECRET_KEY}
	@echo SCW_PROJECT_ID=${SCW_PROJECT_ID}
	@echo SCW_BACKEND_NAME=${SCW_BACKEND_NAME}
	@echo 'End printing env vars for testing'

.PHONY: black
black:
	pip install black
	black ..
