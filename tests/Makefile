SHELL := /bin/bash

.PHONY: install-pennylane-scaleway
install-pennylane-scaleway:
	pip install "..[aer]"

.PHONY: install-dependencies
install-dependencies: install
	pip install --upgrade pip
	pip install -Ur requirements.txt

.PHONY: test-aer
test-aer: install-pennylane-scaleway print-env
	pytest -s --showprogress -vv test_aer_device.py

.PHONY: randomized-circuits-test
randomized-circuits-test: install-pennylane-scaleway print-env
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
