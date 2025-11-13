SHELL := /bin/bash

.PHONY: install
install:
	pip3 install --upgrade pip
	pip3 install -Ur requirements.txt
	pip3 install "..[aer]"

.PHONY: test-aer
test-aer: install print-env
	pytest -s --showprogress -vv test_aer_device.py

.PHONY: test-qnode-stress-test
test-qnode-stress-test: install print-env
	python comparative_tests_generator.py

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
	pip3 install black
	black ..
