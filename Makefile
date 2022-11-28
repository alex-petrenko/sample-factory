.PHONY: build

build: setup.py
	python3 -m pip install --upgrade build && python3 -m build

.PHONY: upload

upload:
	python3 -m twine upload --repository pypi dist/*

.PHONY: upload-test

upload-test:
	python3 -m twine upload --repository testpypi dist/*

.PHONY: clean

clean:
	rm -rf build dist _vizdoom && find . -name "_vizdoom.ini" -delete && find . -type d -name "_vizdoom" -delete


line_len = 120
line_len_arg = --line-length $(line_len)
code_folders = ./


# Format source code automatically
.PHONY: format

format:
	black $(line_len_arg) -t py38 $(code_folders)
	isort $(line_len_arg) --py 38 --profile black $(code_folders)


# Check that source code meets quality standards
.PHONY: check-codestyle

check-codestyle:
	black --check $(line_len_arg) -t py38 $(code_folders)
	isort --check-only $(line_len_arg) --py 38 --profile black $(code_folders)
# ignore some formatting issues already covered by black
	flake8 --max-line-length $(line_len) --ignore=E501,F401,E203,W503,E126,E722 $(code_folders)


# Run tests for the library
.PHONY: test

test:
	pytest -s --maxfail=2
# ; echo "Tests finished. You might need to type 'reset' and press Enter to fix the terminal window"


# Run code coverage test
.PHONY: test-cov

test-cov:
	pytest --cov=./ -v
# ; echo "Tests finished. You might need to type 'reset' and press Enter to fix the terminal window"

# Run code coverage test
.PHONY: test-cov-core

test-cov-core:
	pytest --cov=./ --cov-config=./.core-coveragerc -v
# ; echo "Tests finished. You might need to type 'reset' and press Enter to fix the terminal window"


# docs
.PHONY: docs-serve

docs-serve:
	bash ./docs/cfg-params.sh && mkdocs serve