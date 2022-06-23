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


# Check that source code meets quality standards
check-codestyle:
	black --check --line-length 119 --target-version py37 sample_factory sample_factory_examples
	isort --check-only sample_factory sample_factory_examples
	flake8 sample_factory sample_factory_examples

# Format source code automatically
format:
	black --line-length 119 --target-version py36 sample_factory sample_factory_examples
	isort sample_factory sample_factory_examples

# Run tests for the library
test:
	bash all_tests.sh
