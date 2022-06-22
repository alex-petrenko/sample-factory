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
quality:
	black --check --line-length 119 --target-version py36 tests src
	isort --check-only tests src
	flake8 tests src

# Format source code automatically
style:
	black --line-length 119 --target-version py36 tests src
	isort tests src

# Run tests for the library
test:
	bash all_tests.sh
