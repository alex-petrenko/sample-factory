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

