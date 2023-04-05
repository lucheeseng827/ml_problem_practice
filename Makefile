precommit-all:
	pre-commit run --all-files

precommit-init:
	pre-commit install

precommit-update:
	pre-commit autoupdate

precommit-install:
	pip install pre-commit ./precommit_requirements.txt

# Path: Makefile
