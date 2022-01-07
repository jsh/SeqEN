#!/bin/bash -eu
# Get Git to run isort and black on each Python file commit

pip install pre-commit                      # install Python's Git pre-commit utility
cat > .pre-commit-config.yaml <<'__EOF__'   # configure to run isort and black
repos:
  - repo: local
    hooks:
      - id: isort
        name: Sorting import statements
        entry: bash -c 'isort "$@"; git add -u' --
        language: python
        args: ["--profile", "black", "--filter-files"]
        files: \.py$
      - id: black
        name: Black Python code formatting
        entry: bash -c 'black "$@"; git add -u' --
        language: python
        types: [python]
        args: ["--line-length=100"]
      - id: mypy
        name: Optional static type checker
        entry: bash -c 'mypy "$@"; git add -u' --
        language: python
        files: \.py$
__EOF__

pre-commit install                     # tell pre-commit to create a .git/hooks/pre-commit from the YAML
