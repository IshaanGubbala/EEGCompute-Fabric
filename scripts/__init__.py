"""Scripts package to allow `python -m scripts.*` execution.

This enables imports like `from core...` by ensuring the project root is on sys.path
when invoked with `-m` from the repository root.
"""

