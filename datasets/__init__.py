"""
HearLink ASL — local datasets package.

This __init__.py makes the local datasets/ directory a proper Python package
so that `from datasets.text_normalizer import ...` resolves to the local module
even when the HuggingFace `datasets` package is installed.

The real HuggingFace `datasets` package is pre-loaded into sys.modules under
the key '_hf_datasets' so that third-party libraries (e.g. `evaluate`) continue
to find `Dataset`, `load_dataset`, etc. The conftest.py handles routing
`import datasets` to the real package for third-party use within tests.
"""
