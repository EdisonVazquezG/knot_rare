from __future__ import annotations

import importlib


MODULE_ORDER = (
    "config",
    "preprocessing",
    "qc",
    "representations",
    "io",
    "pca",
    "hardsets",
    "concordance",
    "enrichment",
    "null_models",
    "norms",
    "models",
    "matching",
    "robustness",
    "conditional_nulls",
    "continuous",
    "exploratory",
    "artifacts",
    "pipeline",
    "scaling",
    "autoencoders",
    "heldout",
)


def reload_package():
    """Reload package modules during interactive development.

    For a frozen manuscript run, restarting the kernel is still preferred.
    """

    for module_name in MODULE_ORDER:
        try:
            module = importlib.import_module(f"consensus_hardness.{module_name}")
        except (ImportError, ModuleNotFoundError):
            continue
        importlib.reload(module)

    root = importlib.import_module("consensus_hardness")
    return importlib.reload(root)
