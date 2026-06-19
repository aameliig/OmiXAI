"""Backward-compatible shim. The genome loader/cache now lives in the package.

    from omixai.data.genome import load_genome_cached, CHROMS

This shim keeps existing imports (`from genome_cache import ...`) working.
"""
from omixai.data.genome import (  # noqa: F401
    CHROMS,
    load_genome,
    load_genome_cached,
)
