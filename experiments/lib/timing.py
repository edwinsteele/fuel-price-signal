from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def time_block(label: str) -> Generator[None, None, None]:
    """Context manager that prints '  [label] N.Ns' on exit."""
    t0 = time.perf_counter()
    yield
    print(f"  [{label}] {time.perf_counter() - t0:.1f}s", flush=True)
