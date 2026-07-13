"""
Canonical KV transfer microbenchmark entry.

This wrapper keeps the implementation in kv_transfer_benchmark.py so older
commands remain valid while new documentation can use a cleaner file name.
"""

from kv_transfer_benchmark import main


if __name__ == "__main__":
    main()
