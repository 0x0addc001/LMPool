"""
Canonical end-to-end shared-prefix benchmark entry.

This wrapper keeps the implementation in shared_prefix_benchmark.py so older
commands remain valid while new documentation can use a cleaner file name.
"""

from shared_prefix_benchmark import main


if __name__ == "__main__":
    main()
