"""Fixed BLAS / OpenMP thread count for every probe fit: ONE thread.

Every linear-probe fit in the release runs single-threaded (joblib workers via fixed_threads(fn, 1);
scorers without workers call threadpool_limits(1) after their imports). L-BFGS on raw, unscaled spike
counts is badly conditioned for the voltage-STDP networks, so where it stops depends on the order of
floating-point sums, which changes with the thread count; one thread makes the result independent of
the machine. The shipped tables were recomputed this way (2026-10-25).

History:

The linear-probe fits (L-BFGS on raw spike counts) can move by one test image when the number of
BLAS threads changes. The shipped tables were made on a 36-core machine where loky gave each worker
cpu_count() // n_jobs threads (or 1 where the scorer set OPENBLAS_NUM_THREADS=1); fixed_threads()
pins that number explicitly, so the result no longer depends on the machine's core count.
"""
from functools import partial

from threadpoolctl import threadpool_limits


def _call(fn, n_threads, *args, **kwargs):
    with threadpool_limits(n_threads):
        return fn(*args, **kwargs)


def fixed_threads(fn, n_threads):
    """fn wrapped so that it runs with n_threads BLAS / OpenMP threads (use with joblib.delayed)."""
    return partial(_call, fn, n_threads)
