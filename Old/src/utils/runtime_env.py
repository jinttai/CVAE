import os
import sys


def configure_windows_runtime():
    if sys.platform != "win32":
        return

    # Must be set before numpy/torch import to avoid libiomp5md.dll clash
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    thread_defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in thread_defaults.items():
        os.environ.setdefault(key, value)
