import os
import sys


def configure_windows_runtime():
    if sys.platform != "win32":
        return

    thread_defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in thread_defaults.items():
        os.environ.setdefault(key, value)

    # Do not keep the unsafe duplicate-OpenMP override enabled.
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() == "TRUE":
        os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
