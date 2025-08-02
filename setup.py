"""
Builds the 'bro/bridge.pyx' file.
"""

import os
import sys
import traceback

import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

def main():
    # Enable colours.
    os.system("")

    try:
        ext = Extension(
            "bro.bridge",
            sources=["bro/bridge.pyx"],
            libraries=["sim"],
            library_dirs=["bro"],
            include_dirs=["bro", np.get_include()],
        )
        setup(
            ext_modules=cythonize([ext]),
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1

if __name__ != "__main__":
    raise ImportError("do not import setup.py")
sys.exit(main())
