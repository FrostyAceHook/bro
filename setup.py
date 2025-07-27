"""
Builds all '.pyx' files in './bro'.
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
            "bro.sim_burn",
            sources=["bro/sim_burn.pyx"],
            libraries=["sim_burn_impl"],
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
