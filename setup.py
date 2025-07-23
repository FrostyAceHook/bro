"""
Builds all '.pyx' files in './bro'.
"""

import os
from setuptools import setup
from Cython.Build import cythonize

ROOT = "bro"

def main():
    # Enable colours.
    os.system("")

    paths = []
    for dirpath, _, filenames in os.walk(ROOT):
        for fname in filenames:
            if not fname.endswith(".pyx"):
                continue
            paths.append(os.path.join(dirpath, fname))
    setup(
        ext_modules=cythonize(paths),
    )

if __name__ != "__main__":
    raise ImportError("do not import setup.py")
main()
