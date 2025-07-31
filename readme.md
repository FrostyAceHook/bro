BRO - the Big Rocket Optimiser
==============================

Optimises several parameters of a hybrid rocket motor to achieve the highest score over a range of metrics.

Requires:
- numpy
- matplotlib
- scipy (if running 'bro.approximator')
- coolprop
- rocketcea
- setuptools
- cython (+msvc unless u can find the option to use mingw)
- mingw32 gcc

To run:
1. `compile.bat` to build the sim dll and bridge cython module.
2. `py -m bro` to run the entire thing.
