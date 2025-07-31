@pushd "%~dp0"
@setlocal

@set GCC_ARGS=-O3 -Wall -Wextra -Wpedantic -Werror -fmax-errors=5 -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wvla -Wold-style-definition -I bro -march=native -mtune=native -fno-unsafe-math-optimizations -fassociative-math -freciprocal-math -fno-finite-math-only -fno-signed-zeros -fno-trapping-math -fno-rounding-math -fno-signaling-nans -fno-fp-int-builtin-inexact

gcc %GCC_ARGS% -c bro\sim.c -o bro\sim.o
@if not "%ERRORLEVEL%"=="0" goto END

gcc %GCC_ARGS% -DBR_HAVEALOOK=1 -masm=intel -S bro\sim.c -o bro\sim.s
@if not "%ERRORLEVEL%"=="0" goto END

gcc -shared -o bro\sim.dll bro\sim.o -Wl,--out-implib,bro\sim.lib
@if not "%ERRORLEVEL%"=="0" goto END

py setup.py build_ext --inplace
@if not "%ERRORLEVEL%"=="0" goto END

:END
@endlocal
@popd
