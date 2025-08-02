@pushd "%~dp0"
@setlocal

@set GCC_ARGS=-std=c11 -O3 -m64 -march=native -mtune=native -Wall -Wextra -Wpedantic -Werror -fmax-errors=5 -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wvla -Wold-style-definition -I bro -fno-ident -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-stack-protector -fno-unsafe-math-optimizations -fassociative-math -freciprocal-math -fno-finite-math-only -fno-signed-zeros -fno-trapping-math -fno-rounding-math -fno-signaling-nans -fno-fp-int-builtin-inexact

gcc %GCC_ARGS% -c -o bro\sim.o %* bro\sim.c
@if not "%ERRORLEVEL%"=="0" goto END

gcc %GCC_ARGS% -DBR_HAVEALOOK=1 -masm=intel -S -o bro\sim.s %* bro\sim.c
@if not "%ERRORLEVEL%"=="0" goto END

gcc -shared -Wl,--out-implib,bro\sim.lib -o bro\sim.dll bro\sim.o
@if not "%ERRORLEVEL%"=="0" goto END

py setup.py build_ext --inplace
@if not "%ERRORLEVEL%"=="0" goto END

:END
@endlocal
@popd
