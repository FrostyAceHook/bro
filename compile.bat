@pushd "%~dp0"
@setlocal

@set GCC_ARGS=-O3 -Wall -Wextra -Werror -I bro -march=native -mtune=native -fno-unsafe-math-optimizations -fassociative-math -freciprocal-math -fno-finite-math-only -fno-signed-zeros -fno-trapping-math -fno-rounding-math -fno-signaling-nans -fno-fp-int-builtin-inexact

gcc %GCC_ARGS% -c bro\sim_burn_impl.c -o bro\sim_burn_impl.o
@if not "%ERRORLEVEL%"=="0" goto END

gcc %GCC_ARGS% -DBR_HAVEALOOK=1 -masm=intel -S bro\sim_burn_impl.c -o bro\sim_burn_impl.s
@if not "%ERRORLEVEL%"=="0" goto END

gcc -shared -o bro\sim_burn_impl.dll bro\sim_burn_impl.o -Wl,--out-implib,bro\sim_burn_impl.lib
@if not "%ERRORLEVEL%"=="0" goto END

py setup.py build_ext --inplace
@if not "%ERRORLEVEL%"=="0" goto END

:END
@endlocal
@popd
