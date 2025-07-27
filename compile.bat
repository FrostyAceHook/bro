@pushd "%~dp0"

gcc -O3 -Wall -Wextra -Werror -I bro -c bro\sim_burn_impl.c -o bro\sim_burn_impl.o
@if not "%ERRORLEVEL%"=="0" goto END

gcc -shared -o bro\sim_burn_impl.dll bro\sim_burn_impl.o -Wl,--out-implib,bro\sim_burn_impl.lib
@if not "%ERRORLEVEL%"=="0" goto END

py setup.py build_ext --inplace
@if not "%ERRORLEVEL%"=="0" goto END

:END
@popd
