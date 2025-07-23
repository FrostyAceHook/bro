@set errorlevel=0
@pushd "%~dp0"
@echo BUILDING RN
@if %errorlevel% == 0 @call py setup.py build_ext --inplace
@echo(
@echo BROING RN
@if %errorlevel% == 0 @call py -m bro
@popd
