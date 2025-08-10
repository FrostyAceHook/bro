@pushd "%~dp0"
@setlocal ENABLEDELAYEDEXPANSION
@echo OFF

if exist ".\build\" (
    echo Deleting directory: .\build
    rmdir /s /q ".\build"
)

if exist ".\bro\__pycache__\" (
    echo Deleting directory: .\bro\__pycache__
    rmdir /s /q ".\bro\__pycache__"
)

for %%F in (.\bro\sim.*) do (
    if exist "%%F" (
        set "FNAME=%%~NXF"
        if /I not "!FNAME!"=="sim.h" if /I not "!FNAME!"=="sim.c" (
            echo Deleting file: %%F
            del /F /Q "%%F"
        )
    )
)

for %%F in (.\bro\bridge.*) do (
    if exist "%%F" (
        set "FNAME=%%~NXF"
        if /I not "!FNAME!"=="bridge.pyx" (
            echo Deleting file: %%F
            del /F /Q "%%F"
        )
    )
)

if exist ".\bro\approximator_cea_cache.npz" (
    echo Deleting file: .\bro\approximator_cea_cache.npz
    del /F /Q ".\bro\approximator_cea_cache.npz"
)
if exist ".\bro\approximator_cea_cache.lock" (
    echo Deleting file: .\bro\approximator_cea_cache.lock
    del /F /Q ".\bro\approximator_cea_cache.lock"
)

@echo ON
@endlocal
@popd
