@pushd "%~dp0"
@setlocal ENABLEDELAYEDEXPANSION
@echo OFF

if exist ".\build\" (
    echo Deleting directory: .\build
    rmdir /s /q ".\build"
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

@echo ON
@endlocal
@popd
