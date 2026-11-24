@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /EHsc /W4 /I agility\build\native\include depth-uav-probe.cpp d3d12.lib dxgi.lib /Fe:depth-uav-probe.exe
