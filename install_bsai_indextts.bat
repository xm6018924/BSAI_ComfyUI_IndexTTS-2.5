@echo off
chcp 65001 >nul 2>&1
setlocal

echo ==========================================
echo BSAI_ComfyUI_IndexTTS-2.5 安装脚本
echo ==========================================
echo.

set "ROOT=%~dp0..\..\..\"
set "PYTHON=%ROOT%python\python.exe"
set "GIT_PATH=C:\Program Files\Git\cmd"

echo [1/5] 检查 Python 环境...
if not exist "%PYTHON%" (
    echo 错误: 找不到 Python: %PYTHON%
    pause
    exit /b 1
)
"%PYTHON%" --version
echo.

echo [2/5] 安装 hatchling 构建工具...
"%PYTHON%" -m pip install hatchling --index-url https://pypi.org/simple/ -q
echo.

echo [3/5] 从 GitHub 安装 indextts (跳过依赖和版本检查)...
set "PATH=%GIT_PATH%;%PATH%"
"%PYTHON%" -m pip install --no-deps --ignore-requires-python --no-build-isolation "git+https://github.com/index-tts/index-tts.git"
if errorlevel 1 (
    echo git 安装失败，尝试使用 zip 包安装 (无需 git)...
    "%PYTHON%" -m pip install --no-deps --ignore-requires-python "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
    if errorlevel 1 (
        echo 错误: indextts 安装失败
        pause
        exit /b 1
    )
)
echo.

echo [4/5] 安装 indextts 缺失依赖...
"%PYTHON%" -m pip install cn2an descript-audiotools fugashi unidic-lite g2p_en json5 keras munch textstat --index-url https://pypi.org/simple/ -q
echo.

echo [5/5] 修复 protobuf 版本...
"%PYTHON%" -m pip install "protobuf>=5.26.1,<6" --index-url https://pypi.org/simple/ -q
echo.

echo ==========================================
echo 安装完成! 请重启 ComfyUI。
echo ==========================================
echo.
echo 注意: IndexTTS-2.5 模型将在首次使用时自动下载到:
echo   ComfyUI\models\IndexTTS2.5\
echo.
pause
