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

echo [1/6] 检查 Python 环境...
if not exist "%PYTHON%" (
    echo 错误: 找不到 Python: %PYTHON%
    echo 请确认 ComfyUI 的 Python 路径是否正确
    pause
    exit /b 1
)
"%PYTHON%" --version
echo.

echo [2/6] 安装 hatchling 构建工具...
"%PYTHON%" -m pip install hatchling -q
echo.

echo [3/6] 从 GitHub 安装 indextts (多种方式尝试)...
set "PATH=%GIT_PATH%;%PATH%"

echo   方式1: git+https (no-build-isolation)...
"%PYTHON%" -m pip install --no-deps --ignore-requires-python --no-build-isolation "git+https://github.com/index-tts/index-tts.git"
if errorlevel 1 (
    echo   方式1失败，尝试方式2: git+https (build-isolation)...
    "%PYTHON%" -m pip install --no-deps --ignore-requires-python "git+https://github.com/index-tts/index-tts.git"
    if errorlevel 1 (
        echo   方式2失败，尝试方式3: zip包 (build-isolation)...
        "%PYTHON%" -m pip install --no-deps --ignore-requires-python "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
        if errorlevel 1 (
            echo   方式3失败，尝试方式4: zip包 (no-build-isolation)...
            "%PYTHON%" -m pip install --no-deps --ignore-requires-python --no-build-isolation "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
            if errorlevel 1 (
                echo.
                echo 错误: 所有安装方式均失败!
                echo 这通常是网络问题 (无法访问 GitHub)。
                echo.
                echo 手动安装选项:
                echo   1. 使用 VPN/代理后重试
                echo   2. 手动下载 zip: https://github.com/index-tts/index-tts/archive/refs/heads/main.zip
                echo      然后: pip install --no-deps --ignore-requires-python ^<zip路径^>
                pause
                exit /b 1
            )
        )
    )
)
echo.

echo [4/6] 安装 indextts 缺失依赖...
"%PYTHON%" -m pip install cn2an descript-audiotools fugashi unidic-lite g2p_en json5 keras munch textstat -q
echo.

echo [5/6] 修复 protobuf 版本...
"%PYTHON%" -m pip install "protobuf>=5.26.1,<6" -q
echo.

echo [6/6] 验证安装...
"%PYTHON%" -c "import indextts; import os; d=os.path.dirname(indextts.__file__); print('indextts:', d); print('infer_v2_5.py:', os.path.exists(os.path.join(d,'infer_v2_5.py')))"
echo.

echo ==========================================
echo 安装完成! 请重启 ComfyUI。
echo ==========================================
echo.
echo 注意: IndexTTS-2.5 模型将在首次使用时自动下载到:
echo   ComfyUI\models\IndexTTS2.5\
echo.
pause
