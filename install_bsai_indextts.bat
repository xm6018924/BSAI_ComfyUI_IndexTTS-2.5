@echo off
chcp 65001 >nul 2>&1
setlocal

echo ==========================================
echo BSAI_ComfyUI_IndexTTS-2.5 安装脚本
echo ==========================================
echo.

set "ROOT=%~dp0..\..\..\"
rem 规范化 ROOT 为完整路径（解析掉 ..\ 相对层级）
for %%I in ("%ROOT%") do set "ROOT=%%~fI\"
set "PYTHON="
set "GIT_PATH=C:\Program Files\Git\cmd"

echo [1/6] 自动定位 Python 环境...
rem 定位优先级:
rem   1) 环境变量 PYTHON_EXE 显式指定（最高优先级）
rem   2) 常见目录名: python_embeded / python / python3 / venv / .venv / Python / portable_python
rem   3) 扫描 ROOT 下一级与两级子目录中的 python.exe
if defined PYTHON_EXE set "PYTHON=%PYTHON_EXE%"

if not defined PYTHON (
    for %%D in (python_embeded python python3 venv .venv Python portable_python) do (
        if exist "%ROOT%%%D\python.exe" (
            set "PYTHON=%ROOT%%%D\python.exe"
            goto :found_python
        )
    )
)

if not defined PYTHON (
    for /d %%D in ("%ROOT%*") do (
        if exist "%%D\python.exe" (
            set "PYTHON=%%D\python.exe"
            goto :found_python
        )
    )
)

if not defined PYTHON (
    for /d %%D in ("%ROOT%*") do (
        for /d %%E in ("%%D\*") do (
            if exist "%%E\python.exe" (
                set "PYTHON=%%E\python.exe"
                goto :found_python
            )
        )
    )
)

:found_python
if not defined PYTHON (
    echo 错误: 在以下位置均找不到 python.exe:
    echo   %ROOT%
    echo.
    echo 请确认 ComfyUI 便携版已正确解压，或设置环境变量 PYTHON_EXE 指向 python.exe
    echo 例如: set PYTHON_EXE=C:\path\to\python.exe
    pause
    exit /b 1
)

echo   已定位 Python: %PYTHON%
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

echo [4/6] 应用 transformers 兼容补丁 (patch_indextts.py)...
rem 该补丁让 indextts 2.0.0 适配 transformers >= 4.55 / 5.x（补全被移除的
rem QuantizedCacheConfig / SequenceSummary / forced_decoder_ids / TypicalLogitsWarper
rem 等符号，以及 wetext 缺失时的优雅降级）。补丁幂等，可重复运行。
"%PYTHON%" "%~dp0patch_indextts.py"
echo.

echo [5/6] 安装 indextts 2.5 运行依赖...
rem 说明: 以下为 indextts 2.5 推理真正需要的依赖（torch/transformers 等已由 ComfyUI 便携版自带）。
rem 已排除 keras / descript-audiotools / protobuf:
rem   - 这三者 indextts 2.5 运行时均不 import；
rem   - keras==2.9.0 会拉取 tensorflow 2.9，与 Python 3.14 不兼容，安装即冲突。
rem 先装基础依赖（openai-whisper 为硬性依赖，可正常安装）：
"%PYTHON%" -m pip install openai-whisper cn2an fugashi unidic-lite g2p_en json5 munch textstat -q
rem wetext 也是硬性依赖，但需 C++ 构建工具(MSVC + CMake)才能编译；
rem 若本机无构建工具导致安装失败，indextts 已做优雅降级（不影响基本 TTS），此处忽略错误继续。
"%PYTHON%" -m pip install wetext -q
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
