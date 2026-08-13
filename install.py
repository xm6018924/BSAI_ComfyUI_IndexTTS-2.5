"""
BSAI_ComfyUI_IndexTTS-2.5 - Automated Installation Script
ComfyUI-Manager runs this script automatically after cloning the repository.

Handles:
  1. Installing hatchling build tool
  2. Installing indextts from GitHub source (not on PyPI)
  3. Installing missing indextts dependencies
  4. Fixing protobuf version conflicts
  5. Copying compatibility shim to indextts package
"""

import subprocess
import sys
import os
import shutil
import platform

# Use official PyPI for packages missing from Chinese mirrors
PYPI_OFFICIAL = "https://pypi.org/simple/"


def run_pip_install(packages, use_official_pypi=False):
    """Install packages via pip."""
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    if use_official_pypi:
        cmd += ["--index-url", PYPI_OFFICIAL]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: pip install returned code {result.returncode}")
        if result.stderr:
            # Print last few lines of stderr
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
    else:
        print("  OK")
    return result.returncode == 0


def check_package_installed(package_name):
    """Check if a package is already installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {package_name}"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    print("=" * 60)
    print("BSAI_ComfyUI_IndexTTS-2.5 Installation")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

    # ---------------------------------------------------------------
    # Step 1: Check if indextts is already installed
    # ---------------------------------------------------------------
    if check_package_installed("indextts"):
        print("[1/6] indextts is already installed, skipping.")
    else:
        print("[1/6] Installing hatchling build tool...")
        run_pip_install(["hatchling"], use_official_pypi=True)

        print()
        print("[2/6] Installing indextts from GitHub source...")
        print("  (indextts is NOT on PyPI, installing from GitHub)")
        # Try with git in PATH
        env = os.environ.copy()
        # Common git locations on Windows
        git_paths = [
            r"C:\Program Files\Git\cmd",
            r"C:\Program Files (x86)\Git\cmd",
            os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd"),
        ]
        for gp in git_paths:
            if os.path.isdir(gp):
                env["PATH"] = gp + os.pathsep + env.get("PATH", "")
                break

        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-deps",
            "--ignore-requires-python",
            "--no-build-isolation",
            "git+https://github.com/index-tts/index-tts.git"
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"  git install failed, trying zip archive (no git required)...")
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--no-deps",
                "--ignore-requires-python",
                "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
            ]
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"  ERROR: Failed to install indextts")
            print(f"  stderr: {result.stderr[-500:] if result.stderr else 'N/A'}")
            print()
            print("  Try manual installation:")
            print("    1. pip install --no-deps --ignore-requires-python https://github.com/index-tts/index-tts/archive/refs/heads/main.zip")
            print("    2. Or run install_bsai_indextts.bat")
            return False

        print("  OK")

    # ---------------------------------------------------------------
    # Step 3: Install missing indextts dependencies
    # ---------------------------------------------------------------
    print()
    print("[3/6] Installing indextts dependencies...")

    # Check which dependencies are missing
    missing_deps = []
    dep_check = {
        "cn2an": "cn2an",
        "descript_audiotools": "descript-audiotools",
        "fugashi": "fugashi",
        "unidic_lite": "unidic-lite",
        "g2p_en": "g2p_en",
        "json5": "json5",
        "keras": "keras",
        "munch": "munch",
        "textstat": "textstat",
    }
    for import_name, pip_name in dep_check.items():
        if not check_package_installed(import_name):
            missing_deps.append(pip_name)

    if missing_deps:
        print(f"  Missing: {', '.join(missing_deps)}")
        run_pip_install(missing_deps, use_official_pypi=True)
    else:
        print("  All dependencies already installed.")

    # ---------------------------------------------------------------
    # Step 4: Fix protobuf version
    # ---------------------------------------------------------------
    print()
    print("[4/6] Checking protobuf version...")
    try:
        import google.protobuf
        proto_version = google.protobuf.__version__
        print(f"  Current protobuf: {proto_version}")
        # If protobuf is < 4 or > 7, fix it
        major = int(proto_version.split('.')[0])
        if major < 4 or major > 6:
            print("  Fixing protobuf version (targeting 5.x)...")
            run_pip_install(["protobuf>=5.26.1,<6"], use_official_pypi=True)
        else:
            print("  protobuf version OK")
    except Exception as e:
        print(f"  Could not check protobuf: {e}")

    # ---------------------------------------------------------------
    # Step 5: Install compatibility shim for transformers 5.x
    # ---------------------------------------------------------------
    print()
    print("[5/6] Installing transformers 5.x compatibility shim...")

    try:
        import transformers
        tf_version = transformers.__version__
        major_tf = int(tf_version.split('.')[0])
        print(f"  transformers version: {tf_version}")

        if major_tf >= 5:
            # Find indextts package location
            import indextts
            indextts_dir = os.path.dirname(indextts.__file__)
            compat_src = os.path.join(os.path.dirname(__file__), "indextts_compat.py")
            compat_dst = os.path.join(indextts_dir, "_compat.py")

            if os.path.exists(compat_src):
                shutil.copy2(compat_src, compat_dst)
                print(f"  Copied compatibility shim to: {compat_dst}")

                # Patch indextts __init__.py to load compat first
                init_path = os.path.join(indextts_dir, "__init__.py")
                with open(init_path, 'r', encoding='utf-8') as f:
                    init_content = f.read()

                if '_compat' not in init_content:
                    with open(init_path, 'w', encoding='utf-8') as f:
                        f.write("# Load compatibility shim first (for transformers 5.x)\nfrom . import _compat  # noqa: F401\n" + init_content)
                    print("  Patched indextts __init__.py")
                else:
                    print("  indextts __init__.py already patched")
            else:
                print(f"  WARNING: compat source not found: {compat_src}")
        else:
            print("  transformers < 5.x, no compatibility shim needed")
    except ImportError:
        print("  indextts not importable yet, will check on next startup")
    except Exception as e:
        print(f"  WARNING: Could not install compat shim: {e}")

    # ---------------------------------------------------------------
    # Step 6: Register model directory
    # ---------------------------------------------------------------
    print()
    print("[6/6] Setting up model directory...")

    # Try to find ComfyUI base path
    comfyui_base = None
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(current, "folder_paths.py")):
            comfyui_base = current
            break
        current = os.path.dirname(current)

    if comfyui_base:
        model_dir = os.path.join(comfyui_base, "models", "IndexTTS2.5")
        os.makedirs(model_dir, exist_ok=True)
        print(f"  Model directory: {model_dir}")
    else:
        print("  WARNING: Could not locate ComfyUI base path")

    print()
    print("=" * 60)
    print("Installation complete! Please restart ComfyUI.")
    print("=" * 60)
    print()
    print("Notes:")
    print("  - IndexTTS-2.5 models will auto-download on first use")
    print("  - Model location: ComfyUI/models/IndexTTS2.5/")
    print("  - If auto-download fails, download manually from:")
    print("    ModelScope: https://modelscope.cn/models/IndexTeam/IndexTTS-2.5")
    print("    HuggingFace: https://huggingface.co/IndexTeam/IndexTTS-2.5")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
