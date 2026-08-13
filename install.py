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


def run_pip_install(packages, use_official_pypi=False):
    """Install packages via pip."""
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    if use_official_pypi:
        cmd += ["--index-url", "https://pypi.org/simple/"]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: pip install returned code {result.returncode}")
        if result.stderr:
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


def try_install_indextts():
    """Try multiple methods to install indextts from GitHub."""
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

    # Method 1: Git install (no build isolation, needs hatchling)
    print("  Method 1: git+https (no-build-isolation)...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-deps", "--ignore-requires-python", "--no-build-isolation",
        "git+https://github.com/index-tts/index-tts.git"
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print("  OK (method 1)")
        return True
    print(f"  Failed: {result.stderr[-300:] if result.stderr else 'no stderr'}")

    # Method 2: Git install (with build isolation, pip auto-installs hatchling)
    print("  Method 2: git+https (build-isolation)...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-deps", "--ignore-requires-python",
        "git+https://github.com/index-tts/index-tts.git"
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print("  OK (method 2)")
        return True
    print(f"  Failed: {result.stderr[-300:] if result.stderr else 'no stderr'}")

    # Method 3: Zip archive (no git needed, with build isolation)
    print("  Method 3: zip archive (build-isolation)...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-deps", "--ignore-requires-python",
        "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print("  OK (method 3)")
        return True
    print(f"  Failed: {result.stderr[-300:] if result.stderr else 'no stderr'}")

    # Method 4: Zip archive (no build isolation, last resort)
    print("  Method 4: zip archive (no-build-isolation)...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-deps", "--ignore-requires-python", "--no-build-isolation",
        "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print("  OK (method 4)")
        return True
    print(f"  Failed: {result.stderr[-300:] if result.stderr else 'no stderr'}")

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
    # Step 1: Check if indextts is already installed AND has infer_v2_5
    # ---------------------------------------------------------------
    indextts_ok = False
    if check_package_installed("indextts"):
        # Check if infer_v2_5 exists
        try:
            import indextts
            indextts_dir = os.path.dirname(indextts.__file__)
            if os.path.exists(os.path.join(indextts_dir, "infer_v2_5.py")):
                print("[1/6] indextts is already installed with infer_v2_5, skipping.")
                indextts_ok = True
            else:
                print("[1/6] indextts installed but infer_v2_5.py missing — reinstalling...")
                # Uninstall old version
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "indextts"],
                    capture_output=True, text=True
                )
        except Exception as e:
            print(f"[1/6] indextts check failed: {e} — reinstalling...")

    if not indextts_ok:
        print("[1/6] Installing hatchling build tool...")
        run_pip_install(["hatchling"])

        print()
        print("[2/6] Installing indextts from GitHub source...")
        print("  (indextts is NOT on PyPI, installing from GitHub)")

        if not try_install_indextts():
            print()
            print("  ERROR: All installation methods failed!")
            print("  This is usually a network issue (cannot reach GitHub).")
            print()
            print("  Manual installation options:")
            print("    1. Run install_bsai_indextts.bat")
            print("    2. Use a VPN/proxy and retry")
            print("    3. Download the zip manually from:")
            print("       https://github.com/index-tts/index-tts/archive/refs/heads/main.zip")
            print("       Then: pip install --no-deps --ignore-requires-python <path_to_zip>")
            return False

    # ---------------------------------------------------------------
    # Step 3: Install missing indextts dependencies
    # ---------------------------------------------------------------
    print()
    print("[3/6] Installing indextts dependencies...")

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
        run_pip_install(missing_deps)
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
        major = int(proto_version.split('.')[0])
        if major < 4 or major > 6:
            print("  Fixing protobuf version (targeting 5.x)...")
            run_pip_install(["protobuf>=5.26.1,<6"])
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
            import indextts
            indextts_dir = os.path.dirname(indextts.__file__)
            compat_src = os.path.join(os.path.dirname(__file__), "indextts_compat.py")
            compat_dst = os.path.join(indextts_dir, "_compat.py")

            if os.path.exists(compat_src):
                shutil.copy2(compat_src, compat_dst)
                print(f"  Copied compatibility shim to: {compat_dst}")

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
    # Step 6: Verify installation and register model directory
    # ---------------------------------------------------------------
    print()
    print("[6/6] Verifying installation...")

    # Verify indextts can be imported
    if check_package_installed("indextts"):
        try:
            import indextts
            indextts_dir = os.path.dirname(indextts.__file__)
            has_v25 = os.path.exists(os.path.join(indextts_dir, "infer_v2_5.py"))
            print(f"  indextts package: {indextts_dir}")
            print(f"  infer_v2_5.py present: {has_v25}")
            if not has_v25:
                print("  WARNING: infer_v2_5.py not found! Installation may be incomplete.")
        except Exception as e:
            print(f"  WARNING: indextts import error: {e}")
    else:
        print("  WARNING: indextts still not importable!")

    # Set up model directory
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
