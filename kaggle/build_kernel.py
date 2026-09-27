"""Build kaggle/kernel/run_blocking.py (self-contained) from run_template.py and src/fast_*.py.

    python kaggle/build_kernel.py && kaggle kernels push -p kaggle/kernel
"""
import base64
from pathlib import Path

root = Path(__file__).resolve().parent
src = {n: base64.b64encode((root.parent / "src" / n).read_bytes()).decode()
       for n in ("__init__.py", "fast_normalization.py", "fast_blocking.py")}
out = (root / "run_template.py").read_text().replace("__SRC_FILES__", repr(src))
(root / "kernel" / "run_blocking.py").write_text(out)
print("wrote", root / "kernel" / "run_blocking.py")
