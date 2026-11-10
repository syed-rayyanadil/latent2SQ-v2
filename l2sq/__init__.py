"""latent2SQ: decode superquadrics from frozen VGGT tokens."""

import sys
from pathlib import Path

# Run from the repo root, `import vggt` could resolve to the submodule folder instead of the
# package inside it (vggt/vggt/). Putting the submodule first makes `vggt.models` / `vggt.utils`
# importable whether or not vggt is pip-installed.
_VGGT_DIR = str(Path(__file__).resolve().parents[1] / "vggt")
if _VGGT_DIR not in sys.path:
    sys.path.insert(0, _VGGT_DIR)
