"""Make the git repo root importable in tests so ``import v2_microstructure`` resolves.

`v2_microstructure` is a standalone package at the repo root; its sibling legacy library
`ACTUAL_VERSION/` is added to ``sys.path`` by ``v2_microstructure/src/config.py`` on import.
The existing `ACTUAL_VERSION/conftest.py` continues to expose the legacy modules to their tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
