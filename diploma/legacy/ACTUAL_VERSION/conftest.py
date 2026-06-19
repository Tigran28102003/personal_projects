"""Делает корень проекта (ACTUAL_VERSION) импортируемым в тестах.

Pytest добавляет директорию этого conftest в sys.path, поэтому `import metrics`,
`import validation` и т.п. работают из tests/ без установки пакета.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
