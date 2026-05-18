# fmt: off
"""
This file is a manual test to check the functionality of the scanner module.

"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from MVL_AI_Classifier.data.scanner import DatasetBuilder  # noqa: E402
# fmt: on
builder = DatasetBuilder(size=1000, root="./data/subset")
builder.scan()
print(f"Length of dataset: {len(builder.data)}")
builder.filter()
print(f"Length of filtered dataset: {len(builder.data)}")
