import os
import sys

print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Executable: {sys.executable}")

try:
    import pandas
    print("✅ Pandas is installed in this environment!")
except ImportError:
    print("❌ Pandas NOT found. You are likely still using the wrong interpreter.")