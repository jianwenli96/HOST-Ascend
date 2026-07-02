import py_compile
import sys
import os

files_to_check = [
    "tcc/deterministic_alignment.py",
    "tcc/losses.py",
    "train.py",
    "utils.py"
]

has_error = False

print("Checking syntax for modified files...")
for file_path in files_to_check:
    if os.path.exists(file_path):
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"[OK] {file_path}")
        except py_compile.PyCompileError as e:
            print(f"[ERROR] {file_path}:\n{e}")
            has_error = True
        except Exception as e:
            print(f"[ERROR] {file_path}: Unexpected error {e}")
            has_error = True
    else:
        print(f"[WARNING] File not found: {file_path}")

if has_error:
    sys.exit(1)
else:
    print("All checked files adhere to Python syntax.")
    sys.exit(0)
