"""Run the whole pipeline: collect data -> preprocess -> train -> generate."""
import subprocess
import sys

steps = [
    [sys.executable, "collect_data.py"],
    [sys.executable, "train.py"],
    [sys.executable, "generate.py", "--play"],
]
for cmd in steps:
    print("\n>>>", " ".join(cmd))
    if subprocess.call(cmd) != 0:
        raise SystemExit("A step failed - see the error above.")
