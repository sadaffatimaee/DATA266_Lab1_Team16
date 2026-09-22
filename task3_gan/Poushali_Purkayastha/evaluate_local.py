import runpy
import sys
from pathlib import Path

src = Path(__file__).resolve().parent / "src"
sys.argv = ["run.py", "--stage", "metrics"] + sys.argv[1:]
runpy.run_path(str(src / "run.py"), run_name="__main__")
