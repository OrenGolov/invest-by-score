import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "unittest", "tests.test_universe"],
    capture_output=True, text=True, timeout=300,
)
output = (result.stdout or "") + (result.stderr or "")
Path("test_output.txt").write_text(output[-4000:], encoding="utf-8")
print("exit:", result.returncode)
print("written to test_output.txt")
