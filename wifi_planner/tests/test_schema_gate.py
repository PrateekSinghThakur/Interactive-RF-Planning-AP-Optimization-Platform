from pathlib import Path
import subprocess
import sys


def test_validation_models_pass_schema_gate():
    result = subprocess.run(
        [sys.executable, "scripts/validate_models.py", "validation_models"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS") == 5
