import json
from pathlib import Path

import pytest

from intake_graph import advance

CASES = json.loads(Path("tests/golden_logic.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_graph_decision(case):
    result = advance(case["state"])
    assert result["next_action"] == case["expect"], (
        f"{case['name']}: expected {case['expect']}, got {result['next_action']} ({result['instruction']})"
    )