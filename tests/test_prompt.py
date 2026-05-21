from src.prompt import system_prompt


def test_prompt_has_context_placeholder():
    assert "{context}" in system_prompt


def test_prompt_has_safety_disclaimer():
    lower = system_prompt.lower()
    assert "not a substitute for professional medical care" in lower
    assert "emergenc" in lower  # emergency / emergencies


def test_prompt_instructs_uncertainty_and_no_diagnosis():
    lower = system_prompt.lower()
    assert "don't know" in lower or "do not know" in lower
    assert "diagnos" in lower
