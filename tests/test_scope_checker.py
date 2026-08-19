from eczema_rag.Gemini_Scope_Checker import GeminiScopeChecker


class BrokenModels:
    def generate_content(self, **kwargs):
        raise RuntimeError("scope provider unavailable")


class BrokenClient:
    models = BrokenModels()


def checker_with_broken_provider() -> GeminiScopeChecker:
    checker = object.__new__(GeminiScopeChecker)
    checker.client = BrokenClient()
    checker.model_name = "test"
    checker.confidence_threshold = 0.70
    return checker


def test_obvious_eczema_question_has_safe_fallback_when_scope_model_fails():
    decision = checker_with_broken_provider().check(
        "A child has an itchy rash; what diagnostic criteria confirm atopic eczema?"
    )
    assert decision.in_scope is True
    assert decision.status == "fallback_keyword_match"


def test_unrelated_question_remains_blocked_when_scope_model_fails():
    decision = checker_with_broken_provider().check("What is the current stock price?")
    assert decision.in_scope is False
    assert decision.status.startswith("scope_check_error")
