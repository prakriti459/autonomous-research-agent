import re


class SuspiciousInputError(ValueError):
    pass


_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b", re.I),
    re.compile(r"\b(reveal|print|show)\s+(?:the\s+)?(?:hidden\s+|system\s+)?prompt\b", re.I),
    re.compile(r"\bact\s+as\s+(?:a\s+)?(?:system|developer|administrator)\b", re.I),
    re.compile(r"\b(jailbreak|DAN\s+mode)\b", re.I),
)


def validate_user_question(question: str) -> None:
    if any(pattern.search(question) for pattern in _INJECTION_PATTERNS):
        raise SuspiciousInputError(
            "Question contains a prompt-injection pattern and was rejected"
        )
