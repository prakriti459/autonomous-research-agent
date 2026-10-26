from app.agent.memory import trim_history


def test_history_trimming_keeps_latest_complete_messages() -> None:
    messages = [
        {"role": "user", "content": "Old" * 100},
        {"role": "assistant", "content": "Recent answer"},
        {"role": "user", "content": "Follow-up"},
    ]

    trimmed = trim_history(messages, max_tokens=20)

    assert trimmed == messages[1:]


def test_history_trimming_can_drop_all_messages_when_budget_is_too_small() -> None:
    assert trim_history([{"role": "user", "content": "Question"}], max_tokens=1) == []
