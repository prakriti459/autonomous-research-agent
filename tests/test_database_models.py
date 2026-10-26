from app.db.models import Base


def test_expected_database_tables_are_declared() -> None:
    assert set(Base.metadata.tables) == {
        "sessions",
        "messages",
        "papers",
        "request_logs",
    }


def test_paper_metadata_fields_are_persisted() -> None:
    columns = Base.metadata.tables["papers"].columns

    assert {"title", "authors", "year", "file_name"} <= set(columns.keys())
