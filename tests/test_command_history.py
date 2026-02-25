import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from zev.command_history import CommandHistory, CommandHistoryEntry
from zev.llms.types import Command, OptionsResponse


def make_response(command_str: str = "ls") -> OptionsResponse:
    return OptionsResponse(
        commands=[Command(command=command_str, short_explanation="Test", is_dangerous=False)],
        is_valid=True
    )


class TestCommandHistoryEntry:
    def test_serialization_round_trip(self):
        response = make_response("pwd")
        entry = CommandHistoryEntry(query="current dir", response=response)
        
        json_str = entry.model_dump_json()
        restored = CommandHistoryEntry.model_validate_json(json_str)
        
        assert restored.query == entry.query
        assert restored.response.commands[0].command == "pwd"


class TestCommandHistory:
    @pytest.fixture
    def temp_history_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def history(self, temp_history_file):
        with patch.object(Path, 'home', return_value=temp_history_file.parent):
            history = CommandHistory()
            history.path = temp_history_file
            return history

    def test_save_and_retrieve_options(self, history, temp_history_file):
        response = make_response("git status")
        history.save_options("check git", response)
        
        entries = history.get_history()
        
        assert entries is not None
        assert len(entries) == 1
        assert entries[0].query == "check git"
        assert entries[0].response.commands[0].command == "git status"

    def test_get_history_returns_none_when_empty(self, history, temp_history_file):
        temp_history_file.write_text("")
        assert history.get_history() is None

    def test_enforces_max_entries_limit(self, history):
        history.max_entries = 5
        response = make_response("test")
        
        for i in range(10):
            history.save_options(f"query {i}", response)

        entries = history.get_history()
        
        assert len(entries) == 5
        assert entries[0].query == "query 5"
        assert entries[-1].query == "query 9"

    def test_preserves_entry_order(self, history):
        history.save_options("first", make_response("cmd1"))
        history.save_options("second", make_response("cmd2"))

        entries = history.get_history()
        
        assert len(entries) == 2
        assert entries[0].query == "first"
        assert entries[1].query == "second"


class TestCommandHistoryDisplay:
    @pytest.fixture
    def temp_history_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def history(self, temp_history_file):
        with patch.object(Path, 'home', return_value=temp_history_file.parent):
            history = CommandHistory()
            history.path = temp_history_file
            return history

    def test_display_returns_none_for_empty_list(self, history, capsys):
        result = history.display_history_options([])
        
        assert result is None
        captured = capsys.readouterr()
        assert "No command history" in captured.out

    def test_display_returns_none_for_none_input(self, history, capsys):
        result = history.display_history_options(None)
        
        assert result is None
        captured = capsys.readouterr()
        assert "No command history" in captured.out

    @patch('zev.command_history.questionary.select')
    def test_display_returns_selected_entry(self, mock_select, history):
        entries = [
            CommandHistoryEntry(query="query1", response=make_response()),
            CommandHistoryEntry(query="query2", response=make_response()),
        ]
        mock_select.return_value.ask.return_value = entries[0]

        result = history.display_history_options(entries)
        
        assert result == entries[0]

    @patch('zev.command_history.questionary.select')
    def test_show_more_option_appears_when_exceeding_limit(self, mock_select, history):
        entries = [
            CommandHistoryEntry(query=f"query{i}", response=make_response()) 
            for i in range(10)
        ]
        mock_select.return_value.ask.return_value = "Cancel"

        history.display_history_options(entries, show_limit=5)
        
        call_args = mock_select.call_args
        choices = call_args[1]['choices']
        choice_values = [c.value if hasattr(c, 'value') else str(c) for c in choices]
        assert "show_more" in choice_values
