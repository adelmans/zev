from unittest.mock import patch, MagicMock

import pytest

from zev.llms.types import Command


class TestHandleSelectedOption:
    @patch('zev.command_selector.pyperclip.copy')
    @patch('zev.command_selector.rprint')
    def test_copies_command_to_clipboard(self, mock_rprint, mock_copy):
        from zev.command_selector import handle_selected_option
        command = Command(command="ls -la", short_explanation="List", is_dangerous=False)
        
        handle_selected_option(command)
        
        mock_copy.assert_called_once_with("ls -la")

    @patch('zev.command_selector.pyperclip.copy')
    @patch('zev.command_selector.rprint')
    def test_shows_warning_for_dangerous_command(self, mock_rprint, mock_copy):
        from zev.command_selector import handle_selected_option
        command = Command(
            command="rm -rf /",
            short_explanation="Delete all",
            is_dangerous=True,
            dangerous_explanation="Deletes everything"
        )
        
        handle_selected_option(command)
        
        warning_shown = any(
            "warning" in str(call).lower() or "Deletes everything" in str(call) 
            for call in mock_rprint.call_args_list
        )
        assert warning_shown

    @patch('zev.command_selector.pyperclip.copy')
    @patch('zev.command_selector.rprint')
    def test_no_warning_for_safe_command(self, mock_rprint, mock_copy):
        from zev.command_selector import handle_selected_option
        command = Command(command="ls", short_explanation="List", is_dangerous=False)
        
        handle_selected_option(command)
        
        warning_shown = any("warning" in str(call).lower() for call in mock_rprint.call_args_list)
        assert not warning_shown

    def test_does_nothing_for_none_selection(self):
        from zev.command_selector import handle_selected_option
        handle_selected_option(None)

    def test_does_nothing_for_cancel_selection(self):
        from zev.command_selector import handle_selected_option
        handle_selected_option("Cancel")

    @patch('zev.command_selector.pyperclip.copy')
    @patch('zev.command_selector.rprint')
    @patch('zev.command_selector.questionary.confirm')
    def test_handles_clipboard_unavailable(self, mock_confirm, mock_rprint, mock_copy):
        from zev.command_selector import handle_selected_option
        import pyperclip
        mock_copy.side_effect = pyperclip.PyperclipException("No clipboard")
        mock_confirm.return_value.ask.return_value = False

        command = Command(command="echo test", short_explanation="Echo", is_dangerous=False)
        handle_selected_option(command)

        error_shown = any("clipboard" in str(call).lower() for call in mock_rprint.call_args_list)
        assert error_shown


class TestDisplayOptions:
    @patch('zev.command_selector.questionary.select')
    def test_returns_user_selection(self, mock_select):
        from zev.command_selector import display_options
        import questionary
        
        command = Command(command="ls", short_explanation="List", is_dangerous=False)
        mock_select.return_value.ask.return_value = command
        options = [questionary.Choice("ls", value=command)]
        
        result = display_options(options)
        
        assert result == command

    @patch('zev.command_selector.questionary.select')
    def test_returns_none_on_keyboard_interrupt(self, mock_select):
        from zev.command_selector import display_options
        mock_select.return_value.ask.return_value = None
        
        result = display_options([])
        
        assert result is None
