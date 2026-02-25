from unittest.mock import patch


class TestHandleSpecialCase:
    @patch('zev.main.setup')
    def test_setup_flag_triggers_setup(self, mock_setup):
        from zev.main import handle_special_case
        assert handle_special_case("--setup") is True
        mock_setup.assert_called_once()

    @patch('zev.main.setup')
    def test_setup_short_flag_triggers_setup(self, mock_setup):
        from zev.main import handle_special_case
        assert handle_special_case("-s") is True
        mock_setup.assert_called_once()

    @patch('zev.main.command_history')
    def test_recent_flag_shows_history(self, mock_history):
        from zev.main import handle_special_case
        assert handle_special_case("--recent") is True
        mock_history.show_history.assert_called_once()

    @patch('zev.main.command_history')
    def test_recent_short_flag_shows_history(self, mock_history):
        from zev.main import handle_special_case
        assert handle_special_case("-r") is True
        mock_history.show_history.assert_called_once()

    @patch('zev.main.show_help')
    def test_help_flag_shows_help(self, mock_help):
        from zev.main import handle_special_case
        assert handle_special_case("--help") is True
        mock_help.assert_called_once()

    @patch('zev.main.show_help')
    def test_help_short_flag_shows_help(self, mock_help):
        from zev.main import handle_special_case
        assert handle_special_case("-h") is True
        mock_help.assert_called_once()

    def test_returns_false_for_empty_input(self):
        from zev.main import handle_special_case
        assert handle_special_case("") is False
        assert handle_special_case(None) is False
        assert handle_special_case([]) is False

    def test_returns_false_for_multiple_args(self):
        from zev.main import handle_special_case
        assert handle_special_case(["arg1", "arg2"]) is False
        assert handle_special_case("arg1 arg2") is False

    def test_returns_false_for_regular_query(self):
        from zev.main import handle_special_case
        assert handle_special_case("list all files") is False
        assert handle_special_case("how to delete a file") is False

    def test_flags_are_case_insensitive(self):
        from zev.main import handle_special_case
        with patch('zev.main.show_help'):
            assert handle_special_case("--HELP") is True
            assert handle_special_case("--Help") is True
            assert handle_special_case("-H") is True
