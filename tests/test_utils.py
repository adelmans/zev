import os
from unittest.mock import patch

from zev.utils import get_env_context


class TestGetEnvContext:
    def test_includes_os_and_shell_when_shell_is_set(self):
        with patch.dict(os.environ, {"SHELL": "/bin/bash"}, clear=False):
            with patch("platform.platform", return_value="Linux-5.4.0"):
                result = get_env_context()
                assert "OS: Linux-5.4.0" in result
                assert "SHELL: /bin/bash" in result

    def test_returns_os_only_when_no_shell_available(self):
        env = os.environ.copy()
        env.pop("SHELL", None)
        env.pop("COMSPEC", None)
        with patch.dict(os.environ, env, clear=True):
            with patch("platform.platform", return_value="Darwin-21.0"):
                result = get_env_context()
                assert "OS: Darwin-21.0" in result
                assert "SHELL" not in result

    def test_uses_comspec_on_windows_when_shell_not_set(self):
        env = {"COMSPEC": "C:\\Windows\\System32\\cmd.exe"}
        with patch.dict(os.environ, env, clear=True):
            with patch("platform.platform", return_value="Windows-10"):
                result = get_env_context()
                assert "Windows-10" in result
                assert "SHELL: C:\\Windows\\System32\\cmd.exe" in result
