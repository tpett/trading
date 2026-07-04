from trading import notify as notify_module
from trading.notify import notify


def test_notify_invokes_osascript_with_escaped_text(monkeypatch):
    calls = []
    monkeypatch.setattr(notify_module.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    notify('run "failed"', "coverage 42%")
    (args,), _kwargs = calls[0][0], calls[0][1]
    assert args[0] == "osascript"
    assert 'run \\"failed\\"' in args[2]
    assert "coverage 42%" in args[2]


def test_notify_never_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError("no osascript here")

    monkeypatch.setattr(notify_module.subprocess, "run", boom)
    notify("title", "message")  # must not raise
