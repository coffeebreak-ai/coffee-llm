"""Append-only diagnostics. No routing or conversation state lives here."""
import re
from datetime import datetime
from pathlib import Path

_BEARER = re.compile(r"Bearer\s+\S+")
_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]+")
_APIKEY = re.compile(r"apikey_[A-Za-z0-9_]+")


def redact(text):
    text = _BEARER.sub("Bearer [redacted]", text)
    return _APIKEY.sub("[redacted]", _SECRET.sub("[redacted]", text))


class TrafficLog:
    def __init__(self, data_path):
        self.path = Path(data_path) / "traffic.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def access(self, method, path, status, note="", who=""):
        stamp = datetime.now().isoformat(timespec="seconds")
        mark = f"[{who}] " if who else ""
        line = f"{stamp} {mark}{method} {path} {status}"
        if note:
            line += " " + note
        line = redact(line)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write(self, request_text, response_text):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(redact(request_text) + "\n" + redact(response_text) + "\n")

    def text(self):
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""
