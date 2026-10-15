"""Tee stdout/stderr to a training log file (so console output is also persisted to disk)."""
from __future__ import annotations

import sys
import time


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):           # delegate isatty/fileno/etc. to the real stream
        return getattr(self.stream, name)


def setup_logging(path: str):
    """Redirect stdout+stderr through a tee that also appends to `path`. Returns the file handle."""
    fh = open(path, "a", buffering=1)      # line-buffered
    fh.write("\n===== run start: %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh
