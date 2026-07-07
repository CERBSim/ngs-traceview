"""Jupyter integration: ``ShowTrace(tracefile)`` embeds the viewer in a cell.

Each call starts a local ``ngs-traceview`` server (the same one behind
``python -m ngs_traceview``, but with ``--no-browser``) in its own subprocess
and returns an :class:`IPython.display.IFrame` pointing at it, so the full app
— timeline, statistics, search, everything — renders inline in the notebook.
"""

import atexit
import os
import subprocess
import sys
import tempfile
import time

_servers: list[subprocess.Popen] = []


def _cleanup():
    for p in _servers:
        try:
            p.terminate()
        except Exception:
            pass


atexit.register(_cleanup)


def stop_servers():
    """Terminate every viewer server started by :func:`ShowTrace`."""
    _cleanup()
    _servers.clear()


def ShowTrace(tracefile, width="100%", height=720, timeout=60):
    """Show a Paje trace in an embedded viewer inside a Jupyter notebook.

    Parameters
    ----------
    tracefile : str
        Path to a ``.trace`` file.
    width, height :
        Size of the embedded iframe (CSS width, pixel height).
    timeout : float
        Seconds to wait for the server to come up.

    Returns an ``IFrame`` (displayed automatically as a cell result). The
    notebook must run locally — the browser needs to reach ``localhost``.
    """
    tracefile = os.path.abspath(os.path.expanduser(str(tracefile)))
    if not os.path.exists(tracefile):
        raise FileNotFoundError(tracefile)

    fd, url_file = tempfile.mkstemp(prefix="ngs_traceview_", suffix=".url")
    os.close(fd)
    os.remove(url_file)
    log_fd, log_path = tempfile.mkstemp(prefix="ngs_traceview_", suffix=".log")

    env = dict(os.environ, NGS_TRACEVIEW_FILE=tracefile, NGAPP_TEST_URL_FILE=url_file)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ngs_traceview", "--no-browser"],
        env=env, stdout=log_fd, stderr=subprocess.STDOUT,
    )
    os.close(log_fd)
    _servers.append(proc)

    deadline = time.time() + timeout
    url = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _read_tail(log_path)
            raise RuntimeError(f"ngs-traceview server exited early:\n{tail}")
        if os.path.exists(url_file):
            txt = open(url_file).read().strip()
            if txt:
                url = txt
                break
        time.sleep(0.25)
    for p in (url_file, log_path):
        try:
            os.remove(p)
        except OSError:
            pass
    if url is None:
        proc.terminate()
        raise TimeoutError("ngs-traceview server did not start within the timeout")

    try:
        from IPython.display import IFrame

        return IFrame(url, width=width, height=height)
    except ImportError:
        print("ngs-traceview running at:", url)
        return url


def _read_tail(path, n=1500):
    try:
        with open(path) as f:
            return f.read()[-n:]
    except OSError:
        return "(no output)"
