"""The logger app: keeps the Arduino connected and serves the app window at
http://127.0.0.1:<port>/ (opened as its own Chrome window when Chrome is installed)."""

import json
import math
import os
import secrets
import signal
import subprocess
import sys
import threading
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from .recorder import (BASELINE_S, State, next_session_number, reader, safe_name,
                       session_dir, today, todays_sessions)
from .sources import find_arduino_port, list_serial_ports, open_serial, simulated_lines

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
REMEMBERED = ("folder", "mouse", "lo", "hi", "auto_ttl")     # kept between runs
APP_ID = "pressure-logger"


class Controller:
    """Owns the shared State and the Arduino connection; every page action lands here."""

    def __init__(self, args):
        self.args = args
        self.st = State()
        self.sid = secrets.token_hex(4)          # lets the page notice an app restart
        self.server = None
        self._stop = self._thread = None
        self._conn_lock = threading.Lock()
        self._settings_path = os.path.join(args.data_dir, ".app_settings.json")
        self._load_settings()
        self.st.saved = todays_sessions(args.data_dir)

    # -------------------------------------------------------------- remembered settings
    def _load_settings(self):
        try:
            with open(self._settings_path) as f:
                saved = json.load(f)
        except (OSError, ValueError):
            return
        st = self.st
        st.folder = safe_name(saved.get("folder", ""), empty="")
        st.mouse = safe_name(saved.get("mouse", ""), empty="")
        for k in ("lo", "hi"):
            if isinstance(saved.get(k), (int, float)):
                setattr(st, k, float(saved[k]))
        st.auto_ttl = bool(saved.get("auto_ttl", True))

    def _save_settings(self):
        with self.st.lock:
            data = {k: getattr(self.st, k) for k in REMEMBERED}
        try:
            with open(self._settings_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"(could not save settings: {e})")

    # -------------------------------------------------------------- Arduino connection
    def connected(self):
        return self._thread is not None and self._thread.is_alive()

    def connect(self, port=None):
        if not self._conn_lock.acquire(blocking=False):
            return "already connecting"
        if self.connected():
            self._conn_lock.release()
            return "already connected"
        with self.st.lock:
            self.st.status = "connecting"
        threading.Thread(target=self._connect, args=(port,), daemon=True).start()
        return "connecting"

    def _connect(self, port):                    # runs in its own thread, holds _conn_lock
        try:
            if self.args.simulate:
                port, lines = "simulated", simulated_lines()
            else:
                port = port or find_arduino_port()
                if not port:
                    raise RuntimeError("no Arduino found - check the USB cable")
                lines = open_serial(port, self.args.baud)
            self.st.new_connection(port)
            cfg = SimpleNamespace(data_dir=self.args.data_dir, port=port,
                                  ttl_mode=self.args.ttl_mode, ttl_timeout=self.args.ttl_timeout)
            self._stop = threading.Event()
            self._thread = threading.Thread(target=reader, args=(lines, cfg, self._stop, self.st),
                                            daemon=True)
            self._thread.start()
            print(f"Connected: {port}")
        except Exception as e:
            with self.st.lock:
                self.st.status = f"error: {e}"
            print(f"Could not connect: {e}")
        finally:
            self._conn_lock.release()

    def disconnect(self):
        with self._conn_lock:                    # waits for a connect in progress
            if self._stop is not None:
                self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5)     # the reader saves an open session on exit
            self._stop = self._thread = None
        return "disconnected"

    def ports(self):
        return [dict(device=p.device, description=p.description) for p in list_serial_ports()]

    # -------------------------------------------------------------- page actions
    def record(self, action):
        if action not in ("start", "stop"):
            raise ValueError("action must be start or stop")
        if not self.connected():
            raise ValueError("not connected to the Arduino")
        with self.st.lock:
            self.st.request = action
        return action

    def set_meta(self, body):
        with self.st.lock:
            if "folder" in body:
                self.st.folder = safe_name(body["folder"], empty="")
            if "mouse" in body:
                self.st.mouse = safe_name(body["mouse"], empty="")
            if "details" in body:
                self.st.details = str(body["details"])[:5000]
            out = dict(folder=self.st.folder, mouse=self.st.mouse)
        self._save_settings()
        return out

    def set_settings(self, body):
        vals = {}
        for k in ("lo", "hi"):
            if k in body:
                try:
                    vals[k] = float(body[k])
                except (TypeError, ValueError):
                    raise ValueError(f"{k} must be a number") from None
                if not math.isfinite(vals[k]):
                    raise ValueError(f"{k} must be a number")
        with self.st.lock:
            for k, v in vals.items():
                setattr(self.st, k, v)
            if "auto_ttl" in body:
                self.st.auto_ttl = bool(body["auto_ttl"])
        self._save_settings()
        return "ok"

    def set_baseline(self):
        st = self.st
        with st.lock:
            if not st.data:
                raise ValueError("no data yet")
            t_end = st.data[-1][1]
            raw = [d[2] for d in st.data if t_end - d[1] <= BASELINE_S]
            st.baseline = sum(raw) / len(raw)
            return round(st.baseline, 2)

    def open_folder(self):
        with self.st.lock:
            folder, mouse = self.st.folder, self.st.mouse or "unknown"
        root = os.path.abspath(self.args.data_dir)
        path = session_dir(root, folder, mouse, today())
        while path != root and not os.path.isdir(path):     # nearest folder that exists
            path = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
        return path

    def quit(self):
        threading.Thread(target=self._quit, daemon=True).start()
        return "bye"

    def _quit(self):
        self.disconnect()
        if self.server is not None:
            self.server.shutdown()

    # -------------------------------------------------------------- polled by the page
    def snapshot(self, since, saved_n):
        st = self.st
        with st.lock:
            pts = []
            for d in reversed(st.data):              # only the samples the page hasn't seen
                if d[0] <= since:
                    break
                pts.append([d[0], round(d[1], 3), round(d[3], 2), d[4]])
            pts.reverse()
            s = dict(sid=self.sid, conn_id=st.conn_id, status=st.status, port=st.port,
                     simulate=self.args.simulate, rate=round(st.rate_hz, 1), has_ttl=st.has_ttl,
                     pulses=st.pulses, calibrated=st.calibrated, baseline=round(st.baseline, 2),
                     lo=st.lo, hi=st.hi, auto_ttl=st.auto_ttl, folder=st.folder, mouse=st.mouse,
                     details=st.details, rec=dict(st.rec) if st.rec else None,
                     saved_n=len(st.saved), points=pts)
            if saved_n != len(st.saved):
                s["saved"] = st.saved[-50:]
            folder, mouse = st.folder, st.mouse
        s["next"] = self._next_file(folder, mouse)
        s["data_dir"] = os.path.abspath(self.args.data_dir)
        return s

    def _next_file(self, folder, mouse):
        mouse, date = mouse or "unknown", today()
        path = session_dir(self.args.data_dir, folder, mouse, date)
        n = next_session_number(path, mouse, date)
        return os.path.relpath(os.path.join(path, f"{mouse}_{date}_S{n:02d}"), self.args.data_dir)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = False       # never share a port with another app
    daemon_threads = True


def make_handler(ctl):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"     # keep-alive for the 10 Hz polling

        def log_message(self, fmt, *args):       # keep the terminal quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _from_this_computer(self):
            return self.headers.get("Host", "").split(":")[0] in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._from_this_computer():
                return self._json({"error": "forbidden"}, 403)
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                try:
                    since, saved = (int(q.get(k, ["-1"])[0]) for k in ("since", "saved"))
                except ValueError:
                    return self._json({"error": "bad request"}, 400)
                self._json(ctl.snapshot(since, saved))
            elif url.path == "/api/ports":
                self._json(ctl.ports())
            elif url.path == "/api/hello":
                self._json({"app": APP_ID})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            # JSON only: other web pages can't send that to localhost without permission
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip()
            if not self._from_this_computer() or ctype != "application/json":
                return self._json({"error": "forbidden"}, 403)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                body = None
            if not isinstance(body, dict):
                return self._json({"error": "bad request"}, 400)
            actions = {
                "/api/connect": lambda: ctl.connect(body.get("port") or None),
                "/api/disconnect": ctl.disconnect,
                "/api/record": lambda: ctl.record(body.get("action")),
                "/api/meta": lambda: ctl.set_meta(body),
                "/api/settings": lambda: ctl.set_settings(body),
                "/api/baseline": ctl.set_baseline,
                "/api/open_folder": ctl.open_folder,
                "/api/quit": ctl.quit,
            }
            fn = actions.get(urlparse(self.path).path)
            if fn is None:
                return self._json({"error": "not found"}, 404)
            try:
                self._json({"ok": True, "result": fn()})
            except ValueError as e:
                self._json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def open_window(url):
    """Open the app as its own window (Chrome app mode) if Chrome is installed, else a tab."""
    chrome = "/Applications/Google Chrome.app"
    if sys.platform == "darwin" and os.path.isdir(chrome):
        subprocess.Popen(["open", "-na", chrome, "--args", f"--app={url}", "--window-size=1320,900"])
    else:
        webbrowser.open(url)


def _is_logger(url):
    try:
        with urllib.request.urlopen(url + "api/hello", timeout=1) as r:
            return json.load(r).get("app") == APP_ID
    except Exception:
        return False


def run_app(args):
    os.makedirs(args.data_dir, exist_ok=True)
    ctl = Controller(args)
    for port in range(args.http_port, args.http_port + 20):     # first free port
        url = f"http://127.0.0.1:{port}/"
        try:
            server = _Server(("127.0.0.1", port), make_handler(ctl))
            break
        except OSError:
            if _is_logger(url):
                print(f"The logger is already running -> opening {url}")
                open_window(url)
                return
    else:
        sys.exit(f"No free local port in {args.http_port}-{args.http_port + 19} for the app window.")
    ctl.server = server
    for name in ("SIGHUP", "SIGTERM"):           # Terminal window closed: save and stop cleanly
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: ctl.quit())
    ctl.connect(None if args.port == "auto" else args.port)
    print(f"Pressure logger: {url}\nData folder: {os.path.abspath(args.data_dir)}\n"
          "Keep this Terminal window open. Stop with the app's Quit button (or Ctrl+C).")
    if args.window:
        threading.Timer(0.5, open_window, (url,)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.disconnect()
        server.server_close()
        print("Logger stopped.")
