from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import webbrowser

from ego_relation.config import ProjectConfig


class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Development report server: always return the latest generated files."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def serve_reports(cfg: ProjectConfig, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    root = Path(os.path.commonpath([cfg.paths.work_dir.resolve(), cfg.paths.output_dir.resolve()]))
    index = cfg.paths.work_dir / "reports/index.html"
    if not index.is_file():
        raise FileNotFoundError(f"找不到报告首页：{index}。请先运行 ego_relation.cli report 生成 HTML。")
    relative_index = index.resolve().relative_to(root).as_posix()
    handler = partial(NoCacheHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/{relative_index}?v={index.stat().st_mtime_ns}"
    print(f"HTML debug reports: {url}")
    if open_browser:
        webbrowser.open(url)
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
