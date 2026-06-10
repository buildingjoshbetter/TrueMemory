from __future__ import annotations

import sys
import threading
import time


def _start_server(app, host: str, port: int):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _wait_for_server(host: str, port: int, timeout: float = 30.0):
    import httpx
    url = f"http://{host}:{port}/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
            pass
        time.sleep(0.2)
    return False


def main():
    host = "127.0.0.1"
    port = 8484

    from truememory.dashboard.server.app import create_app
    app = create_app()

    server_thread = threading.Thread(
        target=_start_server, args=(app, host, port), daemon=False
    )
    server_thread.start()

    print(f"Starting TrueMemory Dashboard on http://{host}:{port} ...")

    if not _wait_for_server(host, port):
        print("Server failed to start within 30 seconds.", file=sys.stderr)
        sys.exit(1)

    print(f"Dashboard ready at http://{host}:{port}")

    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed. Install with: pip install truememory[dashboard]",
            file=sys.stderr,
        )
        print(f"Dashboard is running at http://{host}:{port}")
        server_thread.join()
        return

    webview.create_window(
        "TrueMemory",
        f"http://{host}:{port}",
        width=1440,
        height=900,
        min_size=(1024, 600),
    )

    webview.start()


if __name__ == "__main__":
    main()
