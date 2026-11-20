"""Lightweight production web dashboard server for SilentX / TrustGuard."""

import http.server
import socketserver
import os
import sys
import json

PORT = int(os.environ.get("PORT", 8080))
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))


class SilentXDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the SilentX dashboard and JSON APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            payload = {
                "status": "healthy",
                "project": "SilentX",
                "framework": "TrustGuard",
                "commit": "ac92dfd",
                "tests_passing": 32,
                "dataset": "OpenML_Electricity_151",
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        elif self.path == "/api/data":
            json_path = os.path.join(DASHBOARD_DIR, "data.json")
            if os.path.exists(json_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(json_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        return super().do_GET()

    def end_headers(self):
        # Prevent aggressive browser caching during dashboard iteration
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def run_server(port=PORT):
    # Bind to 127.0.0.1 for local security
    server_address = ("127.0.0.1", port)
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(server_address, SilentXDashboardHandler) as httpd:
            print(f"=" * 70)
            print(f"  SILENTX // TRUSTGUARD MONITORING DASHBOARD DEPLOYED")
            print(f"  URL: http://127.0.0.1:{port}/")
            print(f"  Health Check: http://127.0.0.1:{port}/api/health")
            print(f"  Framework: TrustGuard | Commit: ac92dfd")
            print(f"=" * 70)
            sys.stdout.flush()
            httpd.serve_forever()
    except OSError as e:
        if port == 8080:
            print(f"Port 8080 occupied, attempting fallback to port 8000...")
            run_server(8000)
        else:
            raise e


if __name__ == "__main__":
    run_server()
