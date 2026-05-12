"""Lightweight dev server — zero external dependencies.
Mounts frontend/ at /ui to match the FastAPI production layout.
"""
import http.server
import os
import sys
import urllib.parse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
PREFIX = "/ui"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        self._serve()

    def do_GET(self):
        self._serve()

    def _serve(self):
        parsed = urllib.parse.urlparse(self.path)
        raw = urllib.parse.unquote(parsed.path)

        if raw == "/" or raw == PREFIX or raw == PREFIX + "/":
            raw = PREFIX + "/index.html"

        if raw.startswith(PREFIX + "/"):
            rel = raw[len(PREFIX) + 1:].lstrip("/")
        else:
            rel = raw.lstrip("/")

        fpath = os.path.normpath(os.path.join(FRONTEND, rel))
        if not fpath.startswith(FRONTEND):
            self.send_error(403)
            return

        if not os.path.isfile(fpath):
            self.send_error(404)
            return

        ext = os.path.splitext(fpath)[1].lower()
        mime = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".mjs": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".ttf": "font/ttf",
        }.get(ext, "application/octet-stream")

        try:
            size = os.path.getsize(fpath)
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if self.command == "GET":
                with open(fpath, "rb") as f:
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, fmt, *args):
        pass  # quiet


if __name__ == "__main__":
    with http.server.HTTPServer(("0.0.0.0", PORT), Handler) as srv:
        print(f"Serving at http://localhost:{PORT}")
        print(f"  / and /ui  ->  {FRONTEND}")
        print("Press Ctrl+C to stop.")
        srv.serve_forever()
