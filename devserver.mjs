import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.join(__dirname, "frontend");
const PORT = 8765;

const MIME = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".glb": "model/gltf-binary",
  ".obj": "text/plain",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".ico": "image/x-icon",
  ".json": "application/json",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
};

http.createServer((req, res) => {
  // Strip /ui/ prefix (FastAPI serves frontend under /ui/)
  let urlPath = req.url.split("?")[0];
  if (urlPath.startsWith("/ui/")) urlPath = urlPath.slice(3); // becomes /models/... etc
  if (urlPath === "/" || urlPath === "") urlPath = "/index.html";

  const filePath = path.join(FRONTEND, urlPath);

  // Security: prevent directory traversal
  if (!filePath.startsWith(FRONTEND)) {
    res.writeHead(403); res.end("Forbidden"); return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      console.log("404:", urlPath);
      res.writeHead(404); res.end("Not found: " + urlPath); return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
}).listen(PORT, "0.0.0.0", () => {
  console.log(`Dev server running at http://localhost:${PORT}`);
});
