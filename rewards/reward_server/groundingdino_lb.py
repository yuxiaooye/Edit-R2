"""
GroundingDINO Load Balancer

Distributes requests in round-robin to multiple GroundingDINO workers, fully transparent to clients.

Usage (automatically called by start_servers.sh):
    python groundingdino_lb.py [--port 12343] [--backends http://localhost:12344,...]

API (identical interface to groundingdino_server.py):
    POST /detect
    GET  /health
"""

import argparse
import itertools
import threading

import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# Backend worker list (populated from command-line arguments)
BACKENDS: list[str] = []

_counter = itertools.count()
_lock = threading.Lock()


def _next_backend() -> str:
    """Thread-safe round-robin selection of the next backend."""
    with _lock:
        idx = next(_counter) % len(BACKENDS)
    return BACKENDS[idx]


@app.route("/detect", methods=["POST"])
def proxy_detect():
    backend = _next_backend()
    try:
        resp = requests.post(
            f"{backend}/detect",
            json=request.get_json(force=True),
            timeout=120,
        )
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"[LB] backend {backend} error: {e}"}), 503


@app.route("/health", methods=["GET"])
def health():
    results = {}
    all_ok = True
    for backend in BACKENDS:
        try:
            resp = requests.get(f"{backend}/health", timeout=5).json()
            results[backend] = resp
            if not resp.get("model_loaded", False):
                all_ok = False
        except Exception as e:
            results[backend] = {"error": str(e)}
            all_ok = False

    return jsonify({
        "status": "healthy" if all_ok else "loading",
        "model_loaded": all_ok,
        "backends": results,
        "num_backends": len(BACKENDS),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundingDINO Load Balancer")
    parser.add_argument("--port", type=int, default=12343, help="LB listening port (default 12343)")
    parser.add_argument(
        "--backends",
        type=str,
        default="http://localhost:12344,http://localhost:12345,http://localhost:12346,http://localhost:12347",
        help="Comma-separated list of backend worker addresses",
    )
    args = parser.parse_args()

    BACKENDS.extend(b.strip() for b in args.backends.split(",") if b.strip())

    print("=" * 60)
    print(f"GroundingDINO Load Balancer  (port={args.port})")
    print(f"Backends ({len(BACKENDS)}):")
    for b in BACKENDS:
        print(f"  {b}")
    print("=" * 60)
    print(f"\nStarting LB on 0.0.0.0:{args.port}...")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
