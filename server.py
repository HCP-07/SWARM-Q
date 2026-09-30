"""Mission Control server entrypoint executing the FastAPI application."""

from __future__ import annotations
import os
import uvicorn
from server.app import create_app
from server.routes import create_solution_zip

app = create_app()


def main():
    """Starts the Uvicorn web server listening on configured host and port."""
    import argparse
    parser = argparse.ArgumentParser(description="Start the AD-QPSO Mission Control Server")
    parser.add_argument("--host", type=str, default=os.getenv("ADSO_HOST", "127.0.0.1"), help="Host IP")
    parser.add_argument("--port", type=int, default=int(os.getenv("ADSO_PORT", "8000")), help="Port")
    args = parser.parse_args()

    host = args.host
    port = args.port

    print("=" * 70)
    print("STARTING AD-QPSO AUTONOMOUS SWARM MISSION CONTROL SERVER")
    print("=" * 70)
    print(f"  -> Web Dashboard: http://{host}:{port}")
    print(f"  -> API Docs:      http://{host}:{port}/docs")
    print(f"  -> Solution ZIP:  http://{host}:{port}/api/download/solution.zip")
    print("=" * 70)

    create_solution_zip()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
