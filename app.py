"""ASGI entrypoint for Vercel (and any ASGI host): exposes the FastAPI ``app`` object.

Kept separate from ``server.py`` because ``server.py`` collides with the ``server/``
package name, which confuses import-based entrypoint detection.
"""

from server.app import create_app

app = create_app()
