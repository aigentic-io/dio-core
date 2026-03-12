"""DIO REST API server (optional module).

Install: pip install aigentic[server]
Run:     uvicorn aigentic.server.app:app --reload
"""

from aigentic.server.app import app

__all__ = ["app"]
