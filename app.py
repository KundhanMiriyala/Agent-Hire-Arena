"""Root-level entrypoint for Hugging Face Spaces.

Spaces expects an importable `app` object at the repository root.
This re-exports the FastAPI app defined in `server.app`.
"""

from server.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
