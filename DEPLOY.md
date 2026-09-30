# Deployment notes

## Vercel (FastAPI dashboard + API)
- Entrypoint: `app.py` (`app = create_app()`), configured via `[tool.vercel] entrypoint = "app:app"`.
- Dependencies come from `pyproject.toml` (`[project]` table) and the committed `uv.lock`.
  After changing dependencies run `uv lock` and commit `uv.lock`.
- The project filesystem is read-only on Vercel: the solution zip is written to `/tmp`, the
  startup benchmark run is skipped, and live `/api/simulate` runs are capped (30 agents, 120 ticks).
  Override with `ADSO_MAX_AGENTS` / `ADSO_MAX_TICKS` / `ADSO_ALLOWED_ORIGINS` env vars.
- Set `ADSO_ALLOWED_ORIGINS` to your Vercel domain if a separate frontend calls the API.

## Streamlit Community Cloud
- Main file path: `streamlit_app.py`.
- Dependencies: Community Cloud uses `uv.lock` first, so it installs the same pinned set
  (Streamlit is preinstalled by the platform).
- Live simulations are capped at 30 agents / 120 ticks for free-tier resources.

## Keeping dependency files in sync
`pyproject.toml` is the source of truth. `requirements.txt` is only used by the Dockerfile.
