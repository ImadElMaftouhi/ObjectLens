# ObjectLens — Frontend

This document describes the frontend application located in this folder. It explains the project structure, key files and components, developer workflow, and deployment notes.

## Overview

The frontend is a Vite + React app using Tailwind CSS for styling. It provides the UI for uploading/searching objects and viewing 2D/3D previews. The app communicates with the backend via `src/api.js`.

## Quick start

Prerequisites:

- Node.js (>= 16)
- npm or yarn

Install dependencies:

```bash
cd frontend
npm install
```

Run dev server:

```bash
npm run dev
```

Build for production:

```bash
npm run build
npm run preview
```

Docker (build & run):

```bash
# Build image
docker build -t objectlens-frontend .

# Run container (serve on port 5173 by default)
docker run -p 5173:5173 objectlens-frontend
```

## Environment

Environment variables are loaded from `.env` at the project root. Typical variables configure the API base URL for requests.

## Project structure

- public/: static files served by Vite
- src/: application source code
  - api.js — API client/wrapper used across the app
  - main.jsx — app bootstrap and router mounting
  - App.jsx — top-level app component
  - Home.jsx — main landing page
  - Pr2d.jsx / Pr3d.jsx — 2D / 3D preview pages
  - components/ModelViewer.jsx — viewer component for 3D models
  - utils/ — small helpers (e.g., `crop.js`)

Files of interest:

- [frontend/src/api.js](frontend/src/api.js#L1) — central API helper; adjust base URL here.
- [frontend/src/main.jsx](frontend/src/main.jsx#L1) — entry point for the React app.
- [frontend/src/App.jsx](frontend/src/App.jsx#L1) — root layout and route definitions.
- [frontend/src/components/ModelViewer.jsx](frontend/src/components/ModelViewer.jsx#L1) — 3D model display logic.

## Styling

Tailwind CSS is configured in `tailwind.config.js` and PostCSS via `postcss.config.js`. Styles are imported in `src/index.css` and `src/App.css`.

## Routing & Pages

- `Home.jsx` — search and dashboard UI
- `Pr2d.jsx` — 2D preview and image operations
- `Pr3d.jsx` — 3D preview using `ModelViewer.jsx`

## Important components

- `ModelViewer.jsx` shows how 3D assets are loaded and rendered in the browser. Review it to change viewer options or add controls.
- `api.js` centralizes fetch logic and endpoints; update here when backend routes change.

## Integrations

- YOLO/model inference is handled by the backend; the frontend sends images or requests to API endpoints.
- `public/raw/` stores static assets referenced by the UI.

## Testing & Linting

- ESLint config is at `eslint.config.js` for lint rules.

## Deployment notes

- The included `Dockerfile` builds the Vite app into a production image. Ensure the backend URL env var is set during build or runtime.
- When deploying behind a reverse proxy, route traffic to the app's served port (default Vite port is 5173).

## Troubleshooting

- If API calls fail, verify the API base URL in [frontend/src/api.js](frontend/src/api.js#L1) and ensure CORS is enabled on the backend.
- If styles don't apply, confirm Tailwind is running and `index.css` is loaded in `main.jsx`.

## How to contribute

1. Fork the repo and create a feature branch.
2. Run the dev server locally with `npm run dev`.
3. Lint and test changes before creating a PR.

---

If you want, I can also add short file-level READMEs or diagram the component tree next.
