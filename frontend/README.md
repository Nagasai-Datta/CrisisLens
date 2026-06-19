# CrisisLens — Frontend

React dashboard with three live-linked panels — an incident feed, a map, and a situation summary — all updating in real time via polling.

---

## Overview

The frontend is a React 18 single-page application built with Vite. It polls the FastAPI backend every 3 seconds for updated cluster data and renders the results across three synchronised panels. No business logic lives in the frontend — it displays what the pipeline produces.

---

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx                     # Root component — owns all shared state
│   ├── App.css                     # Dashboard grid layout
│   ├── index.css                   # Global dark theme variables
│   ├── main.jsx                    # Entry point — mounts React, imports Leaflet CSS
│   └── components/
│       ├── ControlSidebar.jsx      # Left panel — health, upload, urgency counts
│       ├── ControlSidebar.css
│       ├── ClusterFeed.jsx         # Incident card list
│       ├── ClusterFeed.css
│       ├── ClusterCard.jsx         # Individual incident card with expand + resolve
│       ├── ClusterCard.css
│       ├── TacticalMap.jsx         # Map panel — Leaflet map with incident dots
│       ├── TacticalMap.css
│       ├── CommandersReport.jsx    # Summary panel — BART situation summary
│       └── CommandersReport.css
├── public/
│   └── favicon.svg
├── index.html
├── vite.config.js
└── package.json
```

---

## Layout

```
┌─────────────┬──────────────────────┬────────────────────┐
│             │                      │                    │
│  SIDEBAR    │   Incident Feed      │   Map              │
│             │                      │                    │
│  Pipeline   │   Scrollable list    │   Leaflet map,     │
│  status     │   of incident cards  │   one dot per      │
│             │                      │   incident         │
│  Urgency    │   Click card →       │                    │
│  counts     │   highlights map dot │   Click dot →      │
│             │                      │   scrolls to card  │
│  File       ├──────────────────────┴────────────────────┤
│  upload     │   Situation Summary                        │
│             │   BART-generated summary, refreshes ~15 min│
└─────────────┴────────────────────────────────────────────┘
```

---

## Component Breakdown

### `App.jsx` — The Root

Owns all shared state and runs all polling intervals. Child components never fetch data themselves — they receive it as props.

**State:**

- `clusters` — the active incident list, polled from `/api/clusters` every 3 seconds
- `selectedClusterId` — which incident is highlighted (shared between map and feed)
- `report` — the situation summary, polled from `/api/report`
- `health` — model status, polled from `/api/health`

**Cross-panel synchronisation:**

```javascript
// One state variable drives both the feed and the map
const [selectedClusterId, setSelectedClusterId] = useState(null);

// Map dot click → sets selectedClusterId
// ClusterCard checks → isHighlighted={card.id === selectedClusterId}
// Both panels re-render simultaneously
```

---

### `ControlSidebar.jsx` — Left Panel

Three sections:

1. **Pipeline Status** — green/red dot per model from `/api/health`
2. **Active Incidents** — count by urgency tier (CRITICAL / HIGH / MODERATE)
3. **File Upload** — choose a `.txt` or `.csv` file, read it client-side with the FileReader API, send lines to `/api/process`

File reading happens entirely in the browser — nothing is uploaded to a server. The extracted lines are sent to the pipeline as a JSON array.

---

### `ClusterCard.jsx` — Incident Card

Displays one incident with:

- Urgency badge (CRITICAL / HIGH / MODERATE)
- Location name from the geocoder
- The most information-rich post in the cluster, as a representative
- Report count and timestamps
- **Expand toggle** — reveals all source posts for verification
- **Resolve button** — calls `DELETE /api/clusters/{id}`; the card disappears within 3 seconds via polling

Uses `forwardRef` so `ClusterFeed` can call `scrollIntoView` when a map dot is clicked.

---

### `TacticalMap.jsx` — Map Panel

Leaflet map centred on Chennai (`[13.0827, 80.2707]`, zoom 11).

- Uses CartoDB dark-matter tiles — matches the dark theme with no extra configuration
- One `CircleMarker` per incident that has valid coordinates
- Dot radius scales with report count (more reports = bigger dot)
- Dot colour encodes urgency (red = CRITICAL, orange = HIGH, blue = MODERATE)
- The selected incident's dot gets a white border and a larger radius
- Clicking a dot calls `onDotClick(cluster_id)`, which sets `selectedClusterId` in App

**Note:** incidents with `lat: null` (location not found in the gazetteer) are silently skipped — no dot is placed.

**Leaflet CSS** must be imported in `main.jsx` before any react-leaflet component renders. Missing it breaks the map layout and controls.

---

### `CommandersReport.jsx` — Summary Panel

Displays the BART-generated situation summary. Shows a placeholder until the first summary is generated (the backend regenerates it roughly every 15 minutes). Polling is handled in `App.jsx`; this component only renders what it receives as props.

---

## Polling Architecture

The frontend uses `setInterval` inside `useEffect` for all data fetching. No WebSockets.

```javascript
useEffect(() => {
  fetchClusters(); // immediate fetch on mount
  const interval = setInterval(fetchClusters, 3000); // then every 3 seconds
  return () => clearInterval(interval); // cleanup on unmount
}, [fetchClusters]);
```

Polling at 3-second intervals is simple, reliable, and behaves identically in local development and production. It also means the frontend transparently handles a backend that is waking from sleep — failed fetches simply show the components as offline until the next successful poll.

---

## Design System

The UI uses CSS custom properties defined in `index.css`. All colours, spacing, and typography are referenced via variables — changing the theme means changing one file.

```css
:root {
  --bg-primary: #0f0f13; /* main background */
  --bg-secondary: #17171f; /* cards and panels */
  --accent: #e94560; /* brand accent */
  --critical: #e74c3c; /* CRITICAL incidents */
  --high: #e67e22; /* HIGH incidents */
  --moderate: #3498db; /* MODERATE incidents */
}
```

**Design direction:** modern and dark — clean cards, subtle borders, colour used purposefully to communicate urgency.

---

## Getting Started

### Prerequisites

- Node.js 18+
- Backend running at `localhost:8000`

### Install and run

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`.

---

## Environment Variables

For production builds, `frontend/.env.production` sets the backend URL:

```
VITE_API_URL=https://nagasaidatta-crisislens-backend-deployment.hf.space
```

In development the variable is unset and the app defaults to `http://localhost:8000`. It is read in components as:

```javascript
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

---

## Deployment (Azure Static Web Apps)

The frontend is deployed to **Azure Static Web Apps** as a static Vite build, via a GitHub Actions workflow in `.github/workflows/`.

- **App location:** `./frontend` **Output location:** `dist`
- The backend URL is baked in at build time from `frontend/.env.production`
- Every push to `main` triggers an automatic rebuild and redeploy to the global CDN

Live dashboard: https://mango-desert-052139500.4.azurestaticapps.net

---

## Available Scripts

```bash
npm run dev      # development server at localhost:5173
npm run build    # production build → dist/
npm run preview  # preview the production build locally
npm run lint     # run ESLint
```

---

## Dependencies

```json
{
  "react": "^18.0.0",
  "react-dom": "^18.0.0",
  "react-leaflet": "^4.0.0",
  "leaflet": "^1.9.0"
}
```

Build tool: Vite 5. No state-management library — React's built-in `useState` and `useEffect` are sufficient for this application's complexity.
