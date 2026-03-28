# CrisisLens — Frontend

React War Room dashboard. Three live-linked zones — incident feed, tactical map, and commander's report — all updating in real time via polling.

---

## Overview

The frontend is a React 18 single-page application built with Vite. It polls the FastAPI backend every 3 seconds for updated cluster data and renders the results across three synchronised zones. No business logic lives in the frontend — it displays what the pipeline produces.

---

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx                     # Root component — owns all shared state
│   ├── App.css                     # War Room grid layout
│   ├── index.css                   # Global dark theme variables
│   ├── main.jsx                    # Entry point — mounts React, imports Leaflet CSS
│   └── components/
│       ├── ControlSidebar.jsx      # Left panel — health, upload, urgency counts
│       ├── ControlSidebar.css
│       ├── ClusterFeed.jsx         # Zone A — scrollable incident card list
│       ├── ClusterFeed.css
│       ├── ClusterCard.jsx         # Individual incident card with expand + resolve
│       ├── ClusterCard.css
│       ├── TacticalMap.jsx         # Zone B — Leaflet map with incident dots
│       ├── TacticalMap.css
│       ├── CommandersReport.jsx    # Zone C — BART situation summary
│       └── CommandersReport.css
├── public/
│   └── favicon.svg
├── index.html
├── vite.config.js
└── package.json
```

---

## War Room Layout

```
┌─────────────┬──────────────────────┬────────────────────┐
│             │                      │                    │
│  SIDEBAR    │   ZONE A             │   ZONE B           │
│             │   Cluster Feed       │   Tactical Map     │
│  Pipeline   │                      │                    │
│  status     │   Scrollable list    │   Leaflet map      │
│             │   of incident cards  │   one dot per      │
│  Urgency    │                      │   cluster          │
│  counts     │   Click card →       │                    │
│             │   highlights map dot │   Click dot →      │
│  File       │                      │   scrolls to card  │
│  upload     ├──────────────────────┴────────────────────┤
│             │   ZONE C — Commander's Report              │
│             │   BART-generated situation summary         │
│             │   Auto-refreshes every 5 minutes          │
└─────────────┴────────────────────────────────────────────┘
```

---

## Component Breakdown

### `App.jsx` — The Root

Owns all shared state and runs all polling intervals. Child components never fetch data themselves — they receive it as props.

**State:**

- `clusters` — the active incident list, polled from `/api/clusters` every 3 seconds
- `selectedClusterId` — which incident is highlighted (shared between map and feed)
- `report` — BART summary, polled from `/api/report` every 5 minutes
- `health` — model status, polled from `/api/health` every 10 seconds

**Cross-zone synchronisation:**

```javascript
// One state variable drives both Zone A and Zone B
const [selectedClusterId, setSelectedClusterId] = useState(null);

// Map dot click → sets selectedClusterId
// ClusterCard checks → isHighlighted={card.id === selectedClusterId}
// Both zones re-render simultaneously
```

---

### `ControlSidebar.jsx` — Left Panel

Three sections:

1. **Pipeline Status** — green/red dot per model from `/api/health`
2. **Active Incidents** — count by urgency tier (CRITICAL / HIGH / MODERATE)
3. **File Upload** — choose a `.txt` or `.csv` file, reads it client-side with FileReader API, sends lines to `/api/process`

File reading happens entirely in the browser — no upload to a server. The extracted tweet lines are sent to the pipeline as a JSON array.

---

### `ClusterCard.jsx` — Incident Card

Displays one cluster with:

- Urgency badge (CRITICAL / HIGH / MODERATE)
- Location name from geocoder
- Representative tweet (the most information-rich tweet in the cluster)
- Tweet count and timestamps
- **Expand toggle** — reveals all source tweets for fact-checking
- **Resolve button** — calls `DELETE /api/clusters/{id}`, card disappears within 3 seconds via polling

Uses `forwardRef` so `ClusterFeed` can call `scrollIntoView` when a map dot is clicked.

---

### `TacticalMap.jsx` — Zone B

Leaflet map centred on Chennai (`[13.0827, 80.2707]`, zoom 11).

- Uses CartoDB dark matter tiles — matches the dark theme without configuration
- One `CircleMarker` per cluster that has valid coordinates
- Dot radius scales with `tweet_count` (more reports = bigger dot)
- Dot colour encodes urgency (red = CRITICAL, orange = HIGH, blue = MODERATE)
- Selected cluster dot gets a white border + larger radius
- Clicking a dot calls `onDotClick(cluster_id)` which sets `selectedClusterId` in App

**Important:** Clusters with `lat: null` (location not found in gazetteer) are silently skipped — no dot placed.

**Leaflet CSS:** Must be imported in `main.jsx` before any react-leaflet component renders. Missing it breaks map layout and controls.

---

### `CommandersReport.jsx` — Zone C

Displays the BART-generated situation report. Shows a placeholder until the first report is generated (5 minutes after the first tweet submission). Polling is handled in App.jsx — this component only renders what it receives as props.

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

**Why polling instead of WebSockets:**
HuggingFace Spaces free tier does not support WebSocket connections. Polling at 3-second intervals is imperceptible in a disaster response context and works identically in local development and production.

---

## Design System

The UI uses CSS custom properties defined in `index.css`. All colours, spacing, and typography are referenced via variables — changing the theme means changing one file.

```css
:root {
  --bg-primary: #0f0f13; /* main background */
  --bg-secondary: #17171f; /* cards and panels */
  --accent: #e94560; /* brand red */
  --critical: #e74c3c; /* CRITICAL incidents */
  --high: #e67e22; /* HIGH incidents */
  --moderate: #3498db; /* MODERATE incidents */
}
```

**Design direction:** Modern dark — think Vercel dashboard or Linear app. Clean cards, subtle borders, colour used purposefully for urgency communication. Not a military terminal aesthetic.

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

Create `frontend/.env.production` for production builds:

```
VITE_API_URL=https://YOUR_HF_USERNAME-crisislens-backend.hf.space
```

In development, `VITE_API_URL` is not set — the app defaults to `http://localhost:8000`.

The variable is accessed in components as:

```javascript
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

---

## Deployment (Vercel)

The frontend deploys to Vercel free tier as a static build.

### Steps

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **New Project** → import the `crisislens` repository
3. Set **Root Directory** to `frontend`
4. Add environment variable:
   ```
   VITE_API_URL = https://YOUR_HF_USERNAME-crisislens-backend.hf.space
   ```
5. Click **Deploy**

Vercel detects Vite automatically. Build completes in ~60 seconds. Subsequent pushes to `main` auto-deploy.

---

## Available Scripts

```bash
npm run dev      # Start development server at localhost:5173
npm run build    # Production build → dist/
npm run preview  # Preview production build locally
npm run lint     # Run ESLint
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

Build tool: Vite 5. No state management library — React's built-in `useState` and `useEffect` are sufficient for this application's complexity.
