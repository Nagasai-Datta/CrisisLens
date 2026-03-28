import { StrictMode } from 'react'
import { createRoot }  from 'react-dom/client'
import 'leaflet/dist/leaflet.css'    // ← CRITICAL: Leaflet map styles
import './index.css'                  // ← Our global dark theme styles
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)