// =============================================================================
// FILE: frontend/src/components/TacticalMap.jsx
//
// PURPOSE:
//   Zone B — tactical incident map.
//   Renders one circle marker per cluster that has lat/lng coordinates.
//   Clusters with null coordinates (location not in gazetteer) are skipped.
//
// WHY react-leaflet:
//   Leaflet.js is a plain JavaScript map library. react-leaflet wraps it
//   in React components, so we can pass props and state instead of
//   manually calling leaflet's imperative API.
//
// MAP CENTRE:
//   Centred on Chennai (13.0827, 80.2707) at zoom 11.
//   This shows the full city in view on a typical laptop screen.
//
// MARKER SIZE:
//   radius = base + (tweet_count * scale_factor)
//   More reports → bigger circle → visually communicates severity
//
// MARKER COLOUR:
//   CRITICAL → red (#e74c3c)
//   HIGH     → orange (#e67e22)
//   MODERATE → blue (#3498db)
//
// CROSS-ZONE SYNC:
//   When a dot is clicked, onDotClick(cluster_id) is called.
//   This sets selectedClusterId in App.
//   ClusterFeed receives selectedClusterId and highlights + scrolls the card.
//
// IMPORTANT — Leaflet marker icon fix:
//   Leaflet's default marker icons use image files that Vite's bundler
//   can't find automatically. We use CircleMarker (pure CSS circles)
//   instead of the default pin markers to avoid this issue entirely.
// =============================================================================

import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import './TacticalMap.css'

// Chennai centre coordinates
const CHENNAI_CENTER = [13.0827, 80.2707]
const DEFAULT_ZOOM   = 11

// Marker size configuration
const BASE_RADIUS    = 8    // minimum dot size (pixels)
const SCALE_FACTOR   = 1.5  // extra radius per tweet in cluster

// Marker colours by urgency
const URGENCY_COLOURS = {
  CRITICAL: '#e74c3c',
  HIGH:     '#e67e22',
  MODERATE: '#3498db',
}

export default function TacticalMap({ clusters, selectedClusterId, onDotClick }) {

  // Only render clusters that have valid coordinates
  const mappableClusters = clusters.filter(
    c => c.lat !== null && c.lng !== null
  )

  const getMarkerRadius = (tweetCount) =>
    BASE_RADIUS + (tweetCount * SCALE_FACTOR)

  const getMarkerColour = (urgencyLabel) =>
    URGENCY_COLOURS[urgencyLabel] || URGENCY_COLOURS.MODERATE

  return (
    <div className="tactical-map-container">

      {/* Zone header */}
      <div className="map-header">
        <h2 className="map-title">
          Tactical Map
          <span className="map-count">
            {mappableClusters.length} / {clusters.length} plotted
          </span>
        </h2>
      </div>

      {/* Leaflet map */}
      {/* MapContainer must have explicit height — it doesn't inherit from CSS */}
      <MapContainer
        center    = {CHENNAI_CENTER}
        zoom      = {DEFAULT_ZOOM}
        className = "leaflet-map"
        style     = {{ height: 'calc(100% - 48px)', width: '100%' }}
      >
        {/* OpenStreetMap tile layer — free, no API key required */}
        {/* CartoDB dark matter tiles match our dark theme */}
        <TileLayer
          url         = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution = '&copy; <a href="https://carto.com">CARTO</a> &copy; OpenStreetMap contributors'
          maxZoom     = {19}
        />

        {/* One circle marker per geocoded cluster */}
        {mappableClusters.map(cluster => {
          const isSelected = cluster.cluster_id === selectedClusterId
          const colour     = getMarkerColour(cluster.urgency_label)
          const radius     = getMarkerRadius(cluster.tweet_count)

          return (
            <CircleMarker
              key       = {cluster.cluster_id}
              center    = {[cluster.lat, cluster.lng]}
              radius    = {isSelected ? radius + 4 : radius}
              pathOptions = {{
                color:       isSelected ? '#ffffff' : colour,
                fillColor:   colour,
                fillOpacity: isSelected ? 0.95 : 0.75,
                weight:      isSelected ? 3 : 1.5,
              }}
              eventHandlers = {{
                click: () => onDotClick(cluster.cluster_id)
              }}
            >
              {/* Popup appears on hover — shows key info */}
              <Popup>
                <div className="map-popup">
                  <div className="popup-urgency">
                    {cluster.urgency_label}
                  </div>
                  <div className="popup-location">
                    {cluster.resolved_location || 'Unknown location'}
                  </div>
                  <div className="popup-tweet">
                    {cluster.representative_tweet?.substring(0, 100)}
                    {cluster.representative_tweet?.length > 100 ? '...' : ''}
                  </div>
                  <div className="popup-count">
                    {cluster.tweet_count} reports
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          )
        })}
      </MapContainer>
    </div>
  )
}