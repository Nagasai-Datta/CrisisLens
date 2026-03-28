// =============================================================================
// FILE: frontend/src/components/ClusterFeed.jsx
//
// PURPOSE:
//   Zone A — the scrollable incident feed.
//   Renders one ClusterCard per active cluster, sorted by urgency.
//   When selectedClusterId changes, scrolls the matching card into view.
//
// PROPS:
//   clusters          → array of cluster dicts from /api/clusters
//   selectedClusterId → string — currently selected cluster id
//   onSelectCluster   → function(id) — called when card is clicked
//
// WHY CLUSTERS ARE SORTED HERE (not in the backend):
//   The backend already sorts by urgency_score descending.
//   We keep this sort in the frontend too as a safety net —
//   if the backend sort order changes, the UI remains correct.
// =============================================================================

import { useEffect, useRef } from 'react'
import ClusterCard from './ClusterCard'
import './ClusterFeed.css'

export default function ClusterFeed({ clusters, selectedClusterId, onSelectCluster }) {

  // cardRefs: a Map from cluster_id → DOM element reference
  // Used to scroll the selected card into view when map dot is clicked
  const cardRefs = useRef({})

  // When selectedClusterId changes, scroll that card into view
  // This is the Zone A ↔ Zone B synchronisation
  useEffect(() => {
    if (selectedClusterId && cardRefs.current[selectedClusterId]) {
      cardRefs.current[selectedClusterId].scrollIntoView({
        behavior: 'smooth',
        block:    'nearest'   // scroll minimum amount to make card visible
      })
    }
  }, [selectedClusterId])

  // Sort clusters: CRITICAL first, then HIGH, then MODERATE
  // Within the same urgency tier, sort by tweet_count descending
  const urgencyOrder = { CRITICAL: 0, HIGH: 1, MODERATE: 2 }
  const sorted = [...clusters].sort((a, b) => {
    const urgencyDiff = (urgencyOrder[a.urgency_label] ?? 2)
                      - (urgencyOrder[b.urgency_label] ?? 2)
    if (urgencyDiff !== 0) return urgencyDiff
    return b.tweet_count - a.tweet_count
  })

  return (
    <div className="cluster-feed">

      {/* Zone header */}
      <div className="feed-header">
        <h2 className="feed-title">
          Incident Feed
          <span className="feed-count">{clusters.length}</span>
        </h2>
      </div>

      {/* Cluster cards list */}
      <div className="feed-list">
        {sorted.length === 0 ? (
          <div className="feed-empty">
            <div className="feed-empty-icon">⬡</div>
            <div className="feed-empty-text">No active incidents</div>
            <div className="feed-empty-hint">
              Submit tweets using the panel on the left to begin monitoring
            </div>
          </div>
        ) : (
          sorted.map(cluster => (
            <ClusterCard
              key       = {cluster.cluster_id}
              cluster   = {cluster}
              isSelected= {cluster.cluster_id === selectedClusterId}
              onClick   = {() => onSelectCluster(cluster.cluster_id)}
              ref       = {el => {
                // Store the DOM ref for this card by cluster_id
                // Used by scrollIntoView when map dot is clicked
                if (el) cardRefs.current[cluster.cluster_id] = el
                else    delete cardRefs.current[cluster.cluster_id]
              }}
            />
          ))
        )}
      </div>
    </div>
  )
}