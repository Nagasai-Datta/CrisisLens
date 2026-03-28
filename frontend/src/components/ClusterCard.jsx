// =============================================================================
// FILE: frontend/src/components/ClusterCard.jsx
//
// PURPOSE:
//   Renders a single incident cluster as an interactive card.
//
// KEY BEHAVIOUR:
//   isSelected   → when true (map dot was clicked), card has a highlight border
//   expand toggle→ shows/hides source_tweets list (for fact-checking)
//   onClick      → tells App to set selectedClusterId (syncs map dot)
//   resolve btn  → calls DELETE /api/clusters/{id}, removes from active store
//                  App's 3-second polling picks up the removal automatically
//
// WHY forwardRef:
//   ClusterFeed needs to call scrollIntoView on each card's DOM element.
//   React's ref system lets parent components access child DOM nodes,
//   but only if the child component explicitly forwards the ref.
//   Without forwardRef here, cardRefs.current[id] would always be null.
// =============================================================================

import { useState, forwardRef } from 'react'
import './ClusterCard.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// forwardRef allows ClusterFeed to attach a ref to this component's root div
const ClusterCard = forwardRef(function ClusterCard({ cluster, isSelected, onClick }, ref) {

  const [isExpanded, setIsExpanded] = useState(false)
  const [isResolving, setIsResolving] = useState(false)  // button loading state

  // Format ISO timestamp to readable local time
  const formatTime = (isoString) => {
    if (!isoString) return '—'
    return new Date(isoString).toLocaleTimeString([], {
      hour:   '2-digit',
      minute: '2-digit'
    })
  }

  // Determine badge class from urgency label
  const badgeClass = {
    CRITICAL: 'badge-critical',
    HIGH:     'badge-high',
    MODERATE: 'badge-moderate',
  }[cluster.urgency_label] || 'badge-moderate'

  // ── Resolve Handler ───────────────────────────────────────────────────
  // Calls DELETE /api/clusters/{cluster_id} to remove this incident.
  // We don't manually update React state here — App's 3-second poll
  // to GET /api/clusters will return the updated list automatically,
  // and the card will disappear within 3 seconds of clicking Resolve.
  const handleResolve = async (e) => {
    e.stopPropagation()   // don't trigger card selection when clicking button
    setIsResolving(true)
    try {
      await fetch(`${API_BASE}/api/clusters/${cluster.cluster_id}`, {
        method: 'DELETE'
      })
      // No need to setState — polling handles the UI update
    } catch (err) {
      console.error('Failed to resolve cluster:', err)
      setIsResolving(false)
    }
  }

  return (
    <div
      ref       = {ref}
      className = {`cluster-card ${isSelected ? 'selected' : ''} urgency-${cluster.urgency_label?.toLowerCase()}`}
      onClick   = {onClick}
    >
      {/* ── Card Header ────────────────────────────────────────────── */}
      <div className="card-header">
        <div className="card-header-left">
          <span className={`badge ${badgeClass}`}>
            {cluster.urgency_label}
          </span>
          {cluster.resolved_location && (
            <span className="card-location">
              📍 {cluster.resolved_location}
            </span>
          )}
        </div>
        <div className="card-header-right">
          <span className="card-count">
            {cluster.tweet_count} {cluster.tweet_count === 1 ? 'report' : 'reports'}
          </span>
        </div>
      </div>

      {/* ── Representative Tweet ─────────────────────────────────────── */}
      <div className="card-representative">
        {cluster.representative_tweet}
      </div>

      {/* ── Card Footer ─────────────────────────────────────────────── */}
      <div className="card-footer">
        <div className="card-timestamps">
          <span>First: {formatTime(cluster.first_seen)}</span>
          <span className="timestamp-sep">·</span>
          <span>Last: {formatTime(cluster.last_seen)}</span>
        </div>
        <div className="card-footer-right">

          {/* Urgency score shown as a subtle number */}
          <span className="urgency-score">
            {(cluster.urgency_score * 100).toFixed(0)}
          </span>

          {/* Expand toggle — only show if there are source tweets to reveal */}
          {cluster.source_tweets?.length > 1 && (
            <button
              className = "expand-btn"
              onClick   = {e => {
                e.stopPropagation()
                setIsExpanded(prev => !prev)
              }}
            >
              {isExpanded ? '▲ Hide' : `▼ +${cluster.source_tweets.length - 1} more`}
            </button>
          )}

          {/* Resolve button — marks incident as handled, removes from feed */}
          <button
            className = {`resolve-btn ${isResolving ? 'resolving' : ''}`}
            onClick   = {handleResolve}
            disabled  = {isResolving}
            title     = "Mark incident as resolved"
          >
            {isResolving ? '...' : '✓ Resolve'}
          </button>

        </div>
      </div>

      {/* ── Expanded Source Tweets ──────────────────────────────────── */}
      {isExpanded && cluster.source_tweets && (
        <div className="card-source-tweets">
          <div className="source-tweets-label">All reports in this cluster:</div>
          {cluster.source_tweets.map((tweet, i) => (
            <div key={i} className="source-tweet">
              <span className="source-tweet-index">{i + 1}</span>
              <span className="source-tweet-text">{tweet}</span>
            </div>
          ))}
        </div>
      )}

    </div>
  )
})

export default ClusterCard