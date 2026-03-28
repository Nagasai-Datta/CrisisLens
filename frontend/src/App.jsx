// =============================================================================
// FILE: frontend/src/App.jsx
//
// PURPOSE:
//   Root component. Owns all shared state. Runs polling intervals.
//   Passes data down to child components via props.
//
// STATE EXPLAINED:
//   clusters         → current active incident list from /api/clusters
//   selectedClusterId→ which cluster is highlighted (map dot clicked)
//   report           → latest BART situation summary from /api/report
//   health           → model status from /api/health (green/red sidebar)
//   isSubmitting     → true while POST /api/process is in flight
//
// POLLING EXPLAINED:
//   We use setInterval inside useEffect to poll the backend.
//   useEffect runs after the component mounts (appears on screen).
//   The return function inside useEffect is cleanup — it runs when the
//   component unmounts (page closes). We clear the interval there to
//   prevent memory leaks.
//
//   Polling /api/clusters every 3 seconds:
//     → gets updated cluster list
//     → setClusters triggers re-render of ClusterFeed and TacticalMap
//
//   Polling /api/report every 5 minutes:
//     → gets latest BART summary
//     → setReport triggers re-render of CommandersReport
//
// =============================================================================

import { useState, useEffect, useCallback } from 'react'
import ControlSidebar    from './components/ControlSidebar'
import ClusterFeed       from './components/ClusterFeed'
import TacticalMap       from './components/TacticalMap'
import CommandersReport  from './components/CommandersReport'
import './App.css'

// Base URL for the FastAPI backend
// In development: localhost:8000
// In production:  set VITE_API_URL environment variable to HuggingFace Space URL
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {

  // ── Shared State ───────────────────────────────────────────────────────
  const [clusters,          setClusters]          = useState([])
  const [selectedClusterId, setSelectedClusterId] = useState(null)
  const [report,            setReport]            = useState('')
  const [health,            setHealth]            = useState({})
  const [isSubmitting,      setIsSubmitting]      = useState(false)
  const [lastUpdated,       setLastUpdated]       = useState(null)


  // ── Fetch Clusters (runs every 3 seconds) ──────────────────────────────
  // useCallback memoises this function so it doesn't get recreated on
  // every render — important because it's used inside setInterval
  const fetchClusters = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/api/clusters`)
      const data = await res.json()
      setClusters(data)
      setLastUpdated(new Date())
    } catch (err) {
      // Backend might be temporarily unreachable — fail silently
      // The UI shows stale data rather than crashing
      console.warn('Failed to fetch clusters:', err)
    }
  }, [])


  // ── Fetch Report (runs every 5 minutes) ────────────────────────────────
  const fetchReport = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/api/report`)
      const data = await res.json()
      if (data.report) setReport(data.report)
    } catch (err) {
      console.warn('Failed to fetch report:', err)
    }
  }, [])


  // ── Fetch Health (runs once on mount) ──────────────────────────────────
  const fetchHealth = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/api/health`)
      const data = await res.json()
      setHealth(data)
    } catch (err) {
      console.warn('Failed to fetch health:', err)
    }
  }, [])


  // ── Polling Setup ──────────────────────────────────────────────────────
  useEffect(() => {
    // Fetch immediately on mount — don't wait for first interval
    fetchClusters()
    fetchReport()
    fetchHealth()

    // Set up polling intervals
    const clusterInterval = setInterval(fetchClusters, 3000)   // every 3 seconds
    const reportInterval  = setInterval(fetchReport,  300000)  // every 5 minutes
    const healthInterval  = setInterval(fetchHealth,  10000)   // every 10 seconds

    // Cleanup: clear intervals when component unmounts (page closed)
    // Without this, intervals keep running even after the component is gone
    return () => {
      clearInterval(clusterInterval)
      clearInterval(reportInterval)
      clearInterval(healthInterval)
    }
  }, [fetchClusters, fetchReport, fetchHealth])


  // ── Submit Tweets to Pipeline ───────────────────────────────────────────
  // Called by ControlSidebar when the user submits tweets
  const handleSubmitTweets = async (tweetText) => {
    if (!tweetText.trim()) return

    // Split by newline — each line is one tweet
    const tweets = tweetText
      .split('\n')
      .map(t => t.trim())
      .filter(t => t.length > 0)

    if (tweets.length === 0) return

    setIsSubmitting(true)
    try {
      const res = await fetch(`${API_BASE}/api/process`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ tweets })
      })

      if (res.ok) {
        // Pipeline ran successfully — fetch updated clusters immediately
        // Don't wait for the next 3-second poll
        await fetchClusters()
      }
    } catch (err) {
      console.error('Failed to submit tweets:', err)
    } finally {
      setIsSubmitting(false)
    }
  }


  // ── Urgency Summary (for sidebar count display) ─────────────────────────
  const urgencyCounts = {
    CRITICAL: clusters.filter(c => c.urgency_label === 'CRITICAL').length,
    HIGH:     clusters.filter(c => c.urgency_label === 'HIGH').length,
    MODERATE: clusters.filter(c => c.urgency_label === 'MODERATE').length,
  }


  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="war-room-layout">

      {/* Left sidebar — controls, health, tweet input */}
      <ControlSidebar
        health         = {health}
        urgencyCounts  = {urgencyCounts}
        isSubmitting   = {isSubmitting}
        lastUpdated    = {lastUpdated}
        onSubmitTweets = {handleSubmitTweets}
      />

      {/* Main content — three zones */}
      <div className="war-room-main">

        {/* Zone A — Cluster Feed (left half of main) */}
        <div className="zone-feed">
          <ClusterFeed
            clusters          = {clusters}
            selectedClusterId = {selectedClusterId}
            onSelectCluster   = {setSelectedClusterId}
          />
        </div>

        {/* Right column — Map + Report stacked */}
        <div className="zone-right">

          {/* Zone B — Tactical Map (top right) */}
          <div className="zone-map">
            <TacticalMap
              clusters          = {clusters}
              selectedClusterId = {selectedClusterId}
              onDotClick        = {setSelectedClusterId}
            />
          </div>

          {/* Zone C — Commander's Report (bottom right) */}
          <div className="zone-report">
            <CommandersReport report={report} />
          </div>

        </div>
      </div>
    </div>
  )
}