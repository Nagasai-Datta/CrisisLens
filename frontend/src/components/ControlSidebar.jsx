// =============================================================================
// FILE: frontend/src/components/ControlSidebar.jsx
//
// PURPOSE:
//   Left sidebar panel. Three sections:
//   1. System header — logo and status
//   2. Model health — green/red per component
//   3. Urgency summary — count by tier
//   4. File upload — upload a .txt or .csv of tweets, one per line
//
// WHY FILE UPLOAD INSTEAD OF TEXTAREA:
//   More realistic for actual disaster response use —
//   operators would have collected tweet files, not type manually.
//   Also solves sidebar space constraints cleanly.
//
// HOW FILE READING WORKS:
//   FileReader API reads the file entirely in the browser.
//   No file is sent to the server — just the extracted text lines.
//   Those lines are sent to POST /api/process as a tweets array,
//   exactly the same as the textarea approach did.
// =============================================================================

import { useState, useRef } from 'react'
import './ControlSidebar.css'

export default function ControlSidebar({
  health,
  urgencyCounts,
  isSubmitting,
  lastUpdated,
  onSubmitTweets
}) {
  const [fileName,    setFileName]    = useState('')
  const [fileLines,   setFileLines]   = useState(0)
  const [fileReady,   setFileReady]   = useState(false)
  const [fileContent, setFileContent] = useState('')
  const [fileError,   setFileError]   = useState('')

  // Hidden file input ref — we trigger it programmatically
  // so we can style the button however we want
  const fileInputRef = useRef(null)

  const lastUpdatedStr = lastUpdated
    ? lastUpdated.toLocaleTimeString()
    : 'Never'

const allOk = health.overall === 'healthy'

  // ── File Selection Handler ──────────────────────────────────────────
  // Called when user picks a file from the file picker dialog.
  // Reads the file client-side using FileReader — no server upload.
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return

    // Reset state
    setFileError('')
    setFileReady(false)
    setFileContent('')
    setFileLines(0)

    // Validate file type — accept .txt and .csv only
    const validTypes = ['text/plain', 'text/csv', 'application/csv']
    const validExtensions = ['.txt', '.csv']
    const hasValidExt = validExtensions.some(ext =>
      file.name.toLowerCase().endsWith(ext)
    )

    if (!validTypes.includes(file.type) && !hasValidExt) {
      setFileError('Please upload a .txt or .csv file')
      return
    }

    setFileName(file.name)

    // Read file as plain text
    const reader = new FileReader()

    reader.onload = (event) => {
      const text = event.target.result

      // Split by newline, trim whitespace, remove empty lines
      const lines = text
        .split('\n')
        .map(l => l.trim())
        .filter(l => l.length > 0)
        // If CSV, take only the first column of each row
        // (handles files where tweet is the first field)
        .map(l => l.split(',')[0].replace(/^"|"$/g, '').trim())
        .filter(l => l.length > 5)  // skip very short lines (headers etc.)

      if (lines.length === 0) {
        setFileError('No valid tweets found in file')
        return
      }

      setFileLines(lines.length)
      setFileContent(lines.join('\n'))
      setFileReady(true)
    }

    reader.onerror = () => {
      setFileError('Failed to read file — please try again')
    }

    reader.readAsText(file)
  }

  // ── Submit Handler ─────────────────────────────────────────────────
  const handleSubmit = () => {
    if (!fileReady || !fileContent) return
    onSubmitTweets(fileContent)

    // Reset after submit so user can upload another file
    setFileReady(false)
    setFileName('')
    setFileLines(0)
    setFileContent('')
    // Reset the actual file input element
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // ── Clear Handler ──────────────────────────────────────────────────
  const handleClear = () => {
    setFileReady(false)
    setFileName('')
    setFileLines(0)
    setFileContent('')
    setFileError('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <aside className="sidebar">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <span className="logo-icon">⬡</span>
          <span className="logo-text">CrisisLens</span>
        </div>
        <div className={`system-status ${allOk ? 'ok' : 'degraded'}`}>
          <span className={`status-dot ${allOk ? 'status-dot-ok' : 'status-dot-error'}`} />
          {allOk ? 'All systems operational' : 'System degraded'}
        </div>
        <div className="last-updated">
          Last updated: {lastUpdatedStr}
        </div>
      </div>

      {/* ── Model Health ───────────────────────────────────────────── */}
      <div className="sidebar-section">
        <div className="section-label">Pipeline Status</div>
        <div className="health-grid">
          {['bouncer', 'deduplicator', 'detective', 'geocoder'].map(model => (
            <div key={model} className="health-item">
              <span className={`status-dot ${
                health[model] === 'ok' ? 'status-dot-ok' : 'status-dot-error'
              }`} />
              <span className="health-model-name">{model}</span>
              <span className={`health-status-text ${
                health[model] === 'ok' ? 'text-ok' : 'text-error'
              }`}>
                {health[model] === 'ok' ? 'ready' : 'error'}
              </span>
            </div>
          ))}
        </div>
        {health.active_clusters !== undefined && (
          <div className="health-stats">
            <span>{health.active_clusters} active clusters</span>
            <span>{health.window_size || 0} tweets in window</span>
          </div>
        )}
      </div>

      {/* ── Urgency Summary ─────────────────────────────────────────── */}
      <div className="sidebar-section">
        <div className="section-label">Active Incidents</div>
        <div className="urgency-summary">
          <div className="urgency-item critical">
            <span className="urgency-count">{urgencyCounts.CRITICAL}</span>
            <span className="urgency-label-text">CRITICAL</span>
          </div>
          <div className="urgency-item high">
            <span className="urgency-count">{urgencyCounts.HIGH}</span>
            <span className="urgency-label-text">HIGH</span>
          </div>
          <div className="urgency-item moderate">
            <span className="urgency-count">{urgencyCounts.MODERATE}</span>
            <span className="urgency-label-text">MODERATE</span>
          </div>
        </div>
      </div>

      {/* ── File Upload ─────────────────────────────────────────────── */}
      <div className="sidebar-section upload-section">
        <div className="section-label">Load Tweet File</div>
        <p className="input-hint">
          Upload a .txt or .csv file. One tweet per line.
        </p>

        {/* Hidden native file input */}
        <input
          ref      = {fileInputRef}
          type     = "file"
          accept   = ".txt,.csv,text/plain,text/csv"
          onChange = {handleFileChange}
          style    = {{ display: 'none' }}
        />

        {/* Custom styled upload trigger button */}
        <button
          className = "upload-btn"
          onClick   = {() => fileInputRef.current?.click()}
          disabled  = {isSubmitting}
        >
          ↑ Choose File
        </button>

        {/* File selected — show info and action buttons */}
        {fileName && !fileError && (
          <div className="file-info">
            <div className="file-name">📄 {fileName}</div>
            <div className="file-lines">
              {fileLines} tweets ready to process
            </div>
            <div className="file-actions">
              <button
                className = "clear-file-btn"
                onClick   = {handleClear}
                disabled  = {isSubmitting}
              >
                ✕ Clear
              </button>
              <button
                className = {`submit-btn ${isSubmitting ? 'submitting' : ''}`}
                onClick   = {handleSubmit}
                disabled  = {isSubmitting || !fileReady}
              >
                {isSubmitting ? 'Processing...' : 'Run Pipeline →'}
              </button>
            </div>
          </div>
        )}

        {/* Error state */}
        {fileError && (
          <div className="file-error">
            ⚠️ {fileError}
          </div>
        )}
      </div>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <div className="sidebar-footer">
        <span>Polling every 3s</span>
        <span>{health.gazetteer_size || 0} locations indexed</span>
      </div>

    </aside>
  )
}