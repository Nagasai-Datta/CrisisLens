// =============================================================================
// FILE: frontend/src/components/CommandersReport.jsx
//
// PURPOSE:
//   Zone C — the commander's situation report.
//   Displays the BART-generated executive summary.
//   Report is generated every 5 minutes by editor.py (Phase 7).
//   Polling is handled in App.jsx — this component just renders what it receives.
//
// PROPS:
//   report → string — the BART-generated summary (empty string until Phase 7)
// =============================================================================

import './CommandersReport.css'

export default function CommandersReport({ report }) {
  return (
    <div className="commanders-report">

      <div className="report-header">
        <h2 className="report-title">Commander's Report</h2>
        <span className="report-badge">Auto-generated · 5 min</span>
      </div>

      <div className="report-body">
        {report ? (
          <p className="report-text">{report}</p>
        ) : (
          <div className="report-empty">
            <div className="report-empty-text">
              Awaiting situation report
            </div>
            <div className="report-empty-hint">
              BART generates a summary when incidents are active.
              Submit tweets to begin.
            </div>
          </div>
        )}
      </div>

    </div>
  )
}