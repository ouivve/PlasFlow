import { FormEvent, useEffect, useMemo, useState } from 'react'

type JobStatus = {
  job_id: string
  status: string
  progress: number
  mode: string
  task?: string | null
  read_type?: string | null
  min_length?: number | null
  threshold: number
  requested_mode: string
  used_mode?: string | null
  fallback_reason?: string | null
  error?: string | null
  created_at: string
  started_at?: string | null
  finished_at?: string | null
}

type Artifact = {
  name: string
  path: string
  size_bytes: number
}

type ClassMetrics = {
  macro_f1?: number | null
  precision_macro?: number | null
  recall_macro?: number | null
  accuracy?: number | null
}

type CalibrationMetrics = {
  ece?: number | null
  brier_score?: number | null
  recommended_threshold?: number | null
}

type UncertaintySummary = {
  mean_max_prob?: number | null
  mean_margin?: number | null
  mean_entropy?: number | null
  mean_uncertainty_score?: number | null
}

type ReportSummary = {
  task?: string
  metrics?: {
    binary_domain?: ClassMetrics
    domain4?: ClassMetrics
    calibration?: CalibrationMetrics
  }
  uncertainty_summary?: UncertaintySummary
}

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(2)} MB`
}

function statusTone(status: string | undefined): string {
  if (status === 'completed') return 'ok'
  if (status === 'failed') return 'err'
  if (status === 'running') return 'run'
  return 'idle'
}

function formatMetric(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  return value.toFixed(digits)
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [mode, setMode] = useState<'v1' | 'v2'>('v1')
  const [task, setTask] = useState<'legacy28' | 'binary_domain' | 'domain4'>('legacy28')
  const [readType, setReadType] = useState<'short' | 'long' | 'hybrid' | 'complete'>('short')
  const [minLength, setMinLength] = useState(1000)
  const [threshold, setThreshold] = useState(0.7)
  const [jobId, setJobId] = useState('')
  const [job, setJob] = useState<JobStatus | null>(null)
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [reportSummary, setReportSummary] = useState<ReportSummary | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const isDone = useMemo(() => job?.status === 'completed' || job?.status === 'failed', [job])
  const reportArtifact = useMemo(() => artifacts.find((item) => item.name === 'report.html'), [artifacts])
  const reportJsonArtifact = useMemo(() => artifacts.find((item) => item.name === 'report.json'), [artifacts])
  const tsvArtifact = useMemo(() => artifacts.find((item) => item.name === 'tsv'), [artifacts])

  useEffect(() => {
    if (!jobId || isDone) return
    const timer = setInterval(async () => {
      const statusRes = await fetch(`${API_BASE}/jobs/${jobId}`)
      if (statusRes.ok) {
        const statusJson = (await statusRes.json()) as JobStatus
        setJob(statusJson)
      }
      const artifactRes = await fetch(`${API_BASE}/jobs/${jobId}/artifacts`)
      if (artifactRes.ok) {
        const artifactJson = (await artifactRes.json()) as { artifacts: Artifact[] }
        setArtifacts(artifactJson.artifacts)
      }
    }, 1500)
    return () => clearInterval(timer)
  }, [jobId, isDone])

  useEffect(() => {
    if (!jobId || !reportJsonArtifact || !isDone) return
    const readSummary = async () => {
      try {
        const res = await fetch(`${API_BASE}/jobs/${jobId}/download/report.json`)
        if (!res.ok) return
        const json = (await res.json()) as ReportSummary
        setReportSummary(json)
      } catch {
        // Ignore optional report preview failures in UI.
      }
    }
    void readSummary()
  }, [jobId, reportJsonArtifact, isDone])

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setError('')
    setSubmitting(true)
    setJob(null)
    setArtifacts([])
    setReportSummary(null)

    if (!file) {
      setError('Please choose a FASTA file.')
      setSubmitting(false)
      return
    }

    const form = new FormData()
    form.append('file', file)
    form.append('mode', mode)
    form.append('task', task)
    form.append('read_type', readType)
    form.append('min_length', String(minLength))
    form.append('coverage_source', 'header')
    form.append('circularity_check', 'true')
    form.append('polish', 'none')
    form.append('threshold', String(threshold))

    try {
      const res = await fetch(`${API_BASE}/jobs`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        setError(`Failed to create job (${res.status})`)
        setSubmitting(false)
        return
      }

      const payload = (await res.json()) as { job_id: string }
      setJobId(payload.job_id)

      const statusRes = await fetch(`${API_BASE}/jobs/${payload.job_id}`)
      if (statusRes.ok) {
        setJob((await statusRes.json()) as JobStatus)
      }
    } catch (err) {
      setError('Failed to reach API. Check web/api containers and proxy settings.')
    } finally {
      setSubmitting(false)
    }
  }

  const macroF1 = reportSummary?.metrics?.domain4?.macro_f1 ?? reportSummary?.metrics?.binary_domain?.macro_f1

  return (
    <main className="page">
      <header className="hero card reveal-1">
        <p className="eyebrow">Plasmid Intelligence Workbench</p>
        <h1>PlasFlow v2</h1>
        <p className="hero-copy">v1/v2 compatible plasmid/chromosome classification with confidence and uncertainty reporting.</p>
        <div className="hero-meta">
          <span className="chip">API: {API_BASE}</span>
          <span className="chip">Default threshold: 0.70</span>
        </div>
      </header>

      <section className="layout">
        <div className="stack">
          <section className="card panel reveal-2">
            <div className="panel-head">
              <h2>Run Classification</h2>
              <span className="panel-tag">Input → Predict → Artifacts</span>
            </div>

            <form onSubmit={onSubmit} className="form-grid">
              <label className="field file-field">
                <span>FASTA File</span>
                <input
                  type="file"
                  accept=".fasta,.fa,.fna,.gz"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <small>{file ? file.name : 'No file selected'}</small>
              </label>

              <label className="field">
                <span>Mode</span>
                <select value={mode} onChange={(e) => setMode(e.target.value as 'v1' | 'v2')}>
                  <option value="v1">v1</option>
                  <option value="v2">v2</option>
                </select>
              </label>

              <label className="field">
                <span>Task</span>
                <select value={task} onChange={(e) => setTask(e.target.value as 'legacy28' | 'binary_domain' | 'domain4')}>
                  <option value="legacy28">legacy28</option>
                  <option value="binary_domain">binary_domain</option>
                  <option value="domain4">domain4</option>
                </select>
              </label>

              <label className="field">
                <span>Read Type</span>
                <select value={readType} onChange={(e) => setReadType(e.target.value as 'short' | 'long' | 'hybrid' | 'complete')}>
                  <option value="short">short</option>
                  <option value="long">long</option>
                  <option value="hybrid">hybrid</option>
                  <option value="complete">complete</option>
                </select>
              </label>

              <label className="field">
                <span>Min Length</span>
                <input
                  type="number"
                  step="1"
                  min="1"
                  value={minLength}
                  onChange={(e) => setMinLength(Number(e.target.value))}
                />
              </label>

              <label className="field">
                <span>Threshold</span>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
              </label>
              <div className="form-actions">
                <button type="submit" disabled={submitting} className="primary-btn">
                  {submitting ? 'Submitting...' : 'Start Job'}
                </button>
              </div>
            </form>

            <p className="hint">
              `v1` runs the original engine. If dependencies are missing, fallback to `v2` is applied unless disabled via CLI.
            </p>
            {error && <p className="error">{error}</p>}
          </section>

          <section className="card panel reveal-3" aria-live="polite">
            <div className="panel-head">
              <h2>Job Status</h2>
              <span className={`status ${statusTone(job?.status)}`}>{job?.status ?? 'idle'}</span>
            </div>

            {!job && <p className="muted">No active job.</p>}
            {job && (
              <div className="job-grid">
                <div>
                  <p><strong>Job ID</strong> {job.job_id}</p>
                  <p><strong>Requested</strong> {job.requested_mode}</p>
                  <p><strong>Used</strong> {job.used_mode || '-'}</p>
                  <p><strong>Task</strong> {job.task || task}</p>
                  {job.fallback_reason && <p><strong>Fallback</strong> {job.fallback_reason}</p>}
                  {job.error && <p className="error"><strong>Error</strong> {job.error}</p>}
                </div>
                <div>
                  <p><strong>Created</strong> {job.created_at}</p>
                  <p><strong>Started</strong> {job.started_at || '-'}</p>
                  <p><strong>Finished</strong> {job.finished_at || '-'}</p>
                  <div className="progress-wrap">
                    <div className="progress-bar" style={{ width: `${Math.max(4, (job.progress || 0) * 100)}%` }} />
                  </div>
                </div>
              </div>
            )}
            {reportSummary?.metrics && (
              <div className="metric-row">
                <div className="metric-pill">
                  <span>Macro-F1</span>
                  <strong>{formatMetric(macroF1)}</strong>
                </div>
                <div className="metric-pill">
                  <span>ECE</span>
                  <strong>{formatMetric(reportSummary.metrics.calibration?.ece)}</strong>
                </div>
                <div className="metric-pill">
                  <span>Rec. threshold</span>
                  <strong>{formatMetric(reportSummary.metrics.calibration?.recommended_threshold, 3)}</strong>
                </div>
                <div className="metric-pill">
                  <span>Mean uncertainty</span>
                  <strong>{formatMetric(reportSummary.uncertainty_summary?.mean_uncertainty_score)}</strong>
                </div>
              </div>
            )}
          </section>
        </div>

        <div className="stack">
          <section className="card panel reveal-4">
            <div className="panel-head">
              <h2>Artifacts</h2>
              <span className="panel-tag">Downloads</span>
            </div>

            {artifacts.length === 0 && <p className="muted">No artifacts yet.</p>}
            {artifacts.length > 0 && (
              <ul className="artifact-list">
                {artifacts.map((artifact) => (
                  <li key={artifact.name}>
                    <div>
                      <p className="artifact-title">{artifact.name}</p>
                      <small>{formatBytes(artifact.size_bytes)}</small>
                    </div>
                    <a
                      className="ghost-btn"
                      href={`${API_BASE}/jobs/${jobId}/download/${encodeURIComponent(artifact.name)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Download
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="card panel reveal-5">
            <div className="panel-head">
              <h2>Quick Actions</h2>
              <span className="panel-tag">Review</span>
            </div>

            <div className="quick-actions">
              <a
                className={`quick-btn ${reportArtifact ? '' : 'disabled'}`}
                href={reportArtifact ? `${API_BASE}/jobs/${jobId}/download/report.html` : '#'}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => !reportArtifact && e.preventDefault()}
              >
                Open HTML Report
              </a>
              <a
                className={`quick-btn ${tsvArtifact ? '' : 'disabled'}`}
                href={tsvArtifact ? `${API_BASE}/jobs/${jobId}/download/tsv` : '#'}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => !tsvArtifact && e.preventDefault()}
              >
                Open TSV Output
              </a>
            </div>
          </section>
        </div>
      </section>
    </main>
  )
}
