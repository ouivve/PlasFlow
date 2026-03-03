# PlasFlow v2 Operations Notes

## Runtime data

- Job DB: `runs/plasflow_jobs.db`
- Job artifacts: `runs/<job_id>/`

Override with:
- `PLASFLOW_RUNS_DIR`

## Process settings

- `PLASFLOW_EXECUTOR=process|inline`
- `PLASFLOW_MAX_WORKERS=<int>`

`inline` is useful for deterministic tests.

## Backups

Recommended minimum:
1. Snapshot `runs/plasflow_jobs.db` daily.
2. Archive `runs/<job_id>/` outputs needed for audit/reproducibility.

## Logs

Current implementation relies on container/stdout logs. For production:
1. Capture API stdout/stderr in centralized logging.
2. Add reverse proxy access logs.
3. Keep error traces for failed jobs keyed by `job_id`.

## Failure handling

- `failed`: inspect `GET /api/v1/jobs/{job_id}` `error` field.
- `v1 -> v2` fallback: check `fallback_reason`.

## Cleanup policy

Suggested retention:
1. Keep metadata DB for >= 90 days.
2. Prune artifact directories older than policy threshold unless marked as protected.
