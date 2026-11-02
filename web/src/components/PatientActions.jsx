import { useState } from 'react'
import { VITAL_META } from '../lib/risk.js'
import { useT } from '../lib/i18n.jsx'

/**
 * The three things a nurse can do to one patient: chart vitals, override
 * priority, acknowledge a flag.
 *
 * Nothing here decides anything. Each action posts to the engine and the
 * engine re-scores through its own loop, so what comes back is the same board
 * a replay tick would have produced. There is no local optimistic update,
 * because a second copy of the truth is exactly what "no demo mode" rules out.
 */

function Field({ k, meta, value, onChange, placeholder }) {
  const { t } = useT()
  return (
    <label className="vf-field">
      <span className="vf-label">
        {t(`vital.${k}`, null, meta.label)} <span className="vf-unit">{meta.unit}</span>
      </span>
      <input
        className="vf-input mono"
        type="number"
        step={meta.decimals ? 0.1 : 1}
        inputMode="decimal"
        value={value}
        placeholder={placeholder ?? '—'}
        onChange={(e) => onChange(k, e.target.value)}
      />
    </label>
  )
}

function VitalsEntry({ detail, actions, onError }) {
  const { t } = useT()
  const [values, setValues] = useState({})
  const [busy, setBusy] = useState(false)
  const set = (k, v) => setValues((p) => ({ ...p, [k]: v }))

  const filled = Object.entries(values).filter(([, v]) => v !== '' && v != null)

  const submit = async (e) => {
    e.preventDefault()
    if (!filled.length) return
    setBusy(true)
    onError(null)
    try {
      await actions.enterVitals(
        detail.subject_id,
        Object.fromEntries(filled.map(([k, v]) => [k, Number(v)])),
      )
      setValues({})
    } catch (err) {
      onError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="vitals-form" onSubmit={submit}>
      <div className="vf-grid">
        {Object.entries(VITAL_META).map(([k, meta]) => (
          <Field
            key={k}
            k={k}
            meta={meta}
            value={values[k] ?? ''}
            onChange={set}
            placeholder={
              detail.vitals?.[k] == null ? '—' : detail.vitals[k].toFixed(meta.decimals)
            }
          />
        ))}
      </div>
      <div className="vf-actions">
        <button className="btn btn-primary" disabled={!filled.length || busy}>
          {busy
            ? t('act.charting')
            : filled.length === 0
              ? t('act.chart0')
              : filled.length === 1
                ? t('act.chart1')
                : t('act.chart', { n: filled.length })}
        </button>
        <span className="section-note vf-note">{t('act.vitalsNote')}</span>
      </div>
    </form>
  )
}

function OverrideControl({ detail, actions, onError }) {
  const { t } = useT()
  const [open, setOpen] = useState(false)
  const [direction, setDirection] = useState('up')
  const [reason, setReason] = useState('')
  const [busy, setBusy] = useState(false)
  const ov = detail.override

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    onError(null)
    try {
      await actions.override(detail.subject_id, { direction, reason })
      setReason('')
      setOpen(false)
    } catch (err) {
      onError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const clear = async () => {
    setBusy(true)
    onError(null)
    try {
      await actions.clearOverride(detail.subject_id)
    } catch (err) {
      onError(err.message)
    } finally {
      setBusy(false)
    }
  }

  if (ov) {
    return (
      <div className="override-card">
        <div className="override-head">
          <span className="chip chip-override">
            ✋{' '}
            {ov.direction === 'up'
              ? t('act.priorityRaised', { n: ov.target_esi })
              : t('act.priorityLowered', { n: ov.target_esi })}
          </span>
          <button className="btn" onClick={clear} disabled={busy}>
            {t('act.withdraw')}
          </button>
        </div>
        <div className="override-meta">
          <div>
            <b>{ov.by}</b> ·{' '}
            {t('act.overrideMeta', {
              prior: ov.prior_esi,
              risk: ov.risk_at_override == null ? '—' : ov.risk_at_override.toFixed(2),
            })}
          </div>
          <div className="override-reason">“{ov.reason}”</div>
          <div className="section-note" style={{ margin: 0 }}>
            {new Date(ov.at_wall_time).toLocaleString()} · {t('act.overrideFoot')}
          </div>
        </div>
      </div>
    )
  }

  if (!open) {
    return (
      <button className="btn" onClick={() => setOpen(true)}>
        ✋ {t('act.overrideOpen')}
      </button>
    )
  }

  return (
    <form className="override-form" onSubmit={submit}>
      <div className="seg">
        <button
          type="button"
          className={`btn${direction === 'up' ? ' active' : ''}`}
          onClick={() => setDirection('up')}
          aria-pressed={direction === 'up'}
        >
          ▲ {t('act.raise')}
        </button>
        <button
          type="button"
          className={`btn${direction === 'down' ? ' active' : ''}`}
          onClick={() => setDirection('down')}
          aria-pressed={direction === 'down'}
        >
          ▼ {t('act.lower')}
        </button>
      </div>
      <input
        className="override-reason-input"
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder={t('act.reasonPlaceholder')}
        maxLength={500}
        autoFocus
      />
      <div className="vf-actions">
        <button className="btn btn-primary" disabled={reason.trim().length < 3 || busy}>
          {busy
            ? t('act.saving')
            : direction === 'up'
              ? t('act.confirmRaise')
              : t('act.confirmLower')}
        </button>
        <button type="button" className="btn" onClick={() => setOpen(false)}>
          {t('act.cancel')}
        </button>
      </div>
    </form>
  )
}

function AckControl({ detail, actions, onError }) {
  const { t } = useT()
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  const [open, setOpen] = useState(false)

  if (!detail.retriage_flag) return null
  const ack = detail.ack

  if (ack) {
    return (
      <div className="ack-card">
        <div>
          <b>{t('act.acknowledged')}</b>{' '}
          {t('act.ackBy', { by: ack.by, risk: ack.risk_at_ack.toFixed(2) })}
          {ack.note && <div className="override-reason">“{ack.note}”</div>}
        </div>
        <div className="section-note" style={{ margin: 0 }}>
          {t('act.ackNote')}{' '}
          <b className="mono">{ack.reescalate_at_risk.toFixed(2)}</b>.
        </div>
      </div>
    )
  }

  const submit = async () => {
    setBusy(true)
    onError(null)
    try {
      await actions.acknowledge(detail.subject_id, { note: note.trim() || null })
      setNote('')
      setOpen(false)
    } catch (err) {
      onError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="ack-row">
      {open ? (
        <>
          <input
            className="override-reason-input"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder={t('act.notePlaceholder')}
            maxLength={500}
            autoFocus
          />
          <button className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy ? t('act.saving') : t('act.acknowledge')}
          </button>
          <button className="btn" onClick={() => setOpen(false)}>{t('act.cancel')}</button>
        </>
      ) : (
        <button className="btn" onClick={() => setOpen(true)}>
          ✓ {t('act.ackFlag')}
        </button>
      )}
    </div>
  )
}

export default function PatientActions({ detail, actions }) {
  const { t } = useT()
  const [error, setError] = useState(null)

  return (
    <div className="actions">
      <div className="panel-title" style={{ marginBottom: 8 }}>{t('act.title')}</div>

      {error && <div className="ctl-error action-error">{error}</div>}

      <AckControl detail={detail} actions={actions} onError={setError} />
      <OverrideControl detail={detail} actions={actions} onError={setError} />

      <div className="panel-title" style={{ margin: '16px 0 6px' }}>
        {t('act.chartVitals')}
      </div>
      <VitalsEntry detail={detail} actions={actions} onError={setError} />
    </div>
  )
}
