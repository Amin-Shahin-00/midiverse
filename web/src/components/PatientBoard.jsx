import { duration, pct, riskTier, STATE_LABELS, trendBand } from '../lib/risk.js'
import { useT } from '../lib/i18n.jsx'

function PatientRow({ p, threshold, selected, onSelect }) {
  const { t } = useT()
  const tier = riskTier(p.risk, threshold)
  const trend = trendBand(p.risk_delta)
  const state = STATE_LABELS[p.clinical_state] || STATE_LABELS.stable

  // The rail is the across-the-room signal: solid for a re-triage flag,
  // dimmed for plain risk tiers, nothing for a stable patient. An
  // acknowledged flag keeps the rail — someone still needs to see it from the
  // door — but drops the pulse, which is the part that demands attention.
  const rail = p.retriage_flag ? 'rail-retriage' : `rail-${tier.id}`

  return (
    <button
      className={`prow${selected ? ' selected' : ''}`}
      onClick={() => onSelect(p.subject_id)}
      aria-label={`Patient ${p.subject_id}, ESI ${p.esi}, risk ${t('tier.' + tier.id)}, ${t('trend.' + trend.id)}${
        p.retriage_flag
          ? p.escalating
            ? ', re-triage flag raised'
            : `, re-triage flag acknowledged by ${p.ack?.by ?? 'staff'}`
          : ''
      }${p.override ? `, priority overridden to ESI ${p.override.target_esi} by ${p.override.by}` : ''}`}
    >
      <span className={`rail ${rail}`} aria-hidden="true" />

      <span>
        <div className="pid mono">#{p.subject_id}</div>
        <div className="psub">{p.age}y {p.sex}</div>
      </span>

      <span className={`esi${p.retriage_flag ? ' esi-mismatch' : ''}${
        p.override ? ' esi-override' : ''
      }`}>
        <span className="n mono">{p.override ? p.override.target_esi : p.esi}</span>
        <span className="l">{p.override ? t('row.was', { n: p.esi }) : t('row.esi')}</span>
      </span>

      <span className="risk">
        <span className="risk-glyph" style={{ color: tier.color }} aria-hidden="true">{tier.glyph}</span>
        <span>
          <div className="risk-num mono" style={{ color: tier.color }}>
            {p.risk == null ? '—' : p.risk.toFixed(2)}
          </div>
          <div className="risk-word">{t('tier.' + tier.id)}</div>
        </span>
      </span>

      <span className={`trend ${trend.cls}`}>
        <span className="trend-glyph" aria-hidden="true">{trend.glyph}</span>
        <span>
          <div className="mono">
            {p.risk_delta == null ? '—' : `${p.risk_delta > 0 ? '+' : ''}${p.risk_delta.toFixed(2)}`}
          </div>
          <div className="trend-word">{t('trend.' + trend.id)}</div>
        </span>
      </span>

      <span className="wait mono">{duration(p.minutes_in_dept, { h: t('unit.h'), m: t('unit.m') })}</span>

      <span>
        <div className="bed mono">{pct(p.admit_risk)}</div>
        <div className="bed-sub">{p.admit_flag ? t('row.likelyBed') : t('row.bedUnlikely')}</div>
      </span>

      <span className="chips">
        {p.retriage_flag && (
          <span className={`chip flag-retriage${p.escalating ? '' : ' flag-acked'}`}>
            ⚑ {t('row.retriageTo', { n: p.implied_esi })}{!p.escalating && ` · ${t('row.ack')}`}
          </span>
        )}
        {p.override && (
          <span className="chip chip-override" title={`${p.override.by}: ${p.override.reason}`}>
            ✋ {p.override.direction === 'up' ? t('row.raised') : t('row.lowered')} · {p.override.by}
          </span>
        )}
        {p.injected && (
          <span className="chip chip-injected" title={`Injected demo scenario: ${p.injected}`}>
            ⚙<span className="sr-only"> injected scenario</span>
          </span>
        )}
        {p.manual_entries > 0 && (
          <span
            className="chip chip-manual"
            title={`${p.manual_entries} hand-charted observation${
              p.manual_entries === 1 ? '' : 's'
            } — this patient is off their replay trajectory`}
          >
            ✎ {p.manual_entries}
          </span>
        )}
        <span className={`chip ${state.cls}`}>{state.glyph} {t('state.' + p.clinical_state)}</span>
      </span>
    </button>
  )
}

export default function PatientBoard({ patients, threshold, selected, onSelect, loaded }) {
  const { t } = useT()
  const escalating = patients.filter((p) => p.escalating).length
  const acked = patients.filter((p) => p.retriage_flag && !p.escalating).length

  return (
    <div className="panel" style={{ flex: 1 }}>
      <div className="panel-head">
        <span className="panel-title">{t('board.title')}</span>
        <span
          className="panel-title"
          style={{ color: escalating ? 'var(--status-critical)' : undefined }}
        >
          {escalating === 1 ? t('board.flag', { n: 1 }) : t('board.flags', { n: escalating })}
          {acked > 0 && (
            <span style={{ color: 'var(--text-muted)' }}>
              {' · '}{t('board.acknowledged', { n: acked })}
            </span>
          )}
        </span>
      </div>

      <div className="board-cols" aria-hidden="true">
        <span />
        <span>{t('col.patient')}</span>
        <span>{t('col.triage')}</span>
        <span>{t('col.risk')}</span>
        <span>{t('col.trend')}</span>
        <span>{t('col.waiting')}</span>
        <span>{t('col.bed')}</span>
        <span>{t('col.status')}</span>
      </div>

      <div className="board-scroll">
        {patients.length === 0 && (
          <div className="empty">
            {loaded ? t('board.empty') : t('board.connecting')}
          </div>
        )}
        {patients.map((p) => (
          <PatientRow
            key={p.subject_id}
            p={p}
            threshold={threshold}
            selected={selected === p.subject_id}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  )
}
