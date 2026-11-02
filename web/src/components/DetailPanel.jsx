import LineChart from './LineChart.jsx'
import ExplainPanel from './ExplainPanel.jsx'
import Reasons from './Reasons.jsx'
import PatientActions from './PatientActions.jsx'
import { VITAL_META, duration, pct, riskTier, STATE_LABELS } from '../lib/risk.js'
import { useT } from '../lib/i18n.jsx'

function Tile({ label, value, note }) {
  return (
    <div className="tile">
      <div className="v mono">{value}</div>
      <div className="k">{label}</div>
      {note && <div className="note">{note}</div>}
    </div>
  )
}

export default function DetailPanel({ data, error, threshold, actions, onClose }) {
  const { t } = useT()
  if (error) return <div className="empty">{t('detail.loadError', { e: error })}</div>
  if (!data) return <div className="empty">{t('detail.loading')}</div>

  const { detail, vitals, explain } = data
  const tier = riskTier(detail.risk, threshold)
  const state = STATE_LABELS[detail.clinical_state] || STATE_LABELS.stable
  const obs = vitals?.observations || []

  const riskSeries = (detail.risk_history || []).map((p) => ({ x: p.minutes, y: p.risk }))

  return (
    <div className="panel" style={{ flex: 1 }}>
      <div className="panel-head">
        <div>
          <div className="detail-id mono">#{detail.subject_id}</div>
          <div className="detail-sub">
            {detail.age}y {detail.sex} · ESI {detail.esi} ·{' '}
            {t('detail.inDept', {
              d: duration(detail.minutes_in_dept, { h: t('unit.h'), m: t('unit.m') }),
            })}
          </div>
        </div>
        <button className="btn" onClick={onClose}>{t('detail.close')}</button>
      </div>

      <div className="panel-body">
        <div className="detail-grid">
          <Tile
            label={t('detail.risk')}
            value={detail.risk == null ? '—' : detail.risk.toFixed(2)}
            note={`${tier.glyph} ${t('tier.' + tier.id)}`}
          />
          <Tile
            label={t('detail.news2')}
            value={detail.news2 ?? '—'}
            note={t('detail.instability', {
              v: t(`instability.${detail.instability}`, null, detail.instability),
            })}
          />
          <Tile
            label={t('detail.needsBed')}
            value={pct(detail.admit_risk)}
            note={detail.admit_flag ? t('detail.aboveThreshold') : t('detail.belowThreshold')}
          />
        </div>

        <div className="chips" style={{ marginBottom: 14 }}>
          <span className={`chip ${state.cls}`}>
            {state.glyph} {t('state.' + detail.clinical_state)}
          </span>
          {detail.retriage_flag && (
            <span className={`chip flag-retriage${detail.escalating ? '' : ' flag-acked'}`}>
              ⚑ {t('row.retriageTo', { n: detail.implied_esi })}
              {!detail.escalating && ` · ${t('row.ack')}`}
            </span>
          )}
          {detail.override && (
            <span className="chip chip-override">
              ✋{' '}
              {t('detail.overrideBy', {
                n: detail.override.target_esi,
                by: detail.override.by,
              })}
            </span>
          )}
          {detail.injected && (
            <span className="chip chip-injected">⚙ {t('detail.scenario')}</span>
          )}
          {detail.manual_entries > 0 && (
            <span className="chip chip-manual">
              ✎ {t('detail.manualCount', { n: detail.manual_entries })}
            </span>
          )}
          {!detail.model_applicable && (
            <span className="ood">{t('detail.ood')}</span>
          )}
        </div>

        {actions && (
          <div style={{ marginBottom: 18 }}>
            <PatientActions detail={detail} actions={actions} />
          </div>
        )}

        <div className="panel-title" style={{ marginBottom: 8 }}>{t('detail.whyScore')}</div>
        <Reasons reasons={detail.reasons} applicable={detail.model_applicable} />

        <div className="panel-title" style={{ marginTop: 18, marginBottom: 8 }}>
          {t('detail.trajectory')}
        </div>
        <p className="section-note">{t('detail.trajectoryNote')}</p>
        <div className="chart-card" style={{ marginBottom: 16 }}>
          <LineChart
            points={riskSeries}
            yDomain={[0, 1]}
            threshold={threshold}
            thresholdLabel="alert"
            height={148}
            formatY={(v) => v.toFixed(2)}
            formatX={(v) => duration(v, { h: t('unit.h'), m: t('unit.m') })}
          />
        </div>

        <div className="panel-title" style={{ marginBottom: 8 }}>{t('detail.vitals')}</div>
        <p className="section-note">{t('detail.vitalsNote')}</p>
        <div className="charts">
          {Object.entries(VITAL_META).map(([key, meta]) => {
            const pts = obs.map((o) => ({ x: o.t, y: o[key] }))
            const latest = [...pts].reverse().find((p) => p.y != null)
            return (
              <div className="chart-card" key={key}>
                <div className="chart-head">
                  <span className="chart-title">{t(`vital.${key}`, null, meta.label)}</span>
                  <span className="chart-now mono">
                    {latest ? latest.y.toFixed(meta.decimals) : '—'}
                    <span className="unit">{meta.unit}</span>
                  </span>
                </div>
                <LineChart
                  points={pts}
                  normalBand={meta.normal}
                  height={112}
                  showEndLabel={false}
                  unit={meta.unit}
                  formatY={(v) => v.toFixed(meta.decimals)}
                  formatX={(v) => duration(v, { h: t('unit.h'), m: t('unit.m') })}
                />
              </div>
            )
          })}
        </div>

        {detail.admit_reasons?.length > 0 && (
          <>
            <div className="panel-title" style={{ marginTop: 18, marginBottom: 8 }}>
              {t('detail.whyBed')}
            </div>
            <Reasons reasons={detail.admit_reasons} />
          </>
        )}

        <div style={{ marginTop: 18 }}>
          <ExplainPanel explain={explain} />
        </div>
      </div>
    </div>
  )
}
