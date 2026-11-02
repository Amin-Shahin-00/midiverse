import { pct } from '../lib/risk.js'
import { useT } from '../lib/i18n.jsx'

/**
 * Bed demand. These are stat tiles rather than charts on purpose: each figure
 * is a single number, and a one-bar bar chart is a worse way to show one
 * number than the number itself.
 */
export default function BedBoard({ summary, patients }) {
  const { t } = useT()
  if (!summary) return null

  const occ = summary.occupancy_pct ?? 0
  const over = occ > 100
  const likely = patients.filter((p) => p.admit_flag).sort((a, b) => b.admit_risk - a.admit_risk)

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">{t('bed.title')}</span>
        <span className="panel-title">{t('bed.capacity', { n: summary.capacity })}</span>
      </div>

      <div className="panel-body">
        <div className="tiles">
          <div className="tile">
            <div className="v mono">{summary.census}</div>
            <div className="k">{t('bed.inDept')}</div>
            <div className="occ-bar">
              <div
                className={`occ-fill${over ? ' over' : ''}`}
                style={{ width: `${Math.min(100, occ)}%` }}
              />
            </div>
            <div className="note">
              {t('bed.occupancy', { p: occ.toFixed(0), n: summary.capacity })}
              {over ? t('bed.over') : ''}
            </div>
          </div>

          <div className="tile">
            <div className="v mono">{summary.expected_admissions}</div>
            <div className="k">{t('bed.expected')}</div>
            <div className="note">{t('bed.expectedNote')}</div>
          </div>

          <div className="tile">
            <div className="v mono">{summary.predicted_admits}</div>
            <div className="k">{t('bed.aboveThreshold')}</div>
            <div className="note">{t('bed.aboveNote')}</div>
          </div>

          <div className="tile">
            <div className="v mono">{summary.by_state?.unstable_now ?? 0}</div>
            <div className="k">{t('bed.unstable')}</div>
            <div className="note">{t('bed.unstableNote')}</div>
          </div>
        </div>

        <div className="panel-title" style={{ margin: '18px 0 6px' }}>
          {t('bed.mostLikely')}
        </div>
        {likely.length === 0 && <div className="chart-empty">{t('bed.nobody')}</div>}
        {likely.slice(0, 6).map((p) => (
          <div className="legend-row" key={p.subject_id}>
            <span className="mono">#{p.subject_id} · ESI {p.esi}</span>
            <span className="n mono">{pct(p.admit_risk)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
