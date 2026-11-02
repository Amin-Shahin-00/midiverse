/**
 * Why this patient has this score.
 *
 * Contributions are exact TreeSHAP values from the booster, not a surrogate
 * fit — they sum to the log-odds of the score shown on the board. Warm bars
 * push risk up, cool bars pull it down, and the centre rule is "no effect":
 * a diverging encoding with a neutral midpoint, which is what signed
 * contributions call for.
 */
import { useT } from '../lib/i18n.jsx'

function Contributions({ rows }) {
  const { t } = useT()
  if (!rows?.length) return <div className="chart-empty">{t('reasons.empty')}</div>
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution))) || 1

  return (
    <div className="contrib">
      {rows.map((r) => {
        const frac = Math.abs(r.contribution) / max
        const up = r.contribution > 0
        return (
          <div className="contrib-row" key={r.feature}>
            <div className="contrib-label">
              {r.label} {r.value != null && <b>= {r.value}</b>}
            </div>
            <div className="contrib-bar" title={`${r.direction} risk by ${r.contribution}`}>
              <span className="contrib-mid" style={{ left: '50%' }} />
              <i
                className={up ? 'contrib-up' : 'contrib-down'}
                style={{
                  left: up ? '50%' : `${50 - frac * 50}%`,
                  width: `${frac * 50}%`,
                }}
              />
            </div>
          </div>
        )
      })}
      <div className="section-note" style={{ marginTop: 6 }}>
        <span style={{ color: 'var(--status-serious)' }}>■</span> {t('reasons.raises')} ·{' '}
        <span style={{ color: 'var(--series-1)' }}>■</span> {t('reasons.lowers')}
      </div>
    </div>
  )
}

/**
 * The audit layer under the plain-language reasons at the top of the panel.
 *
 * Same TreeSHAP pass, one step less aggregated: individual columns rather than
 * clinical channels. Nobody runs a shift off this, but it is what gets asked
 * for the moment someone doubts a sentence up top, so it stays one scroll away
 * rather than behind a flag.
 */
export default function ExplainPanel({ explain }) {
  const { t } = useT()
  if (!explain) return null

  return (
    <>
      <div className="panel-title" style={{ marginBottom: 8 }}>
        {t('detail.featureDetail')}
      </div>
      <p className="section-note">{t('detail.featureNote')}</p>
      <Contributions rows={explain.deterioration?.contributions} />

      {explain.admission && (
        <>
          <div className="panel-title" style={{ margin: '18px 0 8px' }}>
            {t('detail.whyBed')} — {t('detail.featureDetail')}
          </div>
          <Contributions rows={explain.admission.contributions} />
        </>
      )}
    </>
  )
}
