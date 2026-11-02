/**
 * The three things driving this patient's score, in clinical language.
 *
 * This is the first block under the header for a reason. An alert nobody can
 * explain gets ignored, and an explanation three scroll-lengths below the
 * number it explains is one nobody reads. The SHAP bar chart further down the
 * panel is the audit trail; this is the answer.
 *
 * The sentences arrive already written from `api/reasons.py` — phrasing lives
 * server-side so the API, any future export and this panel cannot drift into
 * describing the same feature three different ways. What the browser adds is
 * ranking furniture and the direction encoding.
 *
 * Direction is carried three ways, like every other signal on this board: an
 * arrow, a word, and a colour last. Each row also carries the feature name and
 * its exact log-odds contribution in a `title`, so a number on screen can be
 * traced back to the column that produced it without putting `rr_slope30` in
 * front of a nurse.
 */
import { useT } from '../lib/i18n.jsx'

export default function Reasons({ reasons, applicable = true }) {
  const { t } = useT()
  if (!reasons?.length) {
    return (
      <div className="reasons-empty">{t('reasons.empty')}</div>
    )
  }

  return (
    <div className="reasons">
      {reasons.map((r, i) => {
        const up = r.effect === 'increases'
        return (
          <div
            className="reason"
            key={r.channel}
            title={`${r.lead_feature} = ${r.lead_value ?? 'n/a'} · channel contributes ${
              r.contribution > 0 ? '+' : ''
            }${r.contribution} log-odds across ${r.n_features} feature${
              r.n_features === 1 ? '' : 's'
            }`}
          >
            <span className="reason-rank mono" aria-hidden="true">{i + 1}</span>
            <span className="reason-text">{r.text}</span>
            <span className={`reason-effect ${up ? 'reason-up' : 'reason-down'}`}>
              <span aria-hidden="true">{up ? '▲' : '▼'}</span>
              {up ? t('reasons.raises') : t('reasons.lowers')}
            </span>
          </div>
        )
      })}

      <p className="section-note reasons-note">
        {t('reasons.note')}
        {!applicable && t('reasons.ood')}
      </p>
    </div>
  )
}
