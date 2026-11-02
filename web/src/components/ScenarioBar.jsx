import { useState } from 'react'
import { useT } from '../lib/i18n.jsx'

/**
 * One-click scenario injection, for driving a demo.
 *
 * No form and no confirmation: during a demo the gap between deciding to show
 * something and it being on screen should be one click. Each button carries
 * what to expect underneath it, so the person driving does not have to
 * remember which preset is which while talking.
 *
 * The injected patient is scored by the same model as everyone else from the
 * moment it lands — these buttons add a patient, they do not add a mode.
 */
export default function ScenarioBar({ scenarios, actions, onInjected }) {
  const { t } = useT()
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)

  if (!scenarios?.length) return null

  const inject = async (key) => {
    setBusy(key)
    setError(null)
    try {
      const board = await actions.injectScenario(key)
      // Jump straight to the new patient: the point of injecting one is to
      // look at it.
      if (board?.subject_id != null) onInjected?.(board.subject_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="scenarios">
      <span className="ctl-label">{t('ctl.inject')}</span>
      {scenarios.map((s) => (
        <button
          key={s.key}
          className="btn scenario-btn"
          onClick={() => inject(s.key)}
          disabled={busy !== null}
          title={t(`scenario.${s.key}.blurb`, null, s.blurb)}
        >
          <span className="scenario-name">
            {busy === s.key ? t('ctl.adding') : t(`scenario.${s.key}.label`, null, s.label)}
          </span>
          <span className="scenario-blurb">
            {t(`scenario.${s.key}.blurb`, null, s.blurb)}
          </span>
        </button>
      ))}
      {error && <span className="ctl-error">{error}</span>}
    </div>
  )
}
