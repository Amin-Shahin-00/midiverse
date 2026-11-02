import { useEffect, useState } from 'react'
import { duration } from '../lib/risk.js'
import { useT } from '../lib/i18n.jsx'
import LanguagePicker from './LanguagePicker.jsx'

const SPEEDS = [1, 15, 60, 240, 900]

export default function Controls({ clock, totalMinutes, controls }) {
  const { t } = useT()
  const [scrub, setScrub] = useState(null) // non-null only while dragging

  // While the user is not dragging, the slider follows the clock.
  useEffect(() => {
    if (scrub === null) return undefined
    const t = setTimeout(() => setScrub(null), 4000)
    return () => clearTimeout(t)
  }, [scrub])

  if (!clock) return null
  const elapsed = clock.elapsed_minutes ?? 0
  const value = scrub ?? elapsed

  return (
    <div className="controls">
      <span className="ctl-label">{t('ctl.replay')}</span>

      <button
        className="btn"
        onClick={() => controls.setPaused(!clock.paused)}
        aria-pressed={clock.paused}
      >
        {clock.paused ? `▶ ${t('ctl.resume')}` : `⏸ ${t('ctl.pause')}`}
      </button>

      <button className="btn" onClick={() => controls.reset()}>⏮ {t('ctl.reset')}</button>

      <span className="ctl-label">{t('ctl.speed')}</span>
      {SPEEDS.map((s) => (
        <button
          key={s}
          className={`btn${clock.speed === s ? ' active' : ''}`}
          onClick={() => controls.setSpeed(s)}
          aria-pressed={clock.speed === s}
        >
          {s}×
        </button>
      ))}

      <span className="scrub">
        <span className="ctl-label">{t('ctl.scrub')}</span>
        <input
          type="range"
          min={0}
          max={Math.max(1, Math.round(totalMinutes || 1))}
          step={5}
          value={Math.round(value)}
          onChange={(e) => setScrub(Number(e.target.value))}
          onMouseUp={(e) => controls.seek(Number(e.target.value)).then(() => setScrub(null))}
          onTouchEnd={(e) => controls.seek(Number(e.target.value)).then(() => setScrub(null))}
          aria-label={t('ctl.scrubAria')}
        />
        <span className="mono" style={{ minWidth: 92 }}>{duration(value, { h: t('unit.h'), m: t('unit.m') })}</span>
      <LanguagePicker />
      </span>
    </div>
  )
}
