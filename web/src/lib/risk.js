// Risk tiers, trend bands, and formatting.
//
// Every tier carries a GLYPH and a WORD as well as a colour. ED lighting is
// poor and some readers are colourblind, so hue is the third signal here, never
// the first. The status palette's own validator confirms this is necessary:
// as a categorical set these four colours fail CVD separation (warning vs
// serious measure ΔE 13.6), which is exactly why they ship with icon + label.

// Each tier carries an id, not a word. The word is looked up per language
// (`tier.<id>`), so banding logic and vocabulary stay separate and a
// translation cannot accidentally change where a band starts.
export const TIERS = [
  { id: 'critical', min: 0.8, glyph: '▲', color: 'var(--status-critical)' },
  { id: 'high', min: 0.55, glyph: '◆', color: 'var(--status-serious)' },
  { id: 'elevated', min: null, glyph: '●', color: 'var(--status-warning)' },
  { id: 'low', min: -Infinity, glyph: '–', color: 'var(--text-muted)' },
]

/** `elevated` starts at the model's own operating threshold, not a round number. */
export function riskTier(risk, threshold = 0.4) {
  if (risk == null || Number.isNaN(risk)) return TIERS[3]
  if (risk >= TIERS[0].min) return TIERS[0]
  if (risk >= TIERS[1].min) return TIERS[1]
  if (risk >= threshold) return TIERS[2]
  return TIERS[3]
}

// Trend over the server's lookback. Text and arrow shape both carry it.
export function trendBand(delta) {
  if (delta == null || Number.isNaN(delta)) return { id: 'new', glyph: '·', cls: 'trend-flat' }
  if (delta >= 0.15) return { id: 'up-fast', glyph: '▲▲', cls: 'trend-up-fast' }
  if (delta >= 0.05) return { id: 'up', glyph: '▲', cls: 'trend-up' }
  if (delta <= -0.15) return { id: 'down-fast', glyph: '▼▼', cls: 'trend-down' }
  if (delta <= -0.05) return { id: 'down', glyph: '▼', cls: 'trend-down' }
  return { id: 'flat', glyph: '▬', cls: 'trend-flat' }
}

export const STATE_LABELS = {
  unstable_now: { cls: 'state-unstable', glyph: '✚' },
  early_warning: { cls: 'state-early', glyph: '◤' },
  watch: { cls: 'state-watch', glyph: '◒' },
  news2_elevated: { cls: 'state-news', glyph: '◔' },
  stable: { cls: 'state-stable', glyph: '○' },
}

export const pct = (v, digits = 0) =>
  v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`

// Units are a letter per language: "4h 20m" / "4س 20د". Digits stay Western
// in both — see the note in i18n.jsx.
export function duration(minutes, units = { h: 'h', m: 'm' }) {
  if (minutes == null || Number.isNaN(minutes)) return '—'
  const m = Math.max(0, Math.round(minutes))
  const h = Math.floor(m / 60)
  return h > 0 ? `${h}${units.h} ${String(m % 60).padStart(2, '0')}${units.m}` : `${m}${units.m}`
}

export const clockTime = (iso) =>
  !iso ? '—' : new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })

export const clockDate = (iso) =>
  !iso ? '' : new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' })

// Clinically normal ranges, drawn as a recessive band behind each vitals trace
// so "still inside normal but moving" is visible at a glance.
// `label` is the English fallback; components look up `vital.<key>` for the
// active language. Ranges, units and decimals are shared — they are
// measurements, not vocabulary.
export const VITAL_META = {
  hr: { label: 'Heart rate', unit: 'bpm', normal: [60, 100], decimals: 0 },
  rr: { label: 'Resp rate', unit: '/min', normal: [12, 20], decimals: 0 },
  spo2: { label: 'SpO₂', unit: '%', normal: [95, 100], decimals: 0 },
  sbp: { label: 'Systolic BP', unit: 'mmHg', normal: [100, 140], decimals: 0 },
  dbp: { label: 'Diastolic BP', unit: 'mmHg', normal: [60, 90], decimals: 0 },
  temp: { label: 'Temperature', unit: '°C', normal: [36.1, 38.0], decimals: 1 },
}
