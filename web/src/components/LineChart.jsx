import { useMemo, useRef, useState } from 'react'

const PAD = { top: 10, right: 46, bottom: 20, left: 38 }

/**
 * One series, one hue — so no legend is needed and colour never carries
 * identity. Grid and axes are solid hairlines; the only dashed rule is the
 * threshold, where dashing means "threshold" rather than decoration.
 */
export default function LineChart({
  points,
  width = 420,
  height = 132,
  yDomain,
  normalBand,
  threshold,
  thresholdLabel,
  formatY = (v) => String(Math.round(v)),
  formatX = (v) => `${Math.round(v)}m`,
  unit = '',
  showEndLabel = true,
}) {
  const wrapRef = useRef(null)
  const [hover, setHover] = useState(null)

  const clean = useMemo(
    () => (points || []).filter((p) => p.y != null && !Number.isNaN(p.y)),
    [points],
  )

  const scales = useMemo(() => {
    if (clean.length === 0) return null
    const xs = clean.map((p) => p.x)
    const ys = clean.map((p) => p.y)

    // An explicit domain is a statement about the scale (risk is 0..1), so it
    // is used as given. Only an inferred domain gets headroom.
    let lo, hi
    if (yDomain) {
      ;[lo, hi] = yDomain
    } else {
      lo = Math.min(...ys)
      hi = Math.max(...ys)
      if (normalBand) {
        lo = Math.min(lo, normalBand[0])
        hi = Math.max(hi, normalBand[1])
      }
      if (hi - lo < 1e-6) {
        lo -= 1
        hi += 1
      }
      const padY = (hi - lo) * 0.12
      lo -= padY
      hi += padY
    }

    const x0 = Math.min(...xs)
    const x1 = Math.max(...xs)
    const spanX = x1 - x0 || 1

    const sx = (v) => PAD.left + ((v - x0) / spanX) * (width - PAD.left - PAD.right)
    const sy = (v) => PAD.top + (1 - (v - lo) / (hi - lo)) * (height - PAD.top - PAD.bottom)
    return { sx, sy, lo, hi, x0, x1 }
  }, [clean, width, height, yDomain, normalBand])

  if (!scales) return <div className="chart-empty">No data yet</div>

  const { sx, sy, lo, hi, x0, x1 } = scales

  // Break the path wherever the source had a gap, rather than drawing a
  // straight line through missing observations.
  const segments = []
  let current = []
  for (const p of points) {
    if (p.y == null || Number.isNaN(p.y)) {
      if (current.length) segments.push(current)
      current = []
    } else {
      current.push(p)
    }
  }
  if (current.length) segments.push(current)

  const toPath = (seg) => seg.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')

  const last = clean[clean.length - 1]
  const gridYs = [lo + (hi - lo) * 0.5, hi - (hi - lo) * 0.08, lo + (hi - lo) * 0.08]

  const onMove = (event) => {
    const rect = wrapRef.current.getBoundingClientRect()
    const px = ((event.clientX - rect.left) / rect.width) * width
    let best = null
    for (const p of clean) {
      const d = Math.abs(sx(p.x) - px)
      if (!best || d < best.d) best = { d, p }
    }
    if (best) setHover(best.p)
  }

  return (
    <div
      ref={wrapRef}
      style={{ position: 'relative' }}
      onMouseMove={onMove}
      onMouseLeave={() => setHover(null)}
    >
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img">
        {normalBand && (
          <rect
            className="normal-band"
            x={PAD.left}
            y={sy(Math.min(normalBand[1], hi))}
            width={width - PAD.left - PAD.right}
            height={Math.max(0, sy(Math.max(normalBand[0], lo)) - sy(Math.min(normalBand[1], hi)))}
          />
        )}

        {gridYs.map((v, i) => (
          <line key={i} className="grid-line" x1={PAD.left} x2={width - PAD.right} y1={sy(v)} y2={sy(v)} />
        ))}

        {threshold != null && threshold >= lo && threshold <= hi && (
          <>
            <line
              className="threshold-line"
              x1={PAD.left}
              x2={width - PAD.right}
              y1={sy(threshold)}
              y2={sy(threshold)}
            />
            <text className="threshold-text" x={width - PAD.right + 4} y={sy(threshold) + 4}>
              {thresholdLabel}
            </text>
          </>
        )}

        {segments.map((seg, i) => (
          <path key={i} className="series-line" d={toPath(seg)} />
        ))}

        <circle className="series-end" cx={sx(last.x)} cy={sy(last.y)} r={4.5} />
        {showEndLabel && (
          <text className="endpoint-label" x={sx(last.x) + 9} y={sy(last.y) + 4}>
            {formatY(last.y)}
          </text>
        )}

        {/* Label the domain itself, not an inset gridline — a risk axis that
            reads 0.92/0.08 instead of 1.00/0.00 misstates the scale. */}
        <text className="axis-text" x={4} y={PAD.top + 9}>{formatY(hi)}</text>
        <text className="axis-text" x={4} y={height - PAD.bottom - 2}>{formatY(lo)}</text>
        <text className="axis-text" x={PAD.left} y={height - 4}>{formatX(x0)}</text>
        <text className="axis-text" x={width - PAD.right} y={height - 4} textAnchor="end">
          {formatX(x1)}
        </text>

        {hover && (
          <>
            <line className="crosshair" x1={sx(hover.x)} x2={sx(hover.x)} y1={PAD.top} y2={height - PAD.bottom} />
            <circle className="series-end" cx={sx(hover.x)} cy={sy(hover.y)} r={5} />
          </>
        )}
      </svg>

      {hover && (
        <div
          style={{
            position: 'absolute',
            left: `${Math.min(88, (sx(hover.x) / width) * 100)}%`,
            top: 0,
            transform: 'translateX(-50%)',
            background: 'var(--surface-1)',
            border: '1px solid var(--hairline)',
            borderRadius: 6,
            padding: '4px 9px',
            fontSize: 13,
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
          }}
        >
          <b>{formatY(hover.y)}{unit}</b>
          <span style={{ color: 'var(--text-muted)', marginLeft: 7 }}>{formatX(hover.x)}</span>
        </div>
      )}
    </div>
  )
}
