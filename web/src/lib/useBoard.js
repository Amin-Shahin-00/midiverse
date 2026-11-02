import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Subscribes to the board WebSocket and keeps the latest payload.
 *
 * The board is push-only: the server sends a full snapshot every tick, so
 * there is no client-side merge and no risk of the UI drifting from the
 * engine. A dropped socket reconnects with backoff, and the first frame after
 * a reconnect is a complete board, so recovery needs no replay.
 */
export function useBoard() {
  const [board, setBoard] = useState(null)
  const [status, setStatus] = useState('connecting')
  const socketRef = useRef(null)
  const retryRef = useRef(0)
  const timerRef = useRef(null)

  // Seed from the REST snapshot so the board has content before the socket
  // delivers its first frame. Without this the page spends its first moment
  // asserting "No patients in the department", which is not the same thing as
  // "not loaded yet" and is the wrong thing to tell someone looking at a
  // triage board. The socket overwrites it as soon as it arrives, and a
  // failure here is silent because the socket is the real source.
  useEffect(() => {
    let cancelled = false
    fetch('/api/board')
      .then((r) => (r.ok ? r.json() : null))
      .then((snapshot) => {
        if (!cancelled && snapshot) setBoard((current) => current ?? snapshot)
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false

    const connect = () => {
      if (cancelled) return
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${window.location.host}/ws/board`)
      socketRef.current = ws

      ws.onopen = () => {
        if (cancelled) return
        retryRef.current = 0
        setStatus('live')
        // The server treats inbound frames as keepalive only.
        ws.send('hello')
      }
      ws.onmessage = (event) => {
        if (!cancelled) setBoard(JSON.parse(event.data))
      }
      ws.onclose = () => {
        if (cancelled) return
        setStatus('reconnecting')
        const delay = Math.min(1000 * 2 ** retryRef.current, 10000)
        retryRef.current += 1
        timerRef.current = setTimeout(connect, delay)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      cancelled = true
      clearTimeout(timerRef.current)
      socketRef.current?.close()
    }
  }, [])

  return { board, status }
}

async function request(path, method, body) {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    // The engine rejects an impossible action with 409 and a sentence saying
    // why ("flag is already acknowledged"). Surface that sentence rather than
    // a status code -- it is written for the person who clicked.
    let detail = `${res.status}`
    try {
      const j = await res.json()
      if (j?.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
    } catch {
      /* non-JSON error body; the status will have to do */
    }
    throw new Error(detail)
  }
  return res.json()
}

const post = (path, body) => request(path, 'POST', body)
const del = (path) => request(path, 'DELETE')

export function useControls() {
  const setSpeed = useCallback((speed) => post('/api/clock', { speed }), [])
  const setPaused = useCallback((paused) => post('/api/clock', { paused }), [])
  const seek = useCallback((elapsed_minutes) => post('/api/clock/seek', { elapsed_minutes }), [])
  const reset = useCallback(() => post('/api/clock/seek', { reset: true }), [])
  return { setSpeed, setPaused, seek, reset }
}

/**
 * Manual control: charting, scenario injection, override, acknowledgement.
 *
 * None of these update local state. Every one returns the board the server
 * produced by re-running its own scoring pass, and the server broadcasts that
 * same payload to every connected client — so the change arrives through the
 * WebSocket exactly like a replay tick. Optimistically patching state here
 * would create a second source of truth and let this screen drift from the
 * engine, which is the thing the whole design is avoiding.
 */
export function usePatientActions() {
  return {
    enterVitals: useCallback(
      (subjectId, vitals) => post(`/api/patients/${subjectId}/vitals`, vitals), []),
    injectScenario: useCallback(
      (key, by) => post(`/api/scenarios/${key}`, { by }), []),
    override: useCallback(
      (subjectId, body) => post(`/api/patients/${subjectId}/override`, body), []),
    clearOverride: useCallback(
      (subjectId) => del(`/api/patients/${subjectId}/override`), []),
    acknowledge: useCallback(
      (subjectId, body) => post(`/api/patients/${subjectId}/acknowledge`, body), []),
  }
}

/** The scenario catalogue. Static for the life of the server, so fetched once. */
export function useScenarios() {
  const [scenarios, setScenarios] = useState([])
  useEffect(() => {
    let cancelled = false
    fetch('/api/scenarios')
      .then((r) => r.json())
      .then((j) => !cancelled && setScenarios(j.scenarios || []))
      .catch(() => {})
    return () => { cancelled = true }
  }, [])
  return scenarios
}

/**
 * Detail, vitals history and explanation for one patient, in `lang`.
 *
 * The language is part of the request, not a post-processing step: the reason
 * sentences are generated from the model's own structure server-side, so the
 * two languages cannot drift into describing different patients. Changing
 * language re-fetches, which is why `lang` is in the dependency list.
 */
export function usePatientDetail(subjectId, tick, lang = 'en') {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (subjectId == null) {
      setData(null)
      return
    }
    let cancelled = false
    const q = `?lang=${encodeURIComponent(lang)}`
    Promise.all([
      fetch(`/api/patients/${subjectId}${q}`).then((r) => r.json()),
      fetch(`/api/patients/${subjectId}/vitals`).then((r) => r.json()),
      fetch(`/api/patients/${subjectId}/explain${q}`).then((r) => (r.ok ? r.json() : null)),
    ])
      .then(([detail, vitals, explain]) => {
        if (!cancelled) {
          setData({ detail, vitals, explain })
          setError(null)
        }
      })
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
    // Refetch as the clock advances so the panel tracks the live board, and
    // when the language changes so the reasons come back in it.
  }, [subjectId, tick, lang])

  return { data, error }
}
