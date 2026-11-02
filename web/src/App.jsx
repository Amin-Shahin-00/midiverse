import { useState } from 'react'
import {
  useBoard,
  useControls,
  usePatientActions,
  usePatientDetail,
  useScenarios,
} from './lib/useBoard.js'
import PatientBoard from './components/PatientBoard.jsx'
import BedBoard from './components/BedBoard.jsx'
import DetailPanel from './components/DetailPanel.jsx'
import Controls from './components/Controls.jsx'
import ScenarioBar from './components/ScenarioBar.jsx'
import { clockDate, clockTime } from './lib/risk.js'
import { useT } from './lib/i18n.jsx'

export default function App() {
  const { t, lang } = useT()
  const { board, status } = useBoard()
  const controls = useControls()
  const actions = usePatientActions()
  const scenarios = useScenarios()
  // Deep-link: ?patient=10000179 opens straight onto that patient. Handy when
  // someone wants to point at one case without hunting the board for it, and
  // it keeps the URL meaningful during a demo.
  const [selected, setSelected] = useState(() => {
    const q = new URLSearchParams(window.location.search).get('patient')
    return q ? Number(q) : null
  })

  // Detail refetches on each tick so the panel tracks the live board.
  const { data: detail, error } = usePatientDetail(selected, board?.tick, lang)

  const patients = board?.patients ?? []
  const summary = board?.summary
  const threshold = board?.alert_threshold ?? 0.4
  // Escalating, not merely raised: an acknowledged flag is still a flag, but
  // it is no longer something the header should be shouting about.
  const escalating = patients.filter((p) => p.escalating).length
  const acked = patients.filter((p) => p.ack).length

  return (
    <div className="app">
      <div>
        <header className="header">
          <div className="brand">
            {t('app.brand')} <span>{t('app.sub')}</span>
          </div>

          <div className="stat-inline">
            <span className="v mono">{clockTime(board?.clock?.sim_time)}</span>
            <span className="k">{clockDate(board?.clock?.sim_time)} {t('app.sim')}</span>
          </div>
          <div className="stat-inline">
            <span className="v mono">{summary?.census ?? '—'}</span>
            <span className="k">{t('app.inDept')}</span>
          </div>
          <div className="stat-inline">
            <span
              className="v mono"
              style={{ color: escalating ? 'var(--status-critical)' : undefined }}
            >
              {escalating}
            </span>
            <span className="k">
              {t('app.retriage')}
              {acked ? ` · ${t('app.acked', { n: acked })}` : ''}
            </span>
          </div>
          <div className="stat-inline">
            <span className="v mono">{summary?.alerts ?? '—'}</span>
            <span className="k">{t('app.alerting')}</span>
          </div>

          <div className={`conn conn-${status}`}>
            <span className="conn-dot" aria-hidden="true" />
            {t(`conn.${status}`, null, status)}
          </div>
        </header>

        <Controls
          clock={board?.clock}
          totalMinutes={board?.total_minutes}
          controls={controls}
        />

        <ScenarioBar
          scenarios={scenarios}
          actions={actions}
          onInjected={(id) => setSelected(id)}
        />
      </div>

      <div className="main">
        <div className="col">
          <PatientBoard
            patients={patients}
            loaded={board != null}
            threshold={threshold}
            selected={selected}
            onSelect={(id) => setSelected(id === selected ? null : id)}
          />
        </div>

        <div className="col col-scroll">
          {selected == null ? (
            <>
              <BedBoard summary={summary} patients={patients} />
              <div className="panel">
                <div className="panel-head">
                  <span className="panel-title">{t('app.patientDetail')}</span>
                </div>
                <div className="empty">{t('app.selectPatient')}</div>
              </div>
            </>
          ) : (
            <DetailPanel
              data={detail}
              error={error}
              threshold={threshold}
              actions={actions}
              onClose={() => setSelected(null)}
            />
          )}
        </div>
      </div>
    </div>
  )
}
