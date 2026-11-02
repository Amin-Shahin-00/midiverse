import { createContext, useContext, useEffect, useMemo, useState } from 'react'

/**
 * Interface language, and the writing direction that comes with it.
 *
 * Two rules shape this file.
 *
 * **Clinical vocabulary is not decoration.** The words on this board are how a
 * nurse decides who to see next, so the Arabic is clinical Arabic — the terms
 * used on a monitor and in a handover, not a literal rendering of the English.
 * Where an abbreviation is universal in practice (ESI, NEWS2, SpO₂ as a
 * concept) it stays recognisable rather than being invented anew.
 *
 * **Numbers stay Western.** Arabic-Indic numerals (٤٢) are typographically
 * correct, but monitors, charts and infusion pumps across the region show
 * Western digits, and a board read at three metres is the wrong place to make
 * someone translate a figure they are about to act on. Direction, layout and
 * every word flip; the digits do not.
 *
 * The reason sentences in the detail panel are NOT translated here — they are
 * generated from the model's own structure in `api/reasons.py` and requested
 * with `?lang=`, so the two languages cannot drift into describing different
 * patients. See that module's parity tests.
 */

const STRINGS = {
  en: {
    'app.brand': 'WATCHLIST',
    'app.sub': '/ ED BOARD',
    'app.sim': 'sim',
    'app.inDept': 'in dept',
    'app.retriage': 're-triage',
    'app.acked': "{n} ack'd",
    'app.alerting': 'alerting',
    'app.patientDetail': 'Patient detail',
    'app.selectPatient':
      'Select a patient to see vitals trends and why the model scored them the way it did.',
    'conn.live': 'live',
    'conn.connecting': 'connecting',
    'conn.reconnecting': 'reconnecting',

    'ctl.replay': 'Replay',
    'ctl.pause': 'Pause',
    'ctl.resume': 'Resume',
    'ctl.reset': 'Reset',
    'ctl.speed': 'Speed',
    'ctl.scrub': 'Scrub',
    'ctl.scrubAria': 'Scrub replay position',
    'ctl.inject': 'Inject',
    'ctl.adding': 'adding…',
    'ctl.language': 'Language',

    'board.title': 'Patient board · sorted by risk, not arrival',
    'board.flags': '{n} re-triage flags',
    'board.flag': '{n} re-triage flag',
    'board.acknowledged': '{n} acknowledged',
    'board.empty': 'No patients in the department.',
    'board.connecting': 'Connecting to the board…',
    'col.patient': 'Patient',
    'col.triage': 'Triage',
    'col.risk': 'Deterioration risk',
    'col.trend': 'Trend (30m)',
    'col.waiting': 'Waiting',
    'col.bed': 'Needs bed',
    'col.status': 'Status',
    'row.esi': 'ESI',
    'row.was': 'was {n}',
    'row.likelyBed': 'LIKELY BED',
    'row.bedUnlikely': 'bed unlikely',
    'row.retriageTo': 'RE-TRIAGE → ESI {n}',
    'row.ack': 'ACK',
    'row.raised': 'RAISED',
    'row.lowered': 'LOWERED',

    'tier.critical': 'CRITICAL',
    'tier.high': 'HIGH',
    'tier.elevated': 'ELEVATED',
    'tier.low': 'LOW',
    'trend.up-fast': 'rising fast',
    'trend.up': 'rising',
    'trend.down-fast': 'falling fast',
    'trend.down': 'falling',
    'trend.flat': 'steady',
    'trend.new': 'new',
    'state.unstable_now': 'UNSTABLE NOW',
    'state.early_warning': 'EARLY WARNING',
    'state.watch': 'WATCH',
    'state.news2_elevated': 'NEWS2 UP',
    'state.stable': 'STABLE',

    'vital.hr': 'Heart rate',
    'vital.rr': 'Resp rate',
    'vital.spo2': 'SpO₂',
    'vital.sbp': 'Systolic BP',
    'vital.dbp': 'Diastolic BP',
    'vital.temp': 'Temperature',

    'detail.close': 'Close',
    'detail.inDept': '{d} in dept',
    'detail.risk': 'Risk',
    'detail.news2': 'NEWS2',
    'detail.needsBed': 'Needs bed',
    'detail.instability': 'instability: {v}',
    'instability.low': 'low',
    'instability.medium': 'medium',
    'instability.high': 'high',
    'instability.unknown': 'unknown',
    'detail.overrideBy': 'ESI {n} by {by}',
    'detail.aboveThreshold': 'above threshold',
    'detail.belowThreshold': 'below threshold',
    'detail.ood': 'NEWS2 ≥ 7 — already unstable; risk score is extrapolation',
    'detail.whyScore': 'Why this score',
    'detail.whyBed': 'Why this bed call',
    'detail.trajectory': 'Risk trajectory',
    'detail.trajectoryNote':
      'Score history since arrival. The dashed rule is the alert threshold chosen at training for ~80% sensitivity. History restarts after a scrub.',
    'detail.vitals': 'Vitals',
    'detail.vitalsNote':
      'Shaded band is the normal range. Trajectory matters more than the snapshot — a value inside the band that is moving is the case this board exists to catch.',
    'detail.featureDetail': 'Feature detail',
    'detail.featureNote':
      'The individual features behind the reasons at the top of this panel. Exact per-feature contributions (TreeSHAP) — they sum to the log-odds of the score, so this is the model explaining itself, not a second model guessing at it.',
    'detail.scenario': 'SCENARIO',
    'detail.manualCount': '{n} manual',
    'detail.loading': 'Loading…',
    'detail.loadError': 'Could not load patient: {e}',

    'reasons.raises': 'raises risk',
    'reasons.lowers': 'lowers risk',
    'reasons.empty': 'Not enough charting yet to say what is driving this score.',
    'reasons.note':
      "Ranked by each channel's total TreeSHAP contribution, so three reasons are three different things about the patient — not one trend restated three ways. Hover a row for the underlying feature.",
    'reasons.ood': ' This patient is already unstable, so the score itself is extrapolation.',

    'act.title': 'Manual control',
    'act.chartVitals': 'Chart vitals',
    'act.chart': 'Chart {n} values',
    'act.chart0': 'Chart values',
    'act.chart1': 'Chart 1 value',
    'act.charting': 'Charting…',
    'act.vitalsNote':
      'Placeholders are the current reading. Anything left blank carries forward. Charting takes this patient off their replay trajectory.',
    'act.overrideOpen': 'Override priority…',
    'act.raise': 'Raise (sicker)',
    'act.lower': 'Lower',
    'act.reasonPlaceholder': 'Why? (required — kept in the record)',
    'act.confirmRaise': 'Confirm raise',
    'act.confirmLower': 'Confirm lower',
    'act.cancel': 'Cancel',
    'act.saving': 'Saving…',
    'act.withdraw': 'Withdraw',
    'act.priorityRaised': 'PRIORITY RAISED → ESI {n}',
    'act.priorityLowered': 'PRIORITY LOWERED → ESI {n}',
    'act.overrideMeta': 'was ESI {prior} · model risk {risk} at the time',
    'act.overrideFoot': 'the model keeps scoring underneath and no re-score clears this.',
    'act.ackFlag': 'Acknowledge flag',
    'act.acknowledge': 'Acknowledge',
    'act.acknowledged': 'Acknowledged',
    'act.ackBy': 'by {by} · risk was {risk}',
    'act.ackNote':
      'Flag stays in the record and the score keeps updating. It will escalate again on its own if risk reaches',
    'act.notePlaceholder': 'Note (optional)',

    'unit.h': 'h',
    'unit.m': 'm',
    'scenario.septic_elderly.label': 'Septic elderly',
    'scenario.septic_elderly.blurb': 'ESI 2 · already unwell, decompensates over ~45 min',
    'scenario.silent_deteriorator.label': 'Silent deteriorator',
    'scenario.silent_deteriorator.blurb': 'ESI 4 · normal vitals, drifts for 40 min, should trip re-triage',
    'scenario.stable.label': 'Stable',
    'scenario.stable.blurb': 'ESI 3 · no decline — the control, board should stay quiet',
    'bed.title': 'Bed board',
    'bed.capacity': 'capacity {n}',
    'bed.inDept': 'In department',
    'bed.occupancy': '{p}% of {n} spaces',
    'bed.over': ' — over capacity',
    'bed.expected': 'Expected admissions',
    'bed.expectedNote':
      'sum of probabilities across the census — the planning number, independent of any alert threshold',
    'bed.aboveThreshold': 'Above bed threshold',
    'bed.aboveNote': "counted at the model's ~80% sensitivity point",
    'bed.unstable': 'Unstable now',
    'bed.unstableNote': 'NEWS2 ≥ 7 — escalate on the patient, not the forecast',
    'bed.mostLikely': 'Most likely to need a bed',
    'bed.nobody': 'Nobody above threshold.',
  },

  ar: {
    'app.brand': 'قائمة المراقبة',
    'app.sub': '/ لوحة الطوارئ',
    'app.sim': 'محاكاة',
    'app.inDept': 'في القسم',
    'app.retriage': 'إعادة فرز',
    'app.acked': 'أُقرّ {n}',
    'app.alerting': 'تنبيه',
    'app.patientDetail': 'تفاصيل المريض',
    'app.selectPatient':
      'اختر مريضاً لعرض اتجاهات العلامات الحيوية وسبب تقييم النموذج له.',
    'conn.live': 'مباشر',
    'conn.connecting': 'جارٍ الاتصال',
    'conn.reconnecting': 'إعادة الاتصال',

    'ctl.replay': 'إعادة التشغيل',
    'ctl.pause': 'إيقاف مؤقت',
    'ctl.resume': 'استئناف',
    'ctl.reset': 'إعادة ضبط',
    'ctl.speed': 'السرعة',
    'ctl.scrub': 'التمرير',
    'ctl.scrubAria': 'تمرير موضع إعادة التشغيل',
    'ctl.inject': 'إدراج',
    'ctl.adding': 'جارٍ الإضافة…',
    'ctl.language': 'اللغة',

    'board.title': 'لوحة المرضى · مرتبة حسب الخطورة لا حسب وقت الوصول',
    'board.flags': '{n} تنبيهات إعادة فرز',
    'board.flag': 'تنبيه إعادة فرز واحد',
    'board.acknowledged': 'أُقرّ {n}',
    'board.empty': 'لا يوجد مرضى في القسم.',
    'board.connecting': 'جارٍ الاتصال باللوحة…',
    'col.patient': 'المريض',
    'col.triage': 'الفرز',
    'col.risk': 'خطر التدهور',
    'col.trend': 'الاتجاه (30 د)',
    'col.waiting': 'الانتظار',
    'col.bed': 'يحتاج سريراً',
    'col.status': 'الحالة',
    'row.esi': 'ESI',
    'row.was': 'كان {n}',
    'row.likelyBed': 'سرير مرجّح',
    'row.bedUnlikely': 'سرير غير مرجّح',
    'row.retriageTo': 'إعادة فرز ← ESI {n}',
    'row.ack': 'أُقرّ',
    'row.raised': 'رُفعت',
    'row.lowered': 'خُفضت',

    'tier.critical': 'حرج',
    'tier.high': 'مرتفع',
    'tier.elevated': 'متزايد',
    'tier.low': 'منخفض',
    'trend.up-fast': 'ارتفاع سريع',
    'trend.up': 'ارتفاع',
    'trend.down-fast': 'انخفاض سريع',
    'trend.down': 'انخفاض',
    'trend.flat': 'مستقر',
    'trend.new': 'جديد',
    'state.unstable_now': 'غير مستقر الآن',
    'state.early_warning': 'إنذار مبكر',
    'state.watch': 'مراقبة',
    'state.news2_elevated': 'ارتفاع NEWS2',
    'state.stable': 'مستقر',

    'vital.hr': 'معدل النبض',
    'vital.rr': 'معدل التنفس',
    'vital.spo2': 'تشبع الأكسجين',
    'vital.sbp': 'الضغط الانقباضي',
    'vital.dbp': 'الضغط الانبساطي',
    'vital.temp': 'درجة الحرارة',

    'detail.close': 'إغلاق',
    'detail.inDept': '{d} في القسم',
    'detail.risk': 'الخطورة',
    'detail.news2': 'NEWS2',
    'detail.needsBed': 'يحتاج سريراً',
    'detail.instability': 'عدم الاستقرار: {v}',
    'instability.low': 'منخفض',
    'instability.medium': 'متوسط',
    'instability.high': 'مرتفع',
    'instability.unknown': 'غير معروف',
    'detail.overrideBy': 'ESI {n} · {by}',
    'detail.aboveThreshold': 'فوق العتبة',
    'detail.belowThreshold': 'دون العتبة',
    'detail.ood': 'NEWS2 ≥ 7 — غير مستقر بالفعل؛ درجة الخطورة هنا استقراء خارج نطاق التدريب',
    'detail.whyScore': 'سبب هذه الدرجة',
    'detail.whyBed': 'سبب ترجيح السرير',
    'detail.trajectory': 'مسار الخطورة',
    'detail.trajectoryNote':
      'سجل الدرجات منذ الوصول. الخط المتقطع هو عتبة التنبيه المختارة عند التدريب لحساسية ~80%. يبدأ السجل من جديد بعد التمرير.',
    'detail.vitals': 'العلامات الحيوية',
    'detail.vitalsNote':
      'النطاق المظلّل هو المدى الطبيعي. المسار أهم من القراءة اللحظية — القيمة التي تتحرك داخل النطاق الطبيعي هي الحالة التي وُجدت هذه اللوحة لالتقاطها.',
    'detail.featureDetail': 'تفاصيل الخصائص',
    'detail.featureNote':
      'الخصائص الفردية خلف الأسباب أعلى هذه اللوحة. مساهمات دقيقة لكل خاصية (TreeSHAP) — مجموعها يساوي اللوغاريتم الاحتمالي للدرجة، أي أن النموذج يفسّر نفسه ولا يخمّنه نموذج آخر.',
    'detail.scenario': 'سيناريو',
    'detail.manualCount': '{n} يدوي',
    'detail.loading': 'جارٍ التحميل…',
    'detail.loadError': 'تعذّر تحميل المريض: {e}',

    'reasons.raises': 'يرفع الخطورة',
    'reasons.lowers': 'يخفض الخطورة',
    'reasons.empty': 'لا توجد تسجيلات كافية بعد لتحديد سبب هذه الدرجة.',
    'reasons.note':
      'مرتّبة حسب إجمالي مساهمة كل مؤشر (TreeSHAP)، فالأسباب الثلاثة ثلاثة أمور مختلفة عن المريض — لا اتجاه واحد مكرر ثلاث مرات. مرّر المؤشر على أي سطر لرؤية الخاصية الأساسية.',
    'reasons.ood': ' هذا المريض غير مستقر بالفعل، لذا فإن الدرجة نفسها استقراء.',

    'act.title': 'التحكم اليدوي',
    'act.chartVitals': 'تسجيل العلامات الحيوية',
    'act.chart': 'تسجيل {n} قيم',
    'act.chart0': 'تسجيل القيم',
    'act.chart1': 'تسجيل قيمة واحدة',
    'act.charting': 'جارٍ التسجيل…',
    'act.vitalsNote':
      'القيم الباهتة هي القراءة الحالية. أي حقل يُترك فارغاً يُرحّل من القراءة السابقة. التسجيل اليدوي يُخرج هذا المريض من مسار إعادة التشغيل.',
    'act.overrideOpen': 'تجاوز الأولوية…',
    'act.raise': 'رفع (أشد خطورة)',
    'act.lower': 'خفض',
    'act.reasonPlaceholder': 'السبب؟ (إلزامي — يُحفظ في السجل)',
    'act.confirmRaise': 'تأكيد الرفع',
    'act.confirmLower': 'تأكيد الخفض',
    'act.cancel': 'إلغاء',
    'act.saving': 'جارٍ الحفظ…',
    'act.withdraw': 'سحب',
    'act.priorityRaised': 'رُفعت الأولوية ← ESI {n}',
    'act.priorityLowered': 'خُفضت الأولوية ← ESI {n}',
    'act.overrideMeta': 'كان ESI {prior} · خطورة النموذج {risk} حينها',
    'act.overrideFoot': 'يستمر النموذج في التقييم أسفل ذلك، ولا يلغي أي إعادة تقييم هذا القرار.',
    'act.ackFlag': 'الإقرار بالتنبيه',
    'act.acknowledge': 'إقرار',
    'act.acknowledged': 'تم الإقرار',
    'act.ackBy': 'بواسطة {by} · كانت الخطورة {risk}',
    'act.ackNote':
      'يبقى التنبيه في السجل وتستمر الدرجة في التحديث. وسيتصاعد تلقائياً مرة أخرى إذا بلغت الخطورة',
    'act.notePlaceholder': 'ملاحظة (اختياري)',

    'unit.h': 'س‎',
    'unit.m': 'د‎',
    'scenario.septic_elderly.label': 'مسن مصاب بإنتان',
    'scenario.septic_elderly.blurb': 'ESI 2 · متدهور أصلاً، ينهار خلال ~45 دقيقة',
    'scenario.silent_deteriorator.label': 'تدهور صامت',
    'scenario.silent_deteriorator.blurb': 'ESI 4 · علامات طبيعية، تنحرف خلال 40 دقيقة، ويُتوقع أن تُطلق إعادة الفرز',
    'scenario.stable.label': 'مستقر',
    'scenario.stable.blurb': 'ESI 3 · لا تدهور — الحالة الضابطة، يُفترض أن تبقى اللوحة هادئة',
    'bed.title': 'لوحة الأسرة',
    'bed.capacity': 'السعة {n}',
    'bed.inDept': 'في القسم',
    'bed.occupancy': '{p}% من {n} مساحة',
    'bed.over': ' — تجاوز السعة',
    'bed.expected': 'الحالات المتوقع إدخالها',
    'bed.expectedNote':
      'مجموع الاحتمالات عبر القسم — رقم التخطيط، مستقل عن أي عتبة تنبيه',
    'bed.aboveThreshold': 'فوق عتبة السرير',
    'bed.aboveNote': 'محسوبة عند نقطة حساسية النموذج ~80%',
    'bed.unstable': 'غير مستقر الآن',
    'bed.unstableNote': 'NEWS2 ≥ 7 — التصعيد على المريض لا على التوقع',
    'bed.mostLikely': 'الأكثر ترجيحاً للحاجة إلى سرير',
    'bed.nobody': 'لا أحد فوق العتبة.',
  },
}

export const LANGUAGES = [
  { code: 'en', label: 'EN', name: 'English', rtl: false },
  { code: 'ar', label: 'ع', name: 'العربية', rtl: true },
]

const STORAGE_KEY = 'watchlist.lang'
const LangContext = createContext(null)

function read() {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return STRINGS[v] ? v : 'en'
  } catch {
    return 'en' // private window, blocked storage — English is a fine default
  }
}

export function LangProvider({ children }) {
  const [lang, setLangState] = useState(read)
  const rtl = LANGUAGES.find((l) => l.code === lang)?.rtl ?? false

  // dir and lang go on <html>, not a wrapper: the browser needs them for text
  // selection, form controls, scrollbar placement and the bidi algorithm, and
  // none of that follows a nested element.
  useEffect(() => {
    document.documentElement.dir = rtl ? 'rtl' : 'ltr'
    document.documentElement.lang = lang
  }, [lang, rtl])

  const value = useMemo(() => {
    const table = STRINGS[lang] || STRINGS.en
    const t = (key, vars, fallback) => {
      // Falling back to the English string is better than showing a raw key:
      // a missed translation should degrade to a readable board, not a broken
      // one. Falling back to the key itself would put "act.withdraw" on screen.
      let out = table[key] ?? STRINGS.en[key] ?? fallback ?? key
      if (vars) {
        for (const [k, v] of Object.entries(vars)) out = out.replaceAll(`{${k}}`, v)
      }
      return out
    }
    const setLang = (next) => {
      setLangState(next)
      try {
        localStorage.setItem(STORAGE_KEY, next)
      } catch {
        /* preference just will not persist */
      }
    }
    return { lang, rtl, t, setLang }
  }, [lang, rtl])

  return <LangContext.Provider value={value}>{children}</LangContext.Provider>
}

export function useT() {
  const ctx = useContext(LangContext)
  if (!ctx) throw new Error('useT must be used inside <LangProvider>')
  return ctx
}
