import { LANGUAGES, useT } from '../lib/i18n.jsx'

/**
 * Language switch.
 *
 * Deliberately two plain buttons rather than a dropdown: there are two
 * languages, a dropdown would hide both behind a click, and someone who has
 * landed on the wrong one cannot read the dropdown's label to find their way
 * out. Each option is written in its own script for the same reason — عربية is
 * findable by a reader who cannot read "Arabic".
 *
 * Switching changes `dir` on the document root and re-requests the reason
 * sentences from the server in the new language. Nothing about the scores,
 * thresholds or flags changes; this is vocabulary and direction only.
 */
export default function LanguagePicker() {
  const { lang, setLang, t } = useT()

  return (
    <span className="langpick" role="group" aria-label={t('ctl.language')}>
      {LANGUAGES.map((l) => (
        <button
          key={l.code}
          className={`btn btn-lang${lang === l.code ? ' active' : ''}`}
          onClick={() => setLang(l.code)}
          aria-pressed={lang === l.code}
          // The name is always in its own script, so the control is legible
          // to someone who cannot read the current interface language.
          title={l.name}
          lang={l.code}
        >
          {l.name}
        </button>
      ))}
    </span>
  )
}
