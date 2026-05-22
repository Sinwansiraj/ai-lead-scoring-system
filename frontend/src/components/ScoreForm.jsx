import { useState } from "react"

const ENUM_FIELDS = {
  lead_source:  ["Referral", "Cold Call", "Website", "LinkedIn", "Email Campaign", "Partner", "Trade Show", "Other"],
  industry:     ["SaaS", "FinTech", "HealthTech", "E-commerce", "Manufacturing", "Consulting", "Education", "Other"],
  company_size: ["Startup", "SMB", "Mid-Market", "Enterprise"],
  region:       ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa"],
}

const NUM_FIELDS = [
  { key: "website_visits",        label: "Website Visits" },
  { key: "email_opens",           label: "Email Opens" },
  { key: "email_clicks",          label: "Email Clicks" },
  { key: "days_since_interaction",label: "Days Since Contact" },
  { key: "followup_count",        label: "Follow-up Count" },
]

const DEFAULTS = {
  lead_id: "L00042",
  lead_source: "Referral",
  industry: "SaaS",
  company_size: "Enterprise",
  region: "North America",
  website_visits: 12,
  email_opens: 6,
  email_clicks: 4,
  demo_requested: 1,
  days_since_interaction: 2,
  followup_count: 3,
}

const inputCls =
  "w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white " +
  "placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 " +
  "focus:ring-indigo-500/50 transition-colors"

function Field({ label, children }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-400 mb-1.5">{label}</label>
      {children}
    </div>
  )
}

export default function ScoreForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS)

  function set(key, value) {
    setForm(f => ({ ...f, [key]: value }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    onSubmit({
      ...form,
      website_visits:         Number(form.website_visits),
      email_opens:            Number(form.email_opens),
      email_clicks:           Number(form.email_clicks),
      demo_requested:         Number(form.demo_requested),
      days_since_interaction: Number(form.days_since_interaction),
      followup_count:         Number(form.followup_count),
    })
  }

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-6">
      <h2 className="text-base font-semibold text-white mb-5 flex items-center gap-2">
        <span className="w-6 h-6 rounded-md bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-xs font-bold">
          1
        </span>
        Lead Details
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Lead ID */}
        <Field label="Lead ID">
          <input
            className={inputCls}
            value={form.lead_id}
            onChange={e => set("lead_id", e.target.value)}
            placeholder="L00042"
          />
        </Field>

        {/* Enum dropdowns — 2 columns */}
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(ENUM_FIELDS).map(([key, options]) => (
            <Field key={key} label={key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}>
              <select
                className={inputCls}
                value={form[key]}
                onChange={e => set(key, e.target.value)}
              >
                {options.map(o => <option key={o}>{o}</option>)}
              </select>
            </Field>
          ))}
        </div>

        {/* Numeric fields — 3 columns */}
        <div className="grid grid-cols-3 gap-3">
          {NUM_FIELDS.map(({ key, label }) => (
            <Field key={key} label={label}>
              <input
                type="number"
                min={0}
                className={inputCls}
                value={form[key]}
                onChange={e => set(key, e.target.value)}
              />
            </Field>
          ))}

          <Field label="Demo Requested">
            <select
              className={inputCls}
              value={form.demo_requested}
              onChange={e => set("demo_requested", Number(e.target.value))}
            >
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </Field>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 px-6 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm transition-all duration-150 shadow-lg flex items-center justify-center gap-2 mt-2"
        >
          {loading ? (
            <>
              <Spinner />
              Scoring…
            </>
          ) : "⚡ Score Lead"}
        </button>
      </form>
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}
