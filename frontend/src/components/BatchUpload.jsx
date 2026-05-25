import { useState } from "react"

const API_BASE = "https://ai-lead-scoring-system.onrender.com"

const CAT = {
  Hot:  { cls: "text-red-300 bg-red-500/10 border-red-500/25",    bar: "#ef4444", card: "border-red-500/30 bg-red-500/5" },
  Warm: { cls: "text-amber-300 bg-amber-500/10 border-amber-500/25", bar: "#f59e0b", card: "border-amber-500/30 bg-amber-500/5" },
  Cold: { cls: "text-blue-300 bg-blue-500/10 border-blue-500/25",  bar: "#3b82f6", card: "border-blue-500/30 bg-blue-500/5" },
}

const SAMPLE_JSON = JSON.stringify(
  [
    { lead_id: "L001", lead_source: "Referral",     industry: "SaaS",         company_size: "Enterprise", region: "North America", website_visits: 15, email_opens: 8,  email_clicks: 5, demo_requested: 1, days_since_interaction: 1,  followup_count: 4 },
    { lead_id: "L002", lead_source: "LinkedIn Ads", industry: "Fintech",      company_size: "Mid-Market", region: "Europe",        website_visits: 3,  email_opens: 1,  email_clicks: 0, demo_requested: 0, days_since_interaction: 30, followup_count: 1 },
    { lead_id: "L003", lead_source: "Website",      industry: "E-commerce",   company_size: "SMB",        region: "Asia Pacific",  website_visits: 7,  email_opens: 4,  email_clicks: 2, demo_requested: 1, days_since_interaction: 5,  followup_count: 2 },
    { lead_id: "L004", lead_source: "Cold Call",    industry: "Manufacturing", company_size: "SMB",        region: "Latin America", website_visits: 1,  email_opens: 0,  email_clicks: 0, demo_requested: 0, days_since_interaction: 60, followup_count: 0 },
    { lead_id: "L005", lead_source: "Referral",     industry: "Healthcare",   company_size: "Enterprise", region: "North America", website_visits: 20, email_opens: 12, email_clicks: 8, demo_requested: 1, days_since_interaction: 2,  followup_count: 6 },
  ],
  null,
  2
)

function Spinner() {
  return (
    <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
    </svg>
  )
}

function SummaryCard({ label, count, total, color }) {
  const pct = total ? Math.round((count / total) * 100) : 0
  return (
    <div className={`border rounded-xl p-4 ${CAT[label]?.card ?? "border-slate-700 bg-slate-800/40"}`}>
      <div className="flex items-center justify-between mb-2">
        <span className={`text-xs font-bold uppercase tracking-wider ${CAT[label]?.cls.split(" ")[0] ?? "text-slate-400"}`}>
          {label === "Hot" ? "🔥" : label === "Warm" ? "🌡️" : "❄️"} {label}
        </span>
        <span className="text-xl font-bold text-white">{count}</span>
      </div>
      <div className="h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <p className="text-xs text-slate-500 mt-1.5">{pct}% of batch</p>
    </div>
  )
}

export default function BatchUpload() {
  const [json, setJson]       = useState("")
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError]     = useState(null)

  async function handleScore() {
    setError(null); setResults(null)

    let leads
    try {
      const parsed = JSON.parse(json)
      leads = Array.isArray(parsed) ? parsed : parsed.leads
      if (!Array.isArray(leads) || leads.length === 0)
        throw new Error("Expected a JSON array of lead objects (or { leads: [...] }).")
    } catch (e) {
      setError("JSON parse error: " + e.message)
      return
    }

    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/v1/score/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ leads }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        const detail = body.detail
        const msg = Array.isArray(detail)
          ? detail.map(d => `${d.loc?.slice(-1)[0] ?? "field"}: ${d.msg}`).join(" · ")
          : (detail ?? `HTTP ${res.status}`)
        throw new Error(msg)
      }
      setResults(await res.json())
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const counts = results
    ? results.scored_leads.reduce((a, l) => { a[l.lead_category] = (a[l.lead_category] || 0) + 1; return a }, {})
    : {}
  const total = results?.scored_leads.length ?? 0

  return (
    <div className="space-y-6 animate-in">
      {/* Input */}
      <div className="bg-slate-800/40 border border-slate-700/50 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-semibold text-white">Paste JSON</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Array of lead objects or{" "}
              <span className="font-mono bg-slate-900/60 px-1 rounded">{"{ leads: [...] }"}</span>
            </p>
          </div>
          <button
            onClick={() => setJson(SAMPLE_JSON)}
            className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            Load 5 sample leads →
          </button>
        </div>

        <textarea
          className="w-full h-56 bg-slate-900/80 border border-slate-700 rounded-xl px-4 py-3 text-xs font-mono text-slate-300 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/40 resize-none transition-colors"
          placeholder={`[\n  {\n    "lead_id": "L001",\n    "lead_source": "Referral",\n    "industry": "SaaS",\n    "company_size": "Enterprise",\n    ...\n  }\n]`}
          value={json}
          onChange={e => setJson(e.target.value)}
          spellCheck={false}
        />

        {/* Enum quick-reference */}
        <div className="mt-3 grid grid-cols-1 gap-y-1 text-xs text-slate-600">
          {[
            ["lead_source",  "Website · Referral · LinkedIn Ads · Cold Call · Conference · Partner"],
            ["industry",     "SaaS · Fintech · E-commerce · Healthcare · Manufacturing · Consulting"],
            ["company_size", "SMB · Mid-Market · Enterprise"],
            ["region",       "North America · Europe · Asia Pacific · Latin America"],
          ].map(([field, vals]) => (
            <p key={field}>
              <span className="font-mono text-slate-500">{field}:</span>{" "}{vals}
            </p>
          ))}
        </div>

        <div className="flex items-center justify-end mt-4">
          <button
            onClick={handleScore}
            disabled={!json.trim() || loading}
            className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold text-sm transition-all duration-150 flex items-center gap-2 shadow-lg shadow-indigo-500/20"
          >
            {loading ? <><Spinner /> Scoring…</> : "Score Batch"}
          </button>
        </div>

        {error && (
          <p className="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 font-mono whitespace-pre-wrap">
            ⚠ {error}
          </p>
        )}
      </div>

      {/* Results */}
      {results && (
        <div className="space-y-4 animate-in">
          {/* Summary cards */}
          <div className="grid grid-cols-3 gap-4">
            {["Hot", "Warm", "Cold"].map(cat => (
              <SummaryCard key={cat} label={cat} count={counts[cat] ?? 0} total={total} color={CAT[cat].bar} />
            ))}
          </div>

          {/* Table */}
          <div className="bg-slate-800/40 border border-slate-700/50 rounded-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-700/60 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">
                {total} leads ranked by score
              </h3>
              <span className="text-xs text-slate-500">
                ROC-AUC <span className="font-mono text-slate-300">{results.model_roc_auc.toFixed(4)}</span>
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-900/40 border-b border-slate-700/40">
                    {["#", "Lead ID", "Score", "Category", "Probability", "Action"].map(h => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/30">
                  {results.scored_leads.map((lead, i) => (
                    <tr key={lead.lead_id} className="hover:bg-slate-700/20 transition-colors">
                      <td className="px-4 py-3 text-slate-600 font-mono text-xs">#{i + 1}</td>
                      <td className="px-4 py-3 text-white font-mono text-xs font-medium">{lead.lead_id}</td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2.5">
                          <div className="w-20 h-1.5 bg-slate-700/60 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-700"
                              style={{ width: `${lead.lead_quality_score}%`, backgroundColor: CAT[lead.lead_category]?.bar }}
                            />
                          </div>
                          <span className="font-mono font-bold text-white text-xs w-6 text-right">{lead.lead_quality_score}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${CAT[lead.lead_category]?.cls}`}>
                          {lead.lead_category}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-300">{(lead.conversion_probability * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3 text-xs text-slate-400 max-w-xs truncate">{lead.recommended_action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
