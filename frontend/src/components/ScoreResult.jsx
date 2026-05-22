import Gauge from "./Gauge.jsx"

const CARD_THEME = {
  Hot:  "border-red-500/30 shadow-red-500/10",
  Warm: "border-amber-500/30 shadow-amber-500/10",
  Cold: "border-blue-500/30 shadow-blue-500/10",
}

const PROB_BAR = {
  Hot:  "from-red-500 to-red-400",
  Warm: "from-amber-500 to-amber-400",
  Cold: "from-blue-500 to-blue-400",
}

function Spinner() {
  return (
    <svg className="animate-spin w-9 h-9 text-indigo-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

function StatCard({ label, value, sub }) {
  return (
    <div className="bg-slate-900/70 border border-slate-700/40 rounded-xl p-3.5">
      <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider font-medium">{label}</p>
      <p className="text-white font-mono font-bold text-base">{value}</p>
      {sub && <p className="text-slate-600 text-xs mt-0.5">{sub}</p>}
    </div>
  )
}

export default function ScoreResult({ result, error, loading }) {
  if (loading) {
    return (
      <div className="bg-slate-800/40 border border-slate-700/40 rounded-2xl p-8 flex items-center justify-center min-h-72">
        <div className="text-center">
          <Spinner />
          <p className="text-sm text-slate-400">Running inference…</p>
          <p className="text-xs text-slate-600 mt-1">XGBoost · feature pipeline · scoring</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-slate-800/40 border border-red-500/25 rounded-2xl p-6 animate-in">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-full bg-red-500/15 flex items-center justify-center shrink-0 mt-0.5">
            <span className="text-red-400 text-sm">⚠</span>
          </div>
          <div>
            <p className="text-red-300 font-semibold text-sm">Scoring failed</p>
            <p className="text-slate-400 text-xs mt-1.5 font-mono bg-slate-900/50 rounded px-2 py-1">{error}</p>
            <p className="text-slate-500 text-xs mt-2">
              Render's free tier cold-starts after 15 min of inactivity. Wait ~60 s and retry.
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="bg-slate-800/20 border border-dashed border-slate-700/60 rounded-2xl p-8 flex items-center justify-center min-h-72">
        <div className="text-center">
          <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mx-auto mb-4">
            <span className="text-3xl">🎯</span>
          </div>
          <p className="text-slate-400 text-sm">Fill in the lead details and hit</p>
          <p className="text-slate-200 font-semibold text-sm mt-1">⚡ Score Lead</p>
          <p className="text-slate-600 text-xs mt-3">Powered by XGBoost · ROC-AUC 0.89</p>
        </div>
      </div>
    )
  }

  const lead = result.scored_leads[0]
  const cat  = lead.lead_category
  const prob = (lead.conversion_probability * 100).toFixed(1)

  return (
    <div className={`bg-slate-800/50 border rounded-2xl p-6 space-y-5 shadow-xl ${CARD_THEME[cat] ?? "border-slate-700/50"} animate-in`}>
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <span className="w-6 h-6 rounded-md bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xs font-bold">2</span>
          Score Result
        </h2>
        <span className="text-xs font-mono text-slate-500 bg-slate-900/60 px-2 py-1 rounded-md">{lead.lead_id}</span>
      </div>

      {/* Gauge */}
      <Gauge score={lead.lead_quality_score} category={cat} />

      {/* Probability bar */}
      <div>
        <div className="flex justify-between text-xs text-slate-500 mb-1.5">
          <span>Conversion Probability</span>
          <span className="font-mono text-slate-300 font-semibold">{prob}%</span>
        </div>
        <div className="h-2 bg-slate-700/60 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${PROB_BAR[cat]} transition-all duration-1000`}
            style={{ width: `${prob}%` }}
          />
        </div>
      </div>

      {/* Recommended action */}
      <div className="bg-slate-900/60 border border-slate-700/40 rounded-xl p-4">
        <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-2">Recommended Action</p>
        <p className="text-white font-semibold text-sm leading-relaxed">{lead.recommended_action}</p>
      </div>

      {/* Stat grid */}
      <div className="grid grid-cols-2 gap-2.5">
        <StatCard label="Engagement Score" value={lead.engagement_score.toFixed(1)} sub="/ 100" />
        <StatCard label="Recency Score"     value={lead.recency_score.toFixed(1)}    sub="/ 100" />
        <StatCard label="Model ROC-AUC"     value={result.model_roc_auc.toFixed(4)}  sub="XGBoost" />
        <StatCard label="API Version"       value={result.model_version}             sub="live on Render" />
      </div>
    </div>
  )
}
