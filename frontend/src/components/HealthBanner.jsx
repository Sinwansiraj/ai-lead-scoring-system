export default function HealthBanner({ health }) {
  if (!health) {
    return (
      <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-2">
        <div className="max-w-6xl mx-auto flex items-center gap-2 text-xs text-slate-500">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-slate-600 animate-pulse" />
          Connecting to API…
        </div>
      </div>
    )
  }

  const ok = health.status === "ok" && health.model_loaded

  return (
    <div
      className={`border-b px-6 py-2 transition-colors ${
        ok ? "bg-emerald-950/60 border-emerald-900" : "bg-amber-950/60 border-amber-900"
      }`}
    >
      <div className="max-w-6xl mx-auto flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-1.5 h-1.5 rounded-full animate-pulse ${
              ok ? "bg-emerald-400" : "bg-amber-400"
            }`}
          />
          <span className={ok ? "text-emerald-300" : "text-amber-300"}>
            {ok
              ? "Model loaded — live predictions active"
              : "Model not loaded — API may be warming up (cold start ~60 s on free tier)"}
          </span>
        </div>
        <div className="flex items-center gap-4 text-slate-500">
          {health.model_roc_auc != null && (
            <span>
              ROC-AUC{" "}
              <span className="font-mono text-slate-300">{health.model_roc_auc.toFixed(4)}</span>
            </span>
          )}
          <span>v{health.version ?? "—"}</span>
        </div>
      </div>
    </div>
  )
}
