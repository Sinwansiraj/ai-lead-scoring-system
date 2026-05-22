import { useState, useEffect } from "react"
import HealthBanner from "./components/HealthBanner.jsx"
import ScoreForm from "./components/ScoreForm.jsx"
import ScoreResult from "./components/ScoreResult.jsx"
import BatchUpload from "./components/BatchUpload.jsx"

export const API_BASE = "https://ai-lead-scoring-system.onrender.com"

export default function App() {
  const [activeTab, setActiveTab] = useState("score")
  const [health, setHealth] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchHealth()
    const id = setInterval(fetchHealth, 30_000)
    return () => clearInterval(id)
  }, [])

  function fetchHealth() {
    fetch(`${API_BASE}/api/v1/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "error", model_loaded: false, version: "—" }))
  }

  async function handleScore(formData) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_BASE}/api/v1/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      <HealthBanner health={health} />

      {/* Header */}
      <header className="border-b border-slate-700/50 px-6 py-4 bg-slate-900/60 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30 text-lg">
              🎯
            </div>
            <div>
              <h1 className="text-lg font-bold text-white leading-none tracking-tight">Lead Scoring Dashboard</h1>
              <p className="text-xs text-slate-500 mt-0.5">AI-driven B2B lead prioritisation</p>
            </div>
          </div>
          <nav className="flex gap-1 bg-slate-800/80 rounded-xl p-1 border border-slate-700/60">
            {[
              { id: "score", label: "Score a Lead", icon: "⚡" },
              { id: "batch", label: "Batch Score",  icon: "📋" },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                  activeTab === tab.id
                    ? "bg-indigo-600 text-white shadow-lg"
                    : "text-slate-400 hover:text-white hover:bg-slate-700"
                }`}
              >
                <span>{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">
        {activeTab === "score" ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <ScoreForm onSubmit={handleScore} loading={loading} />
            <ScoreResult result={result} error={error} loading={loading} />
          </div>
        ) : (
          <BatchUpload />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 px-6 py-4 text-center text-xs text-slate-600">
        Built with FastAPI · XGBoost · React ·{" "}
        <a
          href="https://github.com/Sinwansiraj/ai-lead-scoring-system"
          className="text-indigo-500 hover:text-indigo-400 transition-colors"
          target="_blank" rel="noreferrer"
        >
          GitHub
        </a>{" · "}
        <a
          href={`${API_BASE}/docs`}
          className="text-indigo-500 hover:text-indigo-400 transition-colors"
          target="_blank" rel="noreferrer"
        >
          API Docs
        </a>
      </footer>
    </div>
  )
}
