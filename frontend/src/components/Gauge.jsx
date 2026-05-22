import { useState, useEffect } from "react"

const THEME = {
  Hot:  { stroke: "#ef4444", glow: "#ef444488", badge: "bg-red-500/15 border-red-500/40 text-red-300",    label: "🔥 Hot"  },
  Warm: { stroke: "#f59e0b", glow: "#f59e0b88", badge: "bg-amber-500/15 border-amber-500/40 text-amber-300", label: "🌡️ Warm" },
  Cold: { stroke: "#3b82f6", glow: "#3b82f688", badge: "bg-blue-500/15 border-blue-500/40 text-blue-300",  label: "❄️ Cold" },
}

function useCountUp(target, delay = 60, duration = 900) {
  const [count, setCount] = useState(0)
  useEffect(() => {
    setCount(0)
    if (!target) return
    const t = setTimeout(() => {
      const start = Date.now()
      const tick = () => {
        const p = Math.min((Date.now() - start) / duration, 1)
        const eased = 1 - Math.pow(1 - p, 3)   // ease-out cubic
        setCount(Math.round(eased * target))
        if (p < 1) requestAnimationFrame(tick)
      }
      requestAnimationFrame(tick)
    }, delay)
    return () => clearTimeout(t)
  }, [target, delay, duration])
  return count
}

export default function Gauge({ score, category }) {
  const displayed = useCountUp(score)

  const r  = 74
  const cx = 100
  const cy = 108
  const C  = 2 * Math.PI * r          // 464.96
  const arc = C * 0.75                 // 270° = 348.72
  const filled = (displayed / 100) * arc

  const { stroke, glow, badge, label } = THEME[category] ?? THEME.Cold

  return (
    <div className="flex flex-col items-center gap-5 animate-in">
      <div style={{ "--glow-color": glow }} className="gauge-glow">
        <svg viewBox="0 0 200 180" className="w-56 mx-auto">
          <defs>
            <filter id="arcGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={stroke} stopOpacity="0.6" />
              <stop offset="100%" stopColor={stroke} />
            </linearGradient>
          </defs>

          {/* Track */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none" stroke="#1e293b" strokeWidth="12" strokeLinecap="round"
            strokeDasharray={`${arc} ${C - arc}`}
            transform={`rotate(135 ${cx} ${cy})`}
          />

          {/* Filled arc with glow */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none" stroke="url(#arcGrad)" strokeWidth="12" strokeLinecap="round"
            strokeDasharray={`${filled} ${C - filled}`}
            transform={`rotate(135 ${cx} ${cy})`}
            filter="url(#arcGlow)"
          />

          {/* Score */}
          <text x={cx} y={cy + 6} textAnchor="middle"
            style={{ fontSize: "46px", fontWeight: "800", fill: "white", fontFamily: "system-ui" }}>
            {displayed}
          </text>
          <text x={cx} y={cy + 26} textAnchor="middle"
            style={{ fontSize: "11px", fill: "#475569", fontFamily: "system-ui", letterSpacing: "0.05em" }}>
            OUT OF 100
          </text>
        </svg>
      </div>

      <span className={`px-6 py-2 rounded-full text-sm font-bold border tracking-wide ${badge}`}>
        {label}
      </span>
    </div>
  )
}
