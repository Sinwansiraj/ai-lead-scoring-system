// Single source of truth for the API base URL.
// Override per environment at build time with VITE_API_BASE (see .env.example).
const raw = import.meta.env.VITE_API_BASE || "https://ai-lead-scoring-system.onrender.com"

// Trim any trailing slash so `${API_BASE}/api/v1/...` is always well-formed.
export const API_BASE = raw.replace(/\/+$/, "")
