const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export async function detectObjects(file) {
  const form = new FormData()
  form.append("file", file) // must match FastAPI param

  const res = await fetch(`${API_BASE}/api/detect`, {
    method: "POST",
    body: form
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || "Detect failed")
  }

  return res.json()
}

// ✅ ONLY retrieval endpoint (object crop → Top-K)
export async function searchTopK({ blob, filename, topK = 20 }) {
  const form = new FormData()

  form.append(
    "file",
    new File([blob], filename, {
      type: blob.type || "image/png"
    })
  )

  const k = Math.max(1, Math.min(200, Number(topK) || 20))

  const res = await fetch(`${API_BASE}/api/search/topk?top_k=${k}`, {
    method: "POST",
    body: form
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || "Top-K failed")
  }

  return res.json()
}
