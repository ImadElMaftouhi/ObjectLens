const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export async function detectObjects(file) {
  const form = new FormData()
  form.append("file", file)

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

// object crop → Top-K (with optional class filtering)
export async function searchTopK({
  blob,
  filename,
  topK = 20,
  queryClass = null,
  sameClassOnly = true,
  metric = "cosine"
}) {
  const form = new FormData()

  form.append(
    "file",
    new File([blob], filename, {
      type: blob.type || "image/png"
    })
  )

  if (queryClass) {
    form.append("query_class", queryClass)
  }

  const k = Math.max(1, Math.min(200, Number(topK) || 20))

  const url = new URL(`${API_BASE}/api/search/topk`)
  url.searchParams.set("top_k", String(k))
  url.searchParams.set("metric", String(metric).toLowerCase())

  // optional: only meaningful if queryClass exists
  if (queryClass) {
    url.searchParams.set("same_class_only", String(Boolean(sameClassOnly)))
  }

  const res = await fetch(url.toString(), {
    method: "POST",
    body: form
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || "Top-K failed")
  }

  return res.json()
}
