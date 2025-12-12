const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export async function detectObjects(file) {
  const form = new FormData()
  form.append("file", file) // MUST match FastAPI param name: file

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

// optional debug endpoint
export async function sendSelectedCrop({ blob, filename, meta }) {
  const form = new FormData()
  form.append(
    "crop",
    new File([blob], filename, { type: blob.type || "image/jpeg" })
  )

  if (meta?.class_name) form.append("class_name", meta.class_name)
  if (typeof meta?.confidence === "number")
    form.append("confidence", String(meta.confidence))
  if (meta?.source_detection_id)
    form.append("source_detection_id", String(meta.source_detection_id))
  if (meta?.image_id) form.append("image_id", String(meta.image_id))

  const res = await fetch(`${API_BASE}/api/search/select-object`, {
    method: "POST",
    body: form
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || "Select-object failed")
  }

  return res.json()
}

// ✅ real retrieval endpoint
export async function searchTopK({ blob, filename, topK = 20 }) {
  const form = new FormData()
  // MUST match backend param name in /api/search/topk: file: UploadFile
  form.append(
    "file",
    new File([blob], filename, { type: blob.type || "image/jpeg" })
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
