export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

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

// object crop → Top-K (with optional class filtering + optional descriptor viz)
export async function searchTopK_2D({
  blob,
  filename,
  topK = 20,
  queryClass = null,
  sameClassOnly = true,
  metric = "cosine",
  includeViz = false // ✅ NEW
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

  // ✅ NEW: ask backend to include descriptor visualizations
  url.searchParams.set("include_viz", String(Boolean(includeViz)))

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

/**
 * Upload a 3D model file and search for similar models using the 3D-topk API.
 * 
 * @param {Object} options
 * @param {File} options.file - The 3D model file (.obj, .stl, .glb, .ply)
 * @param {number} options.topK - Number of similar models to return (default: 10)
 * @param {string} options.method - Descriptor method: 'lfd' or 'depth' (default: 'depth')
 * @param {string} options.metric - Distance metric: 'l2', 'l1', or 'cosine' (default: 'l2')
 * @param {string} options.aggregation - Aggregation: 'mean' or 'sum' (default: 'mean')
 * @param {number} options.imageSize - Rendering resolution (default: 256)
 * @param {boolean} options.l2Normalize - Apply L2 normalization (default: false)
 * @returns {Promise<Object>} API response with results array
 */
export async function searchTopK_3d({
  file,
  topK = 10,
  method = "depth",
  metric = "cosine",
  aggregation = "mean",
  imageSize = 256,
  l2Normalize = false
}) {
  if (!file) {
    throw new Error("No file provided")
  }

  const form = new FormData()
  form.append("file", file)

  // Build URL with query parameters
  const url = new URL(`${API_BASE}/api/search/3D-topk`)
  url.searchParams.set("top_k", String(Math.max(1, Math.min(100, Number(topK) || 10))))
  url.searchParams.set("method", String(method).toLowerCase())
  url.searchParams.set("metric", String(metric).toLowerCase())
  url.searchParams.set("aggregation", String(aggregation).toLowerCase())
  url.searchParams.set("image_size", String(imageSize))
  url.searchParams.set("l2_normalize", String(Boolean(l2Normalize)).toLowerCase())

  const res = await fetch(url.toString(), {
    method: "POST",
    body: form
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || "3D Top-K search failed")
  }

  return res.json()
}
