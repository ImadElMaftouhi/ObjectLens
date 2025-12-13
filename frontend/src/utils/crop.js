function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v))
}

export async function cropToBlob(
  imageUrl,
  bbox,
  mime = "image/png" // lossless -> preserves pixels exactly (no recompression artifacts)
) {
  const img = new Image()
  img.crossOrigin = "anonymous"
  img.src = imageUrl

  await new Promise((resolve, reject) => {
    img.onload = resolve
    img.onerror = reject
  })

  const W = img.naturalWidth
  const H = img.naturalHeight

  // No padding: crop exactly the bbox
  const x = clamp(Math.round(bbox.x), 0, W - 1)
  const y = clamp(Math.round(bbox.y), 0, H - 1)
  const w = clamp(Math.round(bbox.w), 1, W - x)
  const h = clamp(Math.round(bbox.h), 1, H - y)

  const canvas = document.createElement("canvas")
  canvas.width = w
  canvas.height = h

  const ctx = canvas.getContext("2d")
  ctx.drawImage(img, x, y, w, h, 0, 0, w, h)

  const blob = await new Promise((resolve) => {
    // quality param is ignored for PNG (lossless)
    canvas.toBlob(resolve, mime)
  })
  if (!blob) throw new Error("Failed to crop image to blob")

  const previewUrl = URL.createObjectURL(blob)

  return {
    blob,
    previewUrl,
    cropBox: { x, y, w, h },
    originalSize: { W, H }
  }
}
