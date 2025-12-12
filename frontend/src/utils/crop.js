function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v))
}

export async function cropToBlob(
  imageUrl,
  bbox,
  padRatio = 0.03,
  mime = "image/jpeg",
  quality = 0.92
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

  const padX = Math.round(bbox.w * padRatio)
  const padY = Math.round(bbox.h * padRatio)

  const x = clamp(bbox.x - padX, 0, W - 1)
  const y = clamp(bbox.y - padY, 0, H - 1)
  const w = clamp(bbox.w + 2 * padX, 1, W - x)
  const h = clamp(bbox.h + 2 * padY, 1, H - y)

  const canvas = document.createElement("canvas")
  canvas.width = w
  canvas.height = h

  const ctx = canvas.getContext("2d")
  ctx.drawImage(img, x, y, w, h, 0, 0, w, h)

  const blob = await new Promise((resolve) =>
    canvas.toBlob(resolve, mime, quality)
  )
  if (!blob) throw new Error("Failed to crop image to blob")

  const previewUrl = URL.createObjectURL(blob)

  return {
    blob,
    previewUrl,
    cropBox: { x, y, w, h },
    originalSize: { W, H }
  }
}
