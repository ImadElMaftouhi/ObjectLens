import { useMemo, useState } from "react"
import { detectObjects, sendSelectedCrop, searchTopK } from "./api"
import { cropToBlob } from "./utils/crop"

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

export default function App() {
  const [file, setFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)

  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState("")

  const [detectResult, setDetectResult] = useState(null)
  const [crops, setCrops] = useState([])
  const [selectedIndex, setSelectedIndex] = useState(null)

  // Top-K
  const [topK, setTopK] = useState(20)
  const [topkResult, setTopkResult] = useState(null)

  const selected = useMemo(() => {
    if (selectedIndex === null) return null
    return crops[selectedIndex] || null
  }, [crops, selectedIndex])

  function resetAll() {
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    crops.forEach((c) => c.previewUrl && URL.revokeObjectURL(c.previewUrl))

    setFile(null)
    setImageUrl(null)
    setDetectResult(null)
    setCrops([])
    setSelectedIndex(null)
    setTopkResult(null)
    setStatus("")
  }

  async function onPickFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    resetAll()
    setFile(f)
    setImageUrl(URL.createObjectURL(f))
    setStatus("Image loaded. Click Detect Objects.")
  }

  async function runDetect() {
    if (!file || !imageUrl) return
    setLoading(true)
    setStatus("Running YOLO detection...")

    try {
      const res = await detectObjects(file)
      setDetectResult(res)

      if (!res?.detections?.length) {
        setCrops([])
        setSelectedIndex(null)
        setTopkResult(null)
        setStatus("No objects detected.")
        return
      }

      setStatus(`Detected ${res.detections.length} object(s). Cropping...`)

      const cropItems = []
      for (const det of res.detections) {
        // IMPORTANT: your crop util expects bbox = {x,y,w,h}
        const { blob, previewUrl } = await cropToBlob(imageUrl, det.bbox)
        cropItems.push({ det, blob, previewUrl })
      }

      setCrops(cropItems)
      setSelectedIndex(null)
      setTopkResult(null)
      setStatus("Pick an object crop, then click Search Top-K.")
    } catch (err) {
      console.error(err)
      setStatus(err?.message || "Detection failed")
    } finally {
      setLoading(false)
    }
  }

  async function sendSelection() {
    if (!selected || !detectResult) return
    setLoading(true)
    setStatus("Sending selected object + running Top-K search...")

    try {
      const det = selected.det

      // Optional debug endpoint (doesn't affect search)
      await sendSelectedCrop({
        blob: selected.blob,
        filename: `crop_${detectResult.image_id || "img"}_${det.id}.jpg`,
        meta: {
          image_id: detectResult.image_id,
          source_detection_id: det.id,
          class_name: det.class_name,
          confidence: det.confidence
        }
      })

      // Real retrieval
      const k = Math.max(1, Math.min(200, Number(topK) || 20))
      const topk = await searchTopK({
        blob: selected.blob,
        filename: `query_${detectResult.image_id || "img"}_${det.id}.jpg`,
        topK: k
      })

      setTopkResult(topk)
      setStatus(
        `Top-K done. Returned ${topk?.best_images?.length || 0} image(s).`
      )
    } catch (err) {
      console.error(err)
      setTopkResult(null)
      setStatus(err?.message || "Search failed")
    } finally {
      setLoading(false)
    }
  }

  const selectedInfo = selected?.det
    ? {
        class_name: selected.det.class_name,
        id: selected.det.id,
        bbox: selected.det.bbox
      }
    : null

  return (
    <div
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: 24,
        fontFamily: "system-ui, Arial"
      }}
    >
      <h2 style={{ marginBottom: 6 }}>ObjectLens — Detection + Top-K</h2>
      <div style={{ opacity: 0.8, marginBottom: 18 }}>
        Upload image → YOLO detects → pick an object crop → Top-K retrieval
      </div>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        {/* Left */}
        <div
          style={{
            flex: 1,
            border: "1px solid #333",
            borderRadius: 12,
            padding: 16
          }}
        >
          <div
            style={{
              display: "flex",
              gap: 12,
              alignItems: "center",
              marginBottom: 12
            }}
          >
            <input type="file" accept="image/*" onChange={onPickFile} />
            <button disabled={!file || loading} onClick={runDetect}>
              Detect Objects
            </button>
            <button disabled={loading} onClick={resetAll}>
              Reset
            </button>

            <div
              style={{
                marginLeft: "auto",
                display: "flex",
                gap: 8,
                alignItems: "center"
              }}
            >
              <span style={{ opacity: 0.8 }}>TopK:</span>
              <input
                type="number"
                min={1}
                max={200}
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                style={{ width: 80 }}
              />
            </div>
          </div>

          {imageUrl ? (
            <img
              src={imageUrl}
              alt="uploaded"
              style={{
                width: "100%",
                maxHeight: 360,
                objectFit: "contain",
                borderRadius: 10
              }}
            />
          ) : (
            <div style={{ padding: 24, opacity: 0.7 }}>
              Pick an image to start.
            </div>
          )}

          <div style={{ marginTop: 12, opacity: 0.9 }}>
            <b>Status:</b> {status || "Idle"}
          </div>
        </div>

        {/* Right */}
        <div
          style={{
            flex: 1.2,
            border: "1px solid #333",
            borderRadius: 12,
            padding: 16
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center"
            }}
          >
            <h3 style={{ margin: 0 }}>Detected Objects</h3>
            <button disabled={!selected || loading} onClick={sendSelection}>
              Search Top-K for Selected
            </button>
          </div>

          {!crops.length ? (
            <div style={{ padding: 18, opacity: 0.7 }}>
              No crops yet. Run detection first.
            </div>
          ) : (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
                gap: 12,
                marginTop: 12
              }}
            >
              {crops.map((c, idx) => {
                const isSel = idx === selectedIndex
                return (
                  <button
                    key={c.det.id}
                    onClick={() => setSelectedIndex(idx)}
                    style={{
                      textAlign: "left",
                      padding: 10,
                      borderRadius: 12,
                      border: isSel ? "2px solid #fff" : "1px solid #444",
                      background: isSel ? "#222" : "#111",
                      cursor: "pointer"
                    }}
                  >
                    <img
                      src={c.previewUrl}
                      alt={`crop-${c.det.id}`}
                      style={{
                        width: "100%",
                        height: 120,
                        objectFit: "contain",
                        borderRadius: 10,
                        background: "#000"
                      }}
                    />
                    <div style={{ marginTop: 8, fontSize: 13 }}>
                      <div>
                        <b>{c.det.class_name}</b>
                      </div>
                      <div style={{ opacity: 0.8 }}>
                        conf: {Number(c.det.confidence).toFixed(3)}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          )}

          {selectedInfo ? (
            <div
              style={{
                marginTop: 14,
                paddingTop: 12,
                borderTop: "1px solid #333",
                opacity: 0.9
              }}
            >
              <b>Selected:</b> {selectedInfo.class_name} (id={selectedInfo.id})
              <div style={{ fontSize: 13, opacity: 0.8 }}>
                bbox: x={selectedInfo.bbox.x}, y={selectedInfo.bbox.y}, w=
                {selectedInfo.bbox.w}, h={selectedInfo.bbox.h}
              </div>
            </div>
          ) : null}

          {/* Top-K results */}
          {topkResult?.ok ? (
            <div
              style={{
                marginTop: 14,
                paddingTop: 12,
                borderTop: "1px solid #333"
              }}
            >
              <h4 style={{ margin: "0 0 8px 0" }}>Top-K Results</h4>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                  gap: 12
                }}
              >
                {(topkResult.best_images || []).map((r, i) => {
                  // image_url from backend is like "/dataset/images/val/xxx.jpg"
                  const src = r.image_url?.startsWith("http")
                    ? r.image_url
                    : `${API_BASE}${r.image_url || ""}`

                  return (
                    <div
                      key={`${r.image_path}-${i}`}
                      style={{
                        border: "1px solid #333",
                        borderRadius: 12,
                        padding: 10,
                        background: "#111"
                      }}
                    >
                      <div
                        style={{ fontSize: 12, opacity: 0.8, marginBottom: 6 }}
                      >
                        #{i + 1} score: {Number(r.score).toFixed(4)}
                      </div>

                      <img
                        src={src}
                        alt={r.image_path}
                        style={{
                          width: "100%",
                          height: 140,
                          objectFit: "contain",
                          borderRadius: 10,
                          background: "#000"
                        }}
                        onError={(e) => {
                          // helpful visual hint if URL is wrong
                          e.currentTarget.style.opacity = "0.3"
                        }}
                      />

                      <div style={{ fontSize: 12, marginTop: 8 }}>
                        <b>{r.best_class_name}</b>
                      </div>

                      <div
                        style={{
                          fontSize: 11,
                          opacity: 0.7,
                          wordBreak: "break-all"
                        }}
                      >
                        {r.image_path}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
