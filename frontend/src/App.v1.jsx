import { useMemo, useState } from "react"
import { detectObjects, searchTopK } from "./api"
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
    setStatus("Image loaded. Click Detect.")
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
        const { blob, previewUrl } = await cropToBlob(imageUrl, det.bbox)
        cropItems.push({ det, blob, previewUrl })
      }

      setCrops(cropItems)
      setSelectedIndex(null)
      setTopkResult(null)
      setStatus("Select a crop, then Search Top-K.")
    } catch (err) {
      console.error(err)
      setStatus(err?.message || "Detection failed")
    } finally {
      setLoading(false)
    }
  }

  async function runTopK() {
    if (!selected || !detectResult) return
    setLoading(true)
    setStatus("Searching Top-K...")

    try {
      const det = selected.det
      const k = Math.max(1, Math.min(200, Number(topK) || 20))

      const topk = await searchTopK({
        blob: selected.blob,
        filename: `query_${detectResult.image_id || "img"}_${det.id}.png`,
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
        bbox: selected.det.bbox,
        confidence: selected.det.confidence
      }
    : null

  const styles = {
    page: {
      minHeight: "100vh",
      background: "#0b0f17",
      color: "#eaeef7",
      fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial",
      padding: 24
    },
    container: { maxWidth: 1200, margin: "0 auto" },
    header: {
      display: "flex",
      alignItems: "flex-end",
      justifyContent: "space-between",
      gap: 16,
      marginBottom: 18
    },
    titleWrap: { display: "flex", flexDirection: "column", gap: 6 },
    title: { fontSize: 22, fontWeight: 800, margin: 0 },
    subtitle: { margin: 0, opacity: 0.8, fontSize: 13 },
    pill: {
      display: "inline-flex",
      alignItems: "center",
      gap: 8,
      padding: "8px 12px",
      borderRadius: 999,
      border: "1px solid #243045",
      background: "#0f1626",
      fontSize: 12,
      opacity: 0.9
    },
    grid: {
      display: "grid",
      gridTemplateColumns: "1.1fr 0.9fr",
      gap: 16
    },
    card: {
      border: "1px solid #1f2a3d",
      background: "linear-gradient(180deg, #0f1626 0%, #0b0f17 100%)",
      borderRadius: 16,
      padding: 16,
      boxShadow: "0 8px 24px rgba(0,0,0,0.35)"
    },
    cardTitle: { margin: 0, fontSize: 14, opacity: 0.9, fontWeight: 700 },
    row: { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" },
    btn: (variant = "primary") => {
      const base = {
        borderRadius: 12,
        padding: "10px 12px",
        fontWeight: 700,
        fontSize: 13,
        border: "1px solid transparent",
        cursor: "pointer",
        transition: "transform 0.05s ease"
      }
      if (variant === "primary")
        return {
          ...base,
          background: "#4f7cff",
          borderColor: "#4f7cff",
          color: "#061021"
        }
      if (variant === "ghost")
        return {
          ...base,
          background: "transparent",
          borderColor: "#243045",
          color: "#eaeef7"
        }
      return base
    },
    btnDisabled: { opacity: 0.5, cursor: "not-allowed" },
    input: {
      borderRadius: 12,
      border: "1px solid #243045",
      background: "#0b0f17",
      color: "#eaeef7",
      padding: "10px 12px",
      fontSize: 13,
      outline: "none"
    },
    status: {
      marginTop: 12,
      padding: "10px 12px",
      borderRadius: 12,
      border: "1px solid #243045",
      background: "#0b0f17",
      fontSize: 13,
      opacity: 0.95
    },
    imgFrame: {
      marginTop: 12,
      borderRadius: 14,
      overflow: "hidden",
      border: "1px solid #243045",
      background: "#050812"
    },
    mainImg: {
      width: "100%",
      maxHeight: 420,
      objectFit: "contain",
      display: "block"
    },
    sectionGap: { marginTop: 14 },
    cropGrid: {
      marginTop: 12,
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
      gap: 12
    },
    cropBtn: (selected) => ({
      textAlign: "left",
      borderRadius: 14,
      border: selected ? "1px solid #4f7cff" : "1px solid #243045",
      background: selected ? "rgba(79,124,255,0.10)" : "#0b0f17",
      padding: 10,
      cursor: "pointer"
    }),
    cropImg: {
      width: "100%",
      height: 120,
      objectFit: "contain",
      borderRadius: 12,
      background: "#000",
      border: "1px solid #1f2a3d"
    },
    label: { fontSize: 12, opacity: 0.85 },
    strong: { fontSize: 13, fontWeight: 800 },
    resultGrid: {
      marginTop: 12,
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
      gap: 12
    },
    resultCard: {
      borderRadius: 14,
      border: "1px solid #243045",
      background: "#0b0f17",
      padding: 10
    },
    resultImg: {
      width: "100%",
      height: 150,
      objectFit: "contain",
      borderRadius: 12,
      background: "#000",
      border: "1px solid #1f2a3d"
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.titleWrap}>
            <h1 style={styles.title}>ObjectLens</h1>
            <p style={styles.subtitle}>
              Upload an image → detect objects → select one crop → retrieve
              similar objects (Top-K)
            </p>
          </div>

          <div style={styles.pill}>
            <span style={{ opacity: 0.8 }}>API</span>
            <span style={{ fontWeight: 800 }}>{API_BASE}</span>
          </div>
        </div>

        {/* Layout */}
        <div style={styles.grid}>
          {/* Left: Upload + image */}
          <div style={styles.card}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline"
              }}
            >
              <h2 style={styles.cardTitle}>1) Upload & Detect</h2>
              <div style={{ fontSize: 12, opacity: 0.75 }}>
                {loading ? "Working..." : "Idle"}
              </div>
            </div>

            <div style={{ ...styles.row, marginTop: 12 }}>
              <input
                type="file"
                accept="image/*"
                onChange={onPickFile}
                style={styles.input}
              />

              <button
                onClick={runDetect}
                disabled={!file || loading}
                style={{
                  ...styles.btn("primary"),
                  ...(!file || loading ? styles.btnDisabled : null)
                }}
              >
                Detect
              </button>

              <button
                onClick={resetAll}
                disabled={loading}
                style={{
                  ...styles.btn("ghost"),
                  ...(loading ? styles.btnDisabled : null)
                }}
              >
                Reset
              </button>

              <div
                style={{
                  marginLeft: "auto",
                  display: "flex",
                  alignItems: "center",
                  gap: 8
                }}
              >
                <span style={{ fontSize: 12, opacity: 0.8 }}>Top-K</span>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={topK}
                  onChange={(e) => setTopK(e.target.value)}
                  style={{ ...styles.input, width: 90 }}
                />
              </div>
            </div>

            <div style={styles.imgFrame}>
              {imageUrl ? (
                <img src={imageUrl} alt="uploaded" style={styles.mainImg} />
              ) : (
                <div style={{ padding: 18, opacity: 0.75 }}>
                  Pick an image to start.
                </div>
              )}
            </div>

            <div style={styles.status}>
              <b>Status:</b> {status || "Ready."}
            </div>
          </div>

          {/* Right: Selection + Search */}
          <div style={styles.card}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline"
              }}
            >
              <h2 style={styles.cardTitle}>2) Select object & Search</h2>
              <button
                onClick={runTopK}
                disabled={!selected || loading}
                style={{
                  ...styles.btn("primary"),
                  ...(!selected || loading ? styles.btnDisabled : null)
                }}
              >
                Search Top-K
              </button>
            </div>

            {/* Selected preview */}
            <div style={styles.sectionGap}>
              <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                <div style={{ flex: 1 }}>
                  <div style={styles.label}>Selected object</div>
                  {selectedInfo ? (
                    <div style={{ marginTop: 4 }}>
                      <div style={styles.strong}>
                        {selectedInfo.class_name}{" "}
                        <span style={{ opacity: 0.7 }}>
                          · id={selectedInfo.id}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, opacity: 0.8, marginTop: 2 }}>
                        conf: {Number(selectedInfo.confidence).toFixed(3)} ·
                        bbox: x={selectedInfo.bbox.x}, y={selectedInfo.bbox.y},
                        w={selectedInfo.bbox.w}, h={selectedInfo.bbox.h}
                      </div>
                    </div>
                  ) : (
                    <div style={{ marginTop: 6, fontSize: 12, opacity: 0.75 }}>
                      No crop selected yet.
                    </div>
                  )}
                </div>

                <div
                  style={{
                    width: 160,
                    height: 120,
                    borderRadius: 14,
                    border: "1px solid #243045",
                    background: "#050812",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden"
                  }}
                >
                  {selected?.previewUrl ? (
                    <img
                      src={selected.previewUrl}
                      alt="selected-crop"
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        background: "#000"
                      }}
                    />
                  ) : (
                    <div style={{ fontSize: 12, opacity: 0.6 }}>
                      No selection
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Crops grid */}
            <div style={styles.sectionGap}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline"
                }}
              >
                <div style={styles.label}>Detected crops</div>
                <div style={{ fontSize: 12, opacity: 0.75 }}>
                  {crops.length ? `${crops.length} item(s)` : "—"}
                </div>
              </div>

              {!crops.length ? (
                <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>
                  Run detection to see crops.
                </div>
              ) : (
                <div style={styles.cropGrid}>
                  {crops.map((c, idx) => {
                    const isSel = idx === selectedIndex
                    return (
                      <button
                        key={c.det.id}
                        onClick={() => setSelectedIndex(idx)}
                        style={styles.cropBtn(isSel)}
                        title={`Select ${c.det.class_name} (id=${c.det.id})`}
                      >
                        <img
                          src={c.previewUrl}
                          alt={`crop-${c.det.id}`}
                          style={styles.cropImg}
                        />
                        <div style={{ marginTop: 8, fontSize: 12 }}>
                          <div style={{ fontWeight: 800 }}>
                            {c.det.class_name}
                          </div>
                          <div style={{ opacity: 0.75 }}>
                            conf: {Number(c.det.confidence).toFixed(3)}
                          </div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Results */}
        {topkResult?.ok ? (
          <div style={{ ...styles.card, marginTop: 16 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline"
              }}
            >
              <h2 style={styles.cardTitle}>Top-K results</h2>
              <div style={{ fontSize: 12, opacity: 0.75 }}>
                {topkResult?.best_images?.length || 0} returned
              </div>
            </div>

            <div style={styles.resultGrid}>
              {(topkResult.best_images || []).map((r, i) => {
                const src = r.image_url?.startsWith("http")
                  ? r.image_url
                  : `${API_BASE}${r.image_url || ""}`

                return (
                  <div key={`${r.image_path}-${i}`} style={styles.resultCard}>
                    <div
                      style={{ fontSize: 12, opacity: 0.8, marginBottom: 6 }}
                    >
                      #{i + 1} · score: {Number(r.score).toFixed(4)}
                    </div>

                    <img
                      src={src}
                      alt={r.image_path}
                      style={styles.resultImg}
                      onError={(e) => {
                        e.currentTarget.style.opacity = "0.25"
                      }}
                    />

                    <div style={{ marginTop: 8, fontSize: 12 }}>
                      <div style={{ fontWeight: 900 }}>{r.best_class_name}</div>
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
                  </div>
                )
              })}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
