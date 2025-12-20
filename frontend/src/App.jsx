import { useMemo, useState } from "react"
import { detectObjects, searchTopK } from "./api"
import { cropToBlob } from "./utils/crop"

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

function ImgB64({ b64, mime = "image/png", alt, style }) {
  if (!b64) return null
  return (
    <img
      src={`data:${mime};base64,${b64}`}
      alt={alt || "viz"}
      style={style}
      onError={(e) => {
        e.currentTarget.style.opacity = "0.25"
      }}
    />
  )
}

function StatPill({ label, value }) {
  return (
    <div
      style={{
        display: "inline-flex",
        gap: 8,
        alignItems: "center",
        padding: "7px 10px",
        borderRadius: 999,
        border: "1px solid #243045",
        background: "rgba(15,22,38,0.65)",
        fontSize: 12
      }}
    >
      <span style={{ opacity: 0.7 }}>{label}</span>
      <span style={{ fontWeight: 900 }}>{value}</span>
    </div>
  )
}

function SectionHeader({ title, subtitle, right }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        gap: 12
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <h3 style={{ margin: 0, fontSize: 13, fontWeight: 900, opacity: 0.95 }}>
          {title}
        </h3>
        {subtitle ? (
          <div style={{ fontSize: 12, opacity: 0.7 }}>{subtitle}</div>
        ) : null}
      </div>
      {right ? <div>{right}</div> : null}
    </div>
  )
}

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

  // ✅ NEW: descriptor viz toggle
  const [showDescriptors, setShowDescriptors] = useState(true)

  const selected = useMemo(() => {
    if (selectedIndex === null) return null
    return crops[selectedIndex] || null
  }, [crops, selectedIndex])

  function revokeCropUrls(list) {
    ;(list || []).forEach((c) => {
      if (c?.previewUrl) URL.revokeObjectURL(c.previewUrl)
    })
  }

  function resetAll() {
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    revokeCropUrls(crops)

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

    if (imageUrl) URL.revokeObjectURL(imageUrl)
    revokeCropUrls(crops)

    setFile(f)
    setImageUrl(URL.createObjectURL(f))

    setDetectResult(null)
    setCrops([])
    setSelectedIndex(null)
    setTopkResult(null)
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
        revokeCropUrls(crops)
        setCrops([])
        setSelectedIndex(null)
        setTopkResult(null)
        setStatus("No objects detected.")
        return
      }

      setStatus(`Detected ${res.detections.length} object(s). Cropping...`)
      revokeCropUrls(crops)

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
    setStatus("Searching Top-K (class-filtered)...")

    try {
      const det = selected.det
      const k = Math.max(1, Math.min(200, Number(topK) || 20))

      const topk = await searchTopK({
        blob: selected.blob,
        filename: `query_${detectResult.image_id || "img"}_${det.id}.png`,
        topK: k,
        queryClass: det.class_name,
        sameClassOnly: true,
        metric: "cosine",
        includeViz: Boolean(showDescriptors) // ✅ NEW
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

  const qd = topkResult?.query_descriptors || null
  const qdImgs = qd?.images_b64 || {}
  const qdSum = qd?.summaries || {}
  const tamura = qdSum?.texture?.tamura || null

  const styles = {
    page: {
      minHeight: "100vh",
      background:
        "radial-gradient(1200px 700px at 15% -10%, rgba(79,124,255,0.25), transparent 55%), #0b0f17",
      color: "#eaeef7",
      fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial",
      padding: 24
    },
    container: { maxWidth: 1240, margin: "0 auto" },
    header: {
      display: "flex",
      alignItems: "flex-end",
      justifyContent: "space-between",
      gap: 16,
      marginBottom: 18
    },
    titleWrap: { display: "flex", flexDirection: "column", gap: 6 },
    title: { fontSize: 24, fontWeight: 900, margin: 0, letterSpacing: 0.2 },
    subtitle: { margin: 0, opacity: 0.8, fontSize: 13, lineHeight: 1.35 },
    pill: {
      display: "inline-flex",
      alignItems: "center",
      gap: 8,
      padding: "8px 12px",
      borderRadius: 999,
      border: "1px solid #243045",
      background: "rgba(15,22,38,0.75)",
      fontSize: 12,
      opacity: 0.95,
      backdropFilter: "blur(8px)"
    },
    grid: {
      display: "grid",
      gridTemplateColumns: "1.05fr 0.95fr",
      gap: 16
    },
    card: {
      border: "1px solid rgba(31,42,61,0.9)",
      background:
        "linear-gradient(180deg, rgba(15,22,38,0.85) 0%, rgba(11,15,23,0.85) 100%)",
      borderRadius: 18,
      padding: 16,
      boxShadow: "0 10px 28px rgba(0,0,0,0.35)",
      backdropFilter: "blur(8px)"
    },
    cardTitle: { margin: 0, fontSize: 14, opacity: 0.92, fontWeight: 800 },
    row: { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" },
    btn: (variant = "primary") => {
      const base = {
        borderRadius: 12,
        padding: "10px 12px",
        fontWeight: 800,
        fontSize: 13,
        border: "1px solid transparent",
        cursor: "pointer",
        transition: "transform 0.05s ease, opacity 0.15s ease",
        userSelect: "none"
      }
      if (variant === "primary")
        return {
          ...base,
          background: "linear-gradient(180deg,#4f7cff 0%, #3b66f5 100%)",
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
      background: "rgba(11,15,23,0.65)",
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
      background: "rgba(11,15,23,0.65)",
      fontSize: 13,
      opacity: 0.95
    },
    imgFrame: {
      marginTop: 12,
      borderRadius: 16,
      overflow: "hidden",
      border: "1px solid #243045",
      background: "rgba(5,8,18,0.85)"
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
    cropBtn: (isSelected) => ({
      textAlign: "left",
      borderRadius: 16,
      border: isSelected ? "1px solid #4f7cff" : "1px solid #243045",
      background: isSelected ? "rgba(79,124,255,0.10)" : "rgba(11,15,23,0.65)",
      padding: 10,
      cursor: "pointer",
      transition: "transform 0.06s ease, border-color 0.15s ease"
    }),
    cropImg: {
      width: "100%",
      height: 120,
      objectFit: "contain",
      borderRadius: 12,
      background: "#000",
      border: "1px solid #1f2a3d"
    },
    label: { fontSize: 12, opacity: 0.82 },
    strong: { fontSize: 13, fontWeight: 900 },
    resultGrid: {
      marginTop: 12,
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(210px, 1fr))",
      gap: 12
    },
    resultCard: {
      borderRadius: 16,
      border: "1px solid #243045",
      background: "rgba(11,15,23,0.65)",
      padding: 10
    },
    resultImg: {
      width: "100%",
      height: 150,
      objectFit: "contain",
      borderRadius: 12,
      background: "#000",
      border: "1px solid #1f2a3d"
    },

    // Descriptors UI
    vizGrid: {
      marginTop: 12,
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 12
    },
    vizCard: {
      borderRadius: 16,
      border: "1px solid #243045",
      background: "rgba(11,15,23,0.65)",
      padding: 12,
      overflow: "hidden"
    },
    vizImg: {
      width: "100%",
      height: 220,
      objectFit: "contain",
      borderRadius: 12,
      border: "1px solid #1f2a3d",
      background: "#000",
      display: "block"
    },
    vizSmallImg: {
      width: "100%",
      height: 180,
      objectFit: "contain",
      borderRadius: 12,
      border: "1px solid #1f2a3d",
      background: "#000",
      display: "block"
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
              similar objects (Top-K).
              <br />
              (Optional) Show meaningful descriptors: shape (Fourier +
              orientation), texture (Tamura), and color (HSV).
            </p>
          </div>

          <div style={styles.pill}>
            <span style={{ opacity: 0.8 }}>API</span>
            <span style={{ fontWeight: 900 }}>{API_BASE}</span>
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
                  gap: 10,
                  flexWrap: "wrap"
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

                <label
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                    fontSize: 12,
                    opacity: 0.9,
                    padding: "7px 10px",
                    borderRadius: 999,
                    border: "1px solid #243045",
                    background: "rgba(11,15,23,0.45)"
                  }}
                  title="If enabled, backend returns descriptor visualizations for the selected query crop."
                >
                  <input
                    type="checkbox"
                    checked={showDescriptors}
                    onChange={(e) => setShowDescriptors(e.target.checked)}
                  />
                  Show descriptors
                </label>
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
                        bbox: x=
                        {selectedInfo.bbox.x}, y={selectedInfo.bbox.y}, w=
                        {selectedInfo.bbox.w}, h=
                        {selectedInfo.bbox.h}
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
                    width: 170,
                    height: 125,
                    borderRadius: 16,
                    border: "1px solid #243045",
                    background: "rgba(5,8,18,0.8)",
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
                          <div style={{ fontWeight: 900 }}>
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

        {/* Descriptors (query object) */}
        {topkResult?.ok && showDescriptors ? (
          <div style={{ ...styles.card, marginTop: 16 }}>
            <SectionHeader
              title="Query descriptors (meaningful visualizations)"
              subtitle="Shape: Fourier + Orientation · Texture: Tamura · Color: HSV (Hue + S–V)"
              right={
                <div
                  style={{
                    display: "flex",
                    gap: 8,
                    flexWrap: "wrap",
                    justifyContent: "flex-end"
                  }}
                >
                  <StatPill
                    label="metric"
                    value={String(topkResult.metric || "cosine")}
                  />
                  <StatPill
                    label="Top-K"
                    value={String(topkResult.top_k || "")}
                  />
                  {selectedInfo?.class_name ? (
                    <StatPill
                      label="class"
                      value={String(selectedInfo.class_name)}
                    />
                  ) : null}
                </div>
              }
            />

            {qd?.error ? (
              <div style={{ marginTop: 10, fontSize: 12, opacity: 0.85 }}>
                ⚠️ Could not build descriptor visualizations: {String(qd.error)}
              </div>
            ) : null}

            <div style={styles.vizGrid}>
              {/* Crop preview */}
              <div style={styles.vizCard}>
                <SectionHeader
                  title="Object crop"
                  subtitle="Selected query object (context)"
                />
                <div style={{ marginTop: 10 }}>
                  <ImgB64
                    b64={qdImgs.crop_jpg}
                    mime="image/jpeg"
                    alt="query-crop"
                    style={styles.vizImg}
                  />
                </div>
              </div>

              {/* Tamura */}
              <div style={styles.vizCard}>
                <SectionHeader
                  title="Tamura texture"
                  subtitle="Coarseness / Contrast / Directionality"
                  right={
                    tamura ? (
                      <div
                        style={{ display: "flex", gap: 8, flexWrap: "wrap" }}
                      >
                        <StatPill
                          label="coarse"
                          value={Number(tamura.coarseness).toFixed(3)}
                        />
                        <StatPill
                          label="contrast"
                          value={Number(tamura.contrast).toFixed(3)}
                        />
                        <StatPill
                          label="dir"
                          value={Number(tamura.directionality).toFixed(3)}
                        />
                      </div>
                    ) : null
                  }
                />
                <div style={{ marginTop: 10 }}>
                  <ImgB64
                    b64={qdImgs.tamura_png}
                    alt="tamura"
                    style={styles.vizImg}
                  />
                </div>
              </div>

              {/* Fourier */}
              <div style={styles.vizCard}>
                <SectionHeader
                  title="Fourier descriptors"
                  subtitle="Shape signature (15 coefficients)"
                />
                <div style={{ marginTop: 10 }}>
                  <ImgB64
                    b64={qdImgs.fourier_png}
                    alt="fourier"
                    style={styles.vizImg}
                  />
                </div>
              </div>

              {/* Orientation histogram */}
              <div style={styles.vizCard}>
                <SectionHeader
                  title="Orientation histogram"
                  subtitle="Contour orientations (rotation-aligned)"
                />
                <div style={{ marginTop: 10 }}>
                  <ImgB64
                    b64={qdImgs.orientation_hist_png}
                    alt="orientation"
                    style={styles.vizImg}
                  />
                </div>
              </div>

              {/* Hue */}
              <div style={styles.vizCard}>
                <SectionHeader
                  title="Hue distribution"
                  subtitle="Dominant hues (HSV H marginal)"
                />
                <div style={{ marginTop: 10 }}>
                  <ImgB64
                    b64={qdImgs.hue_hist_png}
                    alt="hue"
                    style={styles.vizSmallImg}
                  />
                </div>
              </div>

              {/* SV heatmap */}
              <div style={styles.vizCard}>
                <SectionHeader
                  title="S–V distribution"
                  subtitle="Saturation vs Value (sum over H)"
                />
                <div style={{ marginTop: 10 }}>
                  <ImgB64
                    b64={qdImgs.sv_heatmap_png}
                    alt="sv"
                    style={styles.vizSmallImg}
                  />
                </div>
              </div>
            </div>
          </div>
        ) : null}

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
