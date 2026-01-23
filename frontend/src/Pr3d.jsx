import { useState } from "react"
import { useNavigate } from "react-router-dom"
import ModelViewer from "./components/ModelViewer"
import { API_BASE, searchTopK_3d } from "./api"

export default function Pr3d() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [topK, setTopK] = useState(5)
  const [error, setError] = useState(null)

  // Search parameters
  const [method, setMethod] = useState("depth")
  const [metric, setMetric] = useState("cosine")
  const [showAdvanced, setShowAdvanced] = useState(false)

  const styles = {
    page: {
      minHeight: "100vh",
      background:
        "radial-gradient(1200px 700px at 15% -10%, rgba(79,124,255,0.25), transparent 55%), #0b0f17",
      color: "#eaeef7",
      padding: 24
    },
    layout: {
      maxWidth: 1200,
      margin: "0 auto",
      display: "grid",
      gridTemplateColumns: "320px 1fr",
      gap: 20
    },
    card: {
      borderRadius: 12,
      padding: 16,
      background:
        "linear-gradient(180deg, rgba(12,18,30,0.9), rgba(8,12,18,0.9))",
      border: "1px solid rgba(31,42,61,0.9)"
    },
    input: {
      borderRadius: 10,
      border: "1px solid rgba(255,255,255,0.04)",
      background: "transparent",
      color: "#e6eefb",
      padding: "8px 10px",
      width: 84
    },
    select: {
      borderRadius: 10,
      border: "1px solid rgba(255,255,255,0.04)",
      background: "rgba(8,12,18,0.9)",
      color: "#e6eefb",
      padding: "8px 10px",
      width: "100%",
      cursor: "pointer"
    },
    title: { fontSize: 18, fontWeight: 900, margin: 0 },
    backBtn: {
      color: "#cfe0ff",
      fontWeight: 700,
      background: "transparent",
      border: "1px solid rgba(36,48,69,0.9)",
      padding: "6px 10px",
      borderRadius: 10,
      cursor: "pointer"
    },
    errorBox: {
      marginTop: 12,
      padding: 12,
      borderRadius: 8,
      background: "rgba(255,80,80,0.12)",
      border: "1px solid rgba(255,80,80,0.3)",
      color: "#ff8888",
      fontSize: 13
    }
  }

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files?.[0]
    if (uploadedFile) {
      setFile(uploadedFile)
      setResults([])
      setSelectedModel(null)
      setError(null)
    }
  }

  const handleSearch = async () => {
    if (!file) {
      alert("Please select a 3D model file first")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const data = await searchTopK_3d({
        file,
        topK: Math.max(1, Math.min(100, Number(topK) || 5)),
        method,
        metric,
        aggregation: "mean",
        imageSize: 256,
        l2Normalize: false
      })

      // Map API results to display format
      const mapped = (data.results || []).map((r, i) => ({
        id: i + 1,
        rank: r.rank,
        filename: r.filename,
        className: r.class,
        distance: r.distance,
        similarity: r.similarity_score,
        // Build URL to serve the 3D model
        modelUrl: `${API_BASE}/raw/3D%20Models/${encodeURIComponent(r.class)}/${encodeURIComponent(r.filename)}`
      }))

      setResults(mapped)

      // Auto-select first result if available
      if (mapped.length > 0) {
        setSelectedModel(mapped[0].id)
      }
    } catch (err) {
      console.error("3D search failed", err)
      setError(err.message || "Search failed. Please try again.")
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResults([])
    setSelectedModel(null)
    setError(null)
  }

  return (
    <div style={styles.page}>
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          marginBottom: 18,
          display: "flex",
          alignItems: "center",
          gap: 12
        }}
      >
        <button onClick={() => navigate("/")} style={styles.backBtn}>
          ← Back to Home
        </button>
        <h1
          style={{ fontSize: 24, fontWeight: 900, color: "#eaeef7", margin: 0 }}
        >
          ObjectLens - 3D Model Retrieval
        </h1>
      </div>

      <div style={styles.layout}>
        {/* Left column: Upload */}
        <div style={{ position: "sticky", top: 24, alignSelf: "start" }}>
          <div style={styles.card}>
            <h3 style={styles.title}>Upload 3D Model</h3>

            <div style={{ marginTop: 12 }}>
              {file && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 13, opacity: 0.9, marginBottom: 8 }}>
                    Query Model Preview
                  </div>
                  <div
                    style={{
                      width: "100%",
                      height: 320,
                      borderRadius: 8,
                      overflow: "hidden"
                    }}
                  >
                    <ModelViewer file={file} live={true} />
                  </div>
                </div>
              )}
              <label style={{ display: "block", cursor: "pointer" }}>
                <div
                  style={{
                    border: "2px dashed rgba(255,255,255,0.04)",
                    borderRadius: 10,
                    padding: 18,
                    textAlign: "center"
                  }}
                >
                  <div style={{ fontSize: 28, marginBottom: 6 }}>📦</div>
                  <div style={{ fontWeight: 800 }}>
                    {file ? file.name : "Choose a 3D model file"}
                  </div>
                  <div style={{ fontSize: 12, opacity: 0.75, marginTop: 6 }}>
                    .obj, .stl, .glb, .ply
                  </div>
                </div>
                <input
                  type="file"
                  accept=".obj,.stl,.glb,.ply"
                  onChange={handleFileUpload}
                  style={{ display: "none" }}
                />
              </label>
            </div>

            {file && (
              <div
                style={{
                  marginTop: 12,
                  padding: 10,
                  borderRadius: 8,
                  background: "rgba(255,255,255,0.02)"
                }}
              >
                <div style={{ fontSize: 13 }}>
                  <strong>File:</strong> {file.name}
                </div>
                <div style={{ fontSize: 13, marginTop: 6 }}>
                  Size: {(file.size / 1024).toFixed(2)} KB
                </div>
              </div>
            )}

            {/* Search Parameters */}
            <div
              style={{
                display: "flex",
                gap: 8,
                marginTop: 12,
                alignItems: "flex-end"
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <label style={{ fontSize: 12, opacity: 0.85 }}>Top-K</label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  style={styles.input}
                />
              </div>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{
                  padding: "8px 12px",
                  borderRadius: 8,
                  border: "1px solid rgba(255,255,255,0.08)",
                  background: "transparent",
                  color: "#a0b0c5",
                  fontSize: 12,
                  cursor: "pointer"
                }}
              >
                {showAdvanced ? "Hide" : "Options"} ⚙️
              </button>
            </div>

            {/* Advanced Options */}
            {showAdvanced && (
              <div
                style={{
                  marginTop: 12,
                  padding: 12,
                  borderRadius: 8,
                  background: "rgba(255,255,255,0.02)"
                }}
              >
                <div style={{ marginBottom: 10 }}>
                  <label style={{ fontSize: 12, opacity: 0.85, display: "block", marginBottom: 4 }}>
                    Descriptor Method
                  </label>
                  <select
                    value={method}
                    onChange={(e) => setMethod(e.target.value)}
                    style={styles.select}
                  >
                    <option value="depth">Depth Buffer (recommended)</option>
                    <option value="lfd">Light Field Descriptor (LFD)</option>
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: 12, opacity: 0.85, display: "block", marginBottom: 4 }}>
                    Distance Metric
                  </label>
                  <select
                    value={metric}
                    onChange={(e) => setMetric(e.target.value)}
                    style={styles.select}
                  >
                    <option value="l2">L2 (Euclidean)</option>
                    <option value="l1">L1 (Manhattan)</option>
                    <option value="cosine">Cosine</option>
                  </select>
                </div>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div style={styles.errorBox}>
                <strong>Error:</strong> {error}
              </div>
            )}

            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button
                onClick={handleSearch}
                disabled={!file || loading}
                style={{
                  flex: 1,
                  padding: "10px 12px",
                  borderRadius: 10,
                  background: loading ? "#3a5a9f" : "#4f7cff",
                  color: "#061021",
                  fontWeight: 800,
                  border: "none",
                  cursor: loading ? "wait" : "pointer",
                  opacity: !file ? 0.5 : 1
                }}
              >
                {loading ? "Searching..." : "🔍 Search Models"}
              </button>
              <button
                onClick={handleReset}
                style={{
                  flex: 1,
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.04)",
                  background: "transparent",
                  color: "#e6eefb",
                  cursor: "pointer"
                }}
              >
                Reset
              </button>
            </div>
          </div>
        </div>

        {/* Right column: Results */}
        <div>
          <div style={styles.card}>
            <h3 style={styles.title}>
              {results.length > 0
                ? `Found ${results.length} Similar Models`
                : "Similar Models"}
            </h3>

            <div style={{ marginTop: 12 }}>
              {loading && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: 200
                  }}
                >
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 36, marginBottom: 8 }}>🔄</div>
                    <div style={{ opacity: 0.85, fontWeight: 600 }}>
                      Computing 3D descriptors and searching...
                    </div>
                    <div style={{ fontSize: 12, opacity: 0.6, marginTop: 6 }}>
                      This may take a few seconds
                    </div>
                  </div>
                </div>
              )}

              {!loading && results.length === 0 && !file && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: 200
                  }}
                >
                  <div style={{ textAlign: "center", opacity: 0.75 }}>
                    <div style={{ fontSize: 36, marginBottom: 8 }}>📦</div>
                    <div style={{ fontWeight: 800 }}>
                      Upload a 3D model to start
                    </div>
                    <div style={{ fontSize: 12, marginTop: 6, opacity: 0.7 }}>
                      Supported formats: .obj, .stl, .glb, .ply
                    </div>
                  </div>
                </div>
              )}

              {!loading && results.length === 0 && file && !error && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: 200
                  }}
                >
                  <div style={{ textAlign: "center", opacity: 0.8 }}>
                    <div style={{ fontSize: 36, marginBottom: 8 }}>🔍</div>
                    <div style={{ fontWeight: 800 }}>
                      Click "Search Models" to find similar items
                    </div>
                  </div>
                </div>
              )}

              {!loading && results.length > 0 && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, 1fr)",
                    gap: 12
                  }}
                >
                  {results.map((result) => (
                    <div
                      key={result.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => setSelectedModel(result.id)}
                      style={{
                        textAlign: "left",
                        padding: 12,
                        borderRadius: 8,
                        background:
                          selectedModel === result.id
                            ? "rgba(79,124,255,0.15)"
                            : "transparent",
                        border: selectedModel === result.id
                          ? "1px solid rgba(79,124,255,0.4)"
                          : "1px solid rgba(255,255,255,0.03)",
                        cursor: "pointer",
                        minHeight: 320,
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between",
                        transition: "all 0.2s ease"
                      }}
                    >
                      <div
                        style={{
                          height: 220,
                          borderRadius: 6,
                          overflow: "hidden",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          background: "rgba(0,0,0,0.2)"
                        }}
                      >
                        <div style={{ width: "100%", height: "100%" }}>
                          <ModelViewer url={result.modelUrl} live={true} />
                        </div>
                      </div>

                      <div style={{ marginTop: 10 }}>
                        <div style={{ fontWeight: 800, fontSize: 14 }}>
                          #{result.rank} - {result.className}
                        </div>
                        <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4, wordBreak: "break-all" }}>
                          {result.filename}
                        </div>
                      </div>

                      <div style={{ marginTop: 8 }}>
                        <div style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center"
                        }}>
                          <span style={{ fontSize: 12, opacity: 0.7 }}>Similarity:</span>
                          <span style={{
                            fontWeight: 700,
                            color: result.similarity > 0.7 ? "#7dd87d" :
                              result.similarity > 0.4 ? "#f0d060" : "#ff8888"
                          }}>
                            {(result.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginTop: 4
                        }}>
                          <span style={{ fontSize: 12, opacity: 0.7 }}>Distance:</span>
                          <span style={{ fontSize: 12, opacity: 0.9 }}>
                            {result.distance.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}