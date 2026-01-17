import { useState } from "react"
import { useNavigate } from "react-router-dom"
import ModelViewer from "./components/ModelViewer"
import { API_BASE } from "./api"

export default function Pr3d() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [topK, setTopK] = useState(5)

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
    title: { fontSize: 18, fontWeight: 900, margin: 0 },
    backBtn: {
      color: "#cfe0ff",
      fontWeight: 700,
      background: "transparent",
      border: "1px solid rgba(36,48,69,0.9)",
      padding: "6px 10px",
      borderRadius: 10,
      cursor: "pointer"
    }
  }

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files?.[0]
    if (uploadedFile) {
      setFile(uploadedFile)
      setResults([])
      setSelectedModel(null)
    }
  }

  const handleSearch = async () => {
    if (!file) {
      alert("Please select a 3D model file first")
      return
    }

    setLoading(true)
    try {
      const k = Math.max(1, Math.min(200, Number(topK) || 5))
      const url = `${API_BASE}/api/samples/class?limit=${k}`
      const res = await fetch(url)
      if (!res.ok) throw new Error(`samples API returned ${res.status}`)
      const data = await res.json()
      const files = data.files || []
      let mapped = files.map((f, i) => ({
        id: i + 1,
        name: `Model ${i + 1}`,
        similarity: Math.max(0.5, Math.random()),
        thumbnailUrl: null,
        sampleUrl: f.startsWith("/raw/") ? API_BASE + f : f
      }))
      // If backend returned fewer than k files, duplicate entries to fill slots for testing
      if (mapped.length < k) {
        if (mapped.length === 0) {
          const fallback = API_BASE + "/raw/3D%20Models/Amphora/Amphora_1.obj"
          mapped = Array.from({ length: k }).map((_, i) => ({
            id: i + 1,
            name: `Model ${i + 1}`,
            similarity: Math.max(0.5, Math.random()),
            thumbnailUrl: null,
            sampleUrl: fallback
          }))
        } else {
          const orig = mapped.slice()
          while (mapped.length < k) {
            const next = orig[mapped.length % orig.length]
            mapped.push({ ...next, id: mapped.length + 1 })
          }
        }
      }
      setResults(mapped)
    } catch (err) {
      console.error("search samples failed", err)
      // fall back to empty results
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResults([])
    setSelectedModel(null)
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
          ObjectLens
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
                    Uploaded Model Preview
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

            <div
              style={{
                display: "flex",
                gap: 8,
                marginTop: 12,
                alignItems: "center"
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <label style={{ fontSize: 12, opacity: 0.85 }}>Top-K</label>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  style={styles.input}
                />
              </div>
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button
                onClick={handleSearch}
                disabled={!file || loading}
                style={{
                  flex: 1,
                  padding: "10px 12px",
                  borderRadius: 10,
                  background: "#4f7cff",
                  color: "#061021",
                  fontWeight: 800,
                  border: "none",
                  cursor: "pointer"
                }}
              >
                {loading ? "Searching..." : "Search Models"}
              </button>
              <button
                onClick={handleReset}
                style={{
                  flex: 1,
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.04)",
                  background: "transparent",
                  color: "#e6eefb"
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
                ? `Results (${results.length})`
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
                    <div style={{ fontSize: 28, marginBottom: 8 }}>🔄</div>
                    <div style={{ opacity: 0.85 }}>
                      Searching for similar models...
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

              {!loading && results.length === 0 && file && (
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
                  {results.map((result, idx) => (
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
                            ? "rgba(124,58,237,0.12)"
                            : "transparent",
                        border: "1px solid rgba(255,255,255,0.03)",
                        cursor: "pointer",
                        minHeight: 320,
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between"
                      }}
                    >
                      <div
                        style={{
                          height: 220,
                          borderRadius: 6,
                          overflow: "hidden",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center"
                        }}
                      >
                        <div style={{ width: "100%", height: "100%" }}>
                          <ModelViewer url={result.sampleUrl} live={true} />
                        </div>
                      </div>
                      <div style={{ marginTop: 10, fontWeight: 800 }}>
                        {result.name}
                      </div>
                      <div style={{ marginTop: 6, opacity: 0.85 }}>
                        Similarity: {(result.similarity * 100).toFixed(1)}%
                      </div>
                      <div style={{ marginTop: 8 }}>
                        <small style={{ opacity: 0.8 }}>Test item</small>
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
