import { useNavigate } from "react-router-dom"

export default function Home() {
  const navigate = useNavigate()

  const styles = {
    page: {
      minHeight: "100vh",
      background:
        "radial-gradient(1200px 700px at 15% -10%, rgba(79,124,255,0.12), transparent 55%), #0b0f17",
      color: "#eaeef7",
      fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial",
      padding: 24
    },
    container: { maxWidth: 1240, margin: "0 auto" },
    header: {
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 6,
      marginBottom: 24
    },
    title: { fontSize: 40, fontWeight: 900, margin: 0 },
    subtitle: { margin: 0, opacity: 0.85, fontSize: 15 },
    grid: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 16,
      marginTop: 18
    },
    card: {
      border: "1px solid rgba(31,42,61,0.9)",
      background:
        "linear-gradient(180deg, rgba(15,22,38,0.85), rgba(11,15,23,0.85))",
      borderRadius: 12,
      padding: 18,
      cursor: "pointer"
    },
    cardTitle: { margin: 0, fontSize: 20, opacity: 0.95, fontWeight: 900 },
    cardBody: { color: "rgba(234,238,247,0.85)", marginTop: 8 }
  }

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.header}>
          <h1 style={styles.title}>ObjectLens</h1>
          <p style={styles.subtitle}>Content-Based Retrieval System</p>
          <p style={{ marginTop: 6, opacity: 0.7 }}>
            Search 2D Images & 3D Models
          </p>
        </div>

        <div style={styles.grid}>
          <div style={styles.card} onClick={() => navigate("/cbir-2d")}>
            <h2 style={styles.cardTitle}>CBIR 2D</h2>
            <div style={styles.cardBody}>
              <div style={{ marginTop: 6 }}>Content-Based Image Retrieval</div>
              <div style={{ marginTop: 8, opacity: 0.8, fontSize: 13 }}>
                Search for similar 2D images using deep features and descriptor
                matching.
              </div>
              <div style={{ marginTop: 12 }}>
                <button
                  style={{
                    padding: "10px 14px",
                    borderRadius: 10,
                    background: "#4f7cff",
                    color: "#061021",
                    fontWeight: 800,
                    border: "none",
                    cursor: "pointer"
                  }}
                >
                  Start Searching →
                </button>
              </div>
            </div>
          </div>

          <div style={styles.card} onClick={() => navigate("/3d-search")}>
            <h2 style={styles.cardTitle}>3D Model Search</h2>
            <div style={styles.cardBody}>
              <div style={{ marginTop: 6 }}>Recherche par le contenu 3D</div>
              <div style={{ marginTop: 8, opacity: 0.8, fontSize: 13 }}>
                Recherche par le contenu dans une base d'exemples de modèles 3D.
              </div>
              <div style={{ marginTop: 12 }}>
                <button
                  style={{
                    padding: "10px 14px",
                    borderRadius: 10,
                    background: "#7b5cff",
                    color: "#fff",
                    fontWeight: 800,
                    border: "none",
                    cursor: "pointer"
                  }}
                >
                  Explore Models →
                </button>
              </div>
            </div>
          </div>
        </div>

        <div
          style={{
            marginTop: 20,
            background: "rgba(11,15,23,0.85)",
            padding: 18,
            borderRadius: 12
          }}
        >
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 800 }}>
            Key Features
          </h3>
          <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 28 }}>⚡</div>
              <div style={{ fontWeight: 800, marginTop: 8 }}>Fast Search</div>
            </div>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 28 }}>🎨</div>
              <div style={{ fontWeight: 800, marginTop: 8 }}>
                Smart Matching
              </div>
            </div>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ fontSize: 28 }}>🔐</div>
              <div style={{ fontWeight: 800, marginTop: 8 }}>
                Reliable Results
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
