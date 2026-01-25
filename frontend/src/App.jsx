import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import Home from "./Home"
import Pr2d from "./Pr2d"
import Pr3d from "./Pr3d"

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/cbir-2d" element={<Pr2d />} />
        <Route path="/3d-search" element={<Pr3d />} />
      </Routes>
    </Router>
  )
}
