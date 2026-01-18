import React, { useRef, useEffect } from "react"
import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls"
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader"
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader"
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader"
import { STLLoader } from "three/examples/jsm/loaders/STLLoader"

function safeDisposeMesh(obj) {
  if (!obj) return
  obj.traverse((c) => {
    if (c.geometry) c.geometry.dispose()
    if (c.material) {
      if (Array.isArray(c.material)) c.material.forEach((m) => m.dispose())
      else c.material.dispose()
    }
  })
}

export default React.memo(function ModelViewer({
  url,
  file,
  live = true,
  clearColor = 0x0b0f17
}) {
  const mountRef = useRef(null)
  const rendererRef = useRef(null)
  const controlsRef = useRef(null)
  const sceneRef = useRef(null)
  const cameraRef = useRef(null)
  const activeObjectRef = useRef(null)
  const currentObjectUrlRef = useRef(null)
  const isAnimatingRef = useRef(false)
  const animIdRef = useRef(null)
  const ioRef = useRef(null)
  const roRef = useRef(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    // No global manager or global WebGL counter here — lightweight per-viewer lazy renderer only.

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(clearColor)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000)
    camera.position.set(0, 0, 3)
    cameraRef.current = camera

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6)
    scene.add(hemi)
    const dir = new THREE.DirectionalLight(0xffffff, 0.8)
    dir.position.set(5, 10, 7.5)
    scene.add(dir)

    let controls = null

    const createRendererIfAllowed = () => {
      if (!live) return false
      if (rendererRef.current) return true

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setPixelRatio(window.devicePixelRatio || 1)

      mount.innerHTML = ""
      mount.appendChild(renderer.domElement)
      rendererRef.current = renderer

      controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true
      controls.dampingFactor = 0.07
      controls.enablePan = true
      controlsRef.current = controls

      return true
    }

    // No global registration; viewer will start/stop itself when intersecting.

    const applySize = () => {
      const el = mountRef.current
      if (!el) return
      const w = el.clientWidth || 300
      const h = el.clientHeight || 200
      const renderer = rendererRef.current
      if (renderer) renderer.setSize(w, h)
      if (cameraRef.current) {
        cameraRef.current.aspect = w / h
        cameraRef.current.updateProjectionMatrix()
      }
    }

    const frameFit = (obj) => {
      const box = new THREE.Box3().setFromObject(obj)
      const size = box.getSize(new THREE.Vector3())
      const maxSize = Math.max(size.x, size.y, size.z)
      const fitDistance =
        maxSize / (2 * Math.tan((cameraRef.current.fov * Math.PI) / 360))
      cameraRef.current.position.set(0, 0, Math.max(fitDistance * 1.4, 1.5))
      cameraRef.current.near = Math.max(0.01, fitDistance / 100)
      cameraRef.current.far = fitDistance * 100
      cameraRef.current.updateProjectionMatrix()
      if (controlsRef.current && controlsRef.current.target)
        controlsRef.current.target.set(0, 0, 0)
      controlsRef.current && controlsRef.current.update()
    }

    const cleanupActive = () => {
      if (activeObjectRef.current) {
        try {
          scene.remove(activeObjectRef.current)
          safeDisposeMesh(activeObjectRef.current)
        } catch (e) {}
        activeObjectRef.current = null
      }
      // revoke any created object URL for uploaded files
      try {
        if (currentObjectUrlRef.current) {
          URL.revokeObjectURL(currentObjectUrlRef.current)
          currentObjectUrlRef.current = null
        }
      } catch (e) {}
    }

    const loadModel = (srcOrFile) => {
      cleanupActive()
      if (!srcOrFile) return
      let src = srcOrFile
      let ext = ""

      // If a File object was passed (upload), derive extension from its name
      if (typeof File !== "undefined" && srcOrFile instanceof File) {
        ext = (srcOrFile.name.split(".").pop() || "").toLowerCase()
        // create object URL for loaders
        try {
          src = URL.createObjectURL(srcOrFile)
          currentObjectUrlRef.current = src
        } catch (e) {
          src = null
        }
      } else {
        src = String(srcOrFile)
        ext = (src.split(".").pop() || "").toLowerCase()
      }

      const onLoadGLTF = (gltf) => {
        const root = gltf.scene || gltf
        root.scale.setScalar(1)
        const box = new THREE.Box3().setFromObject(root)
        const center = box.getCenter(new THREE.Vector3())
        root.position.x -= center.x
        root.position.y -= center.y
        root.position.z -= center.z
        scene.add(root)
        activeObjectRef.current = root
        frameFit(root)
      }

      try {
        if (ext === "glb" || ext === "gltf") {
          const loader = new GLTFLoader()
          loader.load(src, onLoadGLTF, undefined, () => {})
          return
        }
        if (ext === "obj") {
          const loader = new OBJLoader()
          loader.load(
            src,
            (obj) => onLoadGLTF(obj),
            undefined,
            () => {}
          )
          return
        }
        if (ext === "ply") {
          const loader = new PLYLoader()
          loader.load(
            src,
            (geom) => {
              geom.computeVertexNormals()
              const mat = new THREE.MeshStandardMaterial({ color: 0x8888ff })
              const mesh = new THREE.Mesh(geom, mat)
              onLoadGLTF(mesh)
            },
            undefined,
            () => {}
          )
          return
        }
        if (ext === "stl") {
          const loader = new STLLoader()
          loader.load(
            src,
            (geom) => {
              const mat = new THREE.MeshStandardMaterial({ color: 0x88eeff })
              const mesh = new THREE.Mesh(geom, mat)
              onLoadGLTF(mesh)
            },
            undefined,
            () => {}
          )
          return
        }
      } catch (err) {
        // loader failed
      }

      // fallback
      const box = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshStandardMaterial({ color: 0x6666aa })
      )
      scene.add(box)
      activeObjectRef.current = box
      frameFit(box)
    }

    const stopAnimation = () => {
      if (animIdRef.current) {
        cancelAnimationFrame(animIdRef.current)
        animIdRef.current = null
      }
      isAnimatingRef.current = false
    }

    const animateLoop = () => {
      if (isAnimatingRef.current) return
      const renderer = rendererRef.current
      if (!renderer) return
      isAnimatingRef.current = true

      const step = () => {
        const r = rendererRef.current
        if (!r) {
          isAnimatingRef.current = false
          return
        }
        try {
          controlsRef.current && controlsRef.current.update()
          r.render(scene, cameraRef.current)
        } catch (err) {
          // diagnostic
          try {
            console.error("Three.js render error:", err)
            scene.traverse((o) => {
              if (o.isMesh) {
                const m = o.material
                if (Array.isArray(m))
                  m.forEach((mm, i) =>
                    console.warn(o.name || o.id, "mat", i, mm && mm.type)
                  )
                else console.warn(o.name || o.id, "mat", m && m.type)
              }
            })
          } catch (e) {}
          isAnimatingRef.current = false
          return
        }
        animIdRef.current = requestAnimationFrame(step)
      }
      step()
    }

    const startIfVisible = (entries) => {
      const e = entries?.[0]
      if (!e) return
      if (e.isIntersecting) {
        if (createRendererIfAllowed()) {
          applySize()
          if (file) loadModel(file)
          else if (url) loadModel(url)
          animateLoop()
        }
      } else {
        stopAnimation()
      }
    }

    try {
      ioRef.current = new IntersectionObserver(startIfVisible, {
        threshold: 0.1
      })
      ioRef.current.observe(mount)
    } catch (e) {
      // fallback: create renderer and start
      // fallback: request immediate activation from manager or create directly
      const mgr = window.__OL_viewer_manager
      if (mgr) {
        try {
          mgr.requestActivate(idRef.current)
        } catch (ee) {}
      } else if (createRendererIfAllowed()) {
        applySize()
        if (file) {
          loadModel(file)
        } else if (url) {
          loadModel(url)
        }
        animateLoop()
      }
    }

    try {
      roRef.current = new ResizeObserver(applySize)
      roRef.current.observe(mount)
    } catch (e) {}

    return () => {
      // cleanup
      try {
        if (ioRef.current && mount) ioRef.current.unobserve(mount)
      } catch (e) {}
      try {
        if (roRef.current && mount) roRef.current.unobserve(mount)
      } catch (e) {}
      stopAnimation()
      cleanupActive()
      // no global manager to unregister
      try {
        if (controlsRef.current) controlsRef.current.dispose()
      } catch (e) {}
      try {
        if (rendererRef.current) {
          const gl =
            rendererRef.current.getContext && rendererRef.current.getContext()
          rendererRef.current.dispose()
          if (gl && gl.getExtension) {
            const ext = gl.getExtension("WEBGL_lose_context")
            ext && ext.loseContext && ext.loseContext()
          }
        }
      } catch (e) {}
      // revoke any last object URL
      try {
        if (currentObjectUrlRef.current) {
          URL.revokeObjectURL(currentObjectUrlRef.current)
          currentObjectUrlRef.current = null
        }
      } catch (e) {}
      try {
        mount.innerHTML = ""
      } catch (e) {}
    }
  }, [url, file, live, clearColor])

  return (
    <div
      ref={mountRef}
      style={{
        width: "100%",
        height: "100%",
        borderRadius: 8,
        overflow: "hidden"
      }}
    />
  )
})
