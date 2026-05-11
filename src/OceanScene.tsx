import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { Water } from 'three/examples/jsm/objects/Water.js'
import { Sky } from 'three/examples/jsm/objects/Sky.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'

// ── Scene constants ────────────────────────────────────────────────────────────
const SKY    = { elevation: 30.3, azimuth: 7.4, exposure: 0.1764 }
const WATER  = { distortionScale: 3.7, size: 0.4 }
const BLOOM  = { strength: 0.1, radius: 0, threshold: 0 }
const CLOUDS = { coverage: 0.41, density: 0.52, elevation: 0.72 }

const CAM_START  = new THREE.Vector3(10, 18, 55)
const CAM_END    = new THREE.Vector3(0, -290, 20)
const LOOK_START = new THREE.Vector3(48, 8, 0)
const LOOK_END   = new THREE.Vector3(0, -305, 0)

// ── Cloud shader ───────────────────────────────────────────────────────────────
const CLOUD_VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`

const CLOUD_FRAG = /* glsl */`
uniform float time;
uniform float coverage;
uniform float density;
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<6;i++){v+=a*noise(p);p*=2.;a*=.5;}return v;}
void main(){
  vec2 uv = vUv-.5;
  float fade = 1.-smoothstep(.55,1.,length(uv)*2.);
  vec2 p = uv*5.+vec2(time*.005,time*.002);
  float c = smoothstep(1.-coverage,1.,fbm(p))*density*fade;
  gl_FragColor = vec4(1.,1.,1.,c);
}`

// ── Chromatic aberration pass ──────────────────────────────────────────────────
const ChromaShader = {
  uniforms: { tDiffuse: { value: null as THREE.Texture | null }, amount: { value: 0.0 } },
  vertexShader: `varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float amount;
    varying vec2 vUv;
    void main(){
      float r = texture2D(tDiffuse, vUv+vec2(amount,0.)).r;
      float g = texture2D(tDiffuse, vUv).g;
      float b = texture2D(tDiffuse, vUv-vec2(amount,0.)).b;
      gl_FragColor = vec4(r,g,b,texture2D(tDiffuse,vUv).a);
    }`,
}

// ── Per-fish agent data ────────────────────────────────────────────────────────
interface FishAgent {
  mesh: THREE.Object3D
  bx: number; by: number; bz: number
  sx: number; sz: number
  amp: number; phase: number
}

// ── Per-plant sway data ────────────────────────────────────────────────────────
interface SwayAgent {
  mesh: THREE.Object3D
  phase: number; freq: number; baseZ: number
}

const LERP = 0.05   // camera smooth-inertia factor per frame (~60 fps target)

// ─────────────────────────────────────────────────────────────────────────────

export default function OceanScene() {
  const mountRef       = useRef<HTMLDivElement>(null)
  const scrollElRef    = useRef<HTMLDivElement>(null)
  const scrollRef      = useRef(0)
  const showRef        = useRef(false)
  const [showLanding, setShowLanding] = useState(false)

  useEffect(() => {
    const container = mountRef.current
    const scrollEl  = scrollElRef.current
    if (!container || !scrollEl) return

    // ── Renderer ───────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ...({ outputBufferType: THREE.HalfFloatType } as any),
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.toneMapping    = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = SKY.exposure
    container.appendChild(renderer.domElement)
    // Canvas must not capture pointer events — scroll must reach the window
    renderer.domElement.style.pointerEvents = 'none'

    // ── Scene & Camera ─────────────────────────────────────────────────────────
    const scene  = new THREE.Scene()
    scene.fog    = new THREE.Fog(0xa8c8dc, 2500, 9000)
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 20000)
    camera.position.copy(CAM_START)

    // Lerp-tracked positions (live camera smoothly follows these)
    const camCur  = CAM_START.clone()
    const lookCur = LOOK_START.clone()
    const camTgt  = CAM_START.clone()
    const lookTgt = LOOK_START.clone()

    // ── Sun ────────────────────────────────────────────────────────────────────
    const sun   = new THREE.Vector3()
    const phi   = THREE.MathUtils.degToRad(90 - SKY.elevation)
    const theta = THREE.MathUtils.degToRad(SKY.azimuth)
    sun.setFromSphericalCoords(1, phi, theta)

    // ── Lighting ───────────────────────────────────────────────────────────────
    const sunLight = new THREE.DirectionalLight(0xffeedd, 2.0)
    sunLight.position.copy(sun).multiplyScalar(500)
    scene.add(sunLight)

    const ambLight = new THREE.AmbientLight(0x88aacc, 0.5)
    scene.add(ambLight)

    // Underwater caustic spot — starts at intensity 0
    const caustic = new THREE.SpotLight(0x3399ff, 0, 400, Math.PI / 5, 0.6)
    caustic.position.set(20, 50, 20)
    caustic.target.position.set(0, -300, 0)
    scene.add(caustic)
    scene.add(caustic.target)

    // ── Sky ────────────────────────────────────────────────────────────────────
    const sky = new Sky()
    sky.scale.setScalar(10000)
    const su = sky.material.uniforms
    su['turbidity'].value       = 10
    su['rayleigh'].value        = 2
    su['mieCoefficient'].value  = 0.005
    su['mieDirectionalG'].value = 0.8
    su['sunPosition'].value.copy(sun)

    const pmrem    = new THREE.PMREMGenerator(renderer)
    const sceneEnv = new THREE.Scene()
    sceneEnv.add(sky)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const envRT: THREE.WebGLRenderTarget = pmrem.fromScene(sceneEnv as any)
    scene.add(sky)
    scene.environment = envRT.texture
    pmrem.dispose()

    // ── Water ──────────────────────────────────────────────────────────────────
    const water = new Water(new THREE.PlaneGeometry(10000, 10000), {
      textureWidth: 512, textureHeight: 512,
      waterNormals: new THREE.TextureLoader().load('/textures/waternormals.jpg', tex => {
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping
      }),
      sunDirection: new THREE.Vector3(),
      sunColor: 0xffffff, waterColor: 0x001e0f,
      distortionScale: WATER.distortionScale, fog: true,
    })
    water.rotation.x = -Math.PI / 2
    water.material.uniforms['size'].value = WATER.size
    water.material.uniforms['sunDirection'].value.copy(sun).normalize()
    scene.add(water)

    // ── Clouds ─────────────────────────────────────────────────────────────────
    const cloudMat = new THREE.ShaderMaterial({
      uniforms: { time: { value: 0 }, coverage: { value: CLOUDS.coverage }, density: { value: CLOUDS.density } },
      vertexShader: CLOUD_VERT, fragmentShader: CLOUD_FRAG,
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
    })
    const cloudMesh = new THREE.Mesh(new THREE.CircleGeometry(7000, 64), cloudMat)
    cloudMesh.rotation.x = -Math.PI / 2
    cloudMesh.position.y = CLOUDS.elevation * 1000
    scene.add(cloudMesh)

    // ── Ocean floor plane ──────────────────────────────────────────────────────
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x00060f, roughness: 1 })
    const floor    = new THREE.Mesh(new THREE.PlaneGeometry(10000, 10000), floorMat)
    floor.rotation.x = -Math.PI / 2
    floor.position.y = -305
    scene.add(floor)

    // ── Post-processing ────────────────────────────────────────────────────────
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    composer.addPass(new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      BLOOM.strength, BLOOM.radius, BLOOM.threshold,
    ))
    const chromaPass = new ShaderPass(ChromaShader)
    composer.addPass(chromaPass)
    composer.addPass(new OutputPass())

    // ── Model loading ──────────────────────────────────────────────────────────
    let ship: THREE.Object3D | null = null
    const fishAgents: FishAgent[]   = []
    const swayAgents: SwayAgent[]   = []
    const loader = new GLTFLoader()

    loader.load('/models/Sail Ship.glb', gltf => {
      ship = gltf.scene
      ship.scale.set(11, 11, 11)
      ship.position.set(50, 0, 0)
      scene.add(ship)
    }, undefined, e => console.warn('Ship:', e))

    // Fish at mid-depth — multiple clones per GLB
    const fishDefs = [
      { path: '/models/Fish.glb',      count: 8, scale: 0.9 },
      { path: '/models/Fish (1).glb',  count: 8, scale: 0.8 },
      { path: '/models/Fish (2).glb',  count: 8, scale: 0.85 },
      { path: '/models/Koi.glb',       count: 6, scale: 1.0 },
      { path: '/models/Shark.glb',     count: 2, scale: 4.0 },
      { path: '/models/Crayfish.glb',  count: 4, scale: 0.7 },
    ]
    fishDefs.forEach(({ path, count, scale }) => {
      loader.load(path, gltf => {
        for (let i = 0; i < count; i++) {
          const mesh = gltf.scene.clone()
          const s    = scale * (0.7 + Math.random() * 0.6)
          mesh.scale.setScalar(s)
          const bx = (Math.random() - 0.5) * 300
          const by = -50 - Math.random() * 140
          const bz = (Math.random() - 0.5) * 300
          mesh.position.set(bx, by, bz)
          scene.add(mesh)
          fishAgents.push({
            mesh, bx, by, bz,
            sx: 0.3 + Math.random() * 0.4,
            sz: 0.2 + Math.random() * 0.35,
            amp:   20 + Math.random() * 50,
            phase: Math.random() * Math.PI * 2,
          })
        }
      }, undefined, e => console.warn(path, e))
    })

    // Bottom environment — scattered across ocean floor
    const bottomDefs = [
      { path: '/models/underwater_enviro_coral.glb', count: 14, scale: 3.5, yOff: 0 },
      { path: '/models/Seaweed.glb',                 count: 20, scale: 2.5, yOff: 0 },
      { path: '/models/kelp.glb',                    count: 12, scale: 5.0, yOff: 0 },
      { path: '/models/Anemone.glb',                 count: 12, scale: 1.8, yOff: 0 },
      { path: '/models/Sea anemone.glb',             count: 8,  scale: 1.5, yOff: 0 },
      { path: '/models/Seashell.glb',                count: 12, scale: 1.2, yOff: 0 },
      { path: '/models/seashell (1).glb',            count: 10, scale: 1.0, yOff: 0 },
      { path: '/models/Starfish.glb',                count: 8,  scale: 1.5, yOff: 0 },
      { path: '/models/Crab.glb',                    count: 6,  scale: 1.5, yOff: 0 },
    ]
    bottomDefs.forEach(({ path, count, scale, yOff }) => {
      loader.load(path, gltf => {
        for (let i = 0; i < count; i++) {
          const mesh = gltf.scene.clone()
          const s    = scale * (0.7 + Math.random() * 0.6)
          mesh.scale.setScalar(s)
          const x = (Math.random() - 0.5) * 500
          const z = (Math.random() - 0.5) * 500
          mesh.position.set(x, -303 + yOff, z)
          mesh.rotation.y = Math.random() * Math.PI * 2
          scene.add(mesh)
          swayAgents.push({
            mesh,
            baseZ: mesh.rotation.z,
            phase: Math.random() * Math.PI * 2,
            freq:  0.15 + Math.random() * 0.3,
          })
        }
      }, undefined, e => console.warn(path, e))
    })

    // ── Scroll listener — reads from dedicated scroll container, not window ────
    const onScroll = () => {
      const max = scrollEl.scrollHeight - scrollEl.clientHeight
      scrollRef.current = max > 0 ? Math.min(scrollEl.scrollTop / max, 1) : 0
    }
    scrollEl.addEventListener('scroll', onScroll, { passive: true })

    // ── Pre-allocated temporaries (avoids per-frame GC) ───────────────────────
    const skyBlue  = new THREE.Color(0xa8c8dc)
    const deepBlue = new THREE.Color(0x00060f)
    const fogTemp  = new THREE.Color()
    const ambSurf  = new THREE.Color(0x88aacc)
    const ambDeep  = new THREE.Color(0x001430)

    // ── Animation loop ─────────────────────────────────────────────────────────
    let animId = 0
    const clock = new THREE.Clock()
    let chromaDecay = 0
    let prevCamY    = CAM_START.y

    const animate = () => {
      animId = requestAnimationFrame(animate)
      const t  = clock.getElapsedTime()
      const p  = scrollRef.current  // scroll progress 0 → 1

      water.material.uniforms['time'].value += 1 / 60
      cloudMat.uniforms['time'].value = t

      // ── Camera target from scroll progress ─────────────────────────────────
      camTgt.lerpVectors(CAM_START, CAM_END, p)
      lookTgt.lerpVectors(LOOK_START, LOOK_END, p)
      // Landing tilt: look slightly upward at 95–100%
      if (p > 0.95) {
        const f = (p - 0.95) / 0.05
        lookTgt.y += f * 40
        // Decelerate: apply easing by pulling target less far down
        camTgt.y = THREE.MathUtils.lerp(camTgt.y, camTgt.y + 20, f * 0.5)
      }

      camCur.lerp(camTgt, LERP)
      lookCur.lerp(lookTgt, LERP)
      camera.position.copy(camCur)
      camera.lookAt(lookCur)

      // ── Depth factor (0 = surface, 1 = floor) ─────────────────────────────
      const depth = THREE.MathUtils.clamp(-camCur.y / 295, 0, 1)

      // ── Chromatic aberration: spike when crossing y=0 ────────────────────
      const crossedSurface = prevCamY >= 0 && camCur.y < 0
      if (crossedSurface) chromaDecay = 1.0
      chromaDecay = Math.max(0, chromaDecay - 0.02)
      prevCamY    = camCur.y
      chromaPass.uniforms['amount'].value = chromaDecay * 0.012

      // ── Fog + background colour ─────────────────────────────────────────────
      fogTemp.lerpColors(skyBlue, deepBlue, depth)
      if (scene.fog instanceof THREE.Fog) {
        scene.fog.color.copy(fogTemp)
        scene.fog.near = THREE.MathUtils.lerp(2500,  30, depth)
        scene.fog.far  = THREE.MathUtils.lerp(9000, 150, depth)
      }
      renderer.setClearColor(fogTemp)

      // ── Lighting ────────────────────────────────────────────────────────────
      sunLight.intensity    = THREE.MathUtils.lerp(2.0, 0.0, depth)
      ambLight.color.lerpColors(ambSurf, ambDeep, depth)
      ambLight.intensity    = THREE.MathUtils.lerp(0.5, 1.4, depth)
      caustic.intensity     = THREE.MathUtils.lerp(0.0, 10,  depth)
      // Caustic slow drift to simulate ray movement
      caustic.position.x   = Math.sin(t * 0.2) * 30
      caustic.position.z   = Math.cos(t * 0.15) * 30

      // ── Surface asset culling (avoid rendering when deep) ──────────────────
      const surfVis = depth < 0.25
      sky.visible       = surfVis
      cloudMesh.visible = surfVis
      water.visible     = depth < 0.45

      if (ship) {
        ship.visible      = depth < 0.2
        // Buoyancy: sin on Y position and subtle rotations
        ship.position.y   = -1 + Math.sin(t) * 0.25
        ship.rotation.z   = Math.sin(t * 0.8) * 0.02
        ship.rotation.x   = Math.cos(t * 0.6) * 0.015
      }

      // ── Fish sine-wave pathing ─────────────────────────────────────────────
      fishAgents.forEach(f => {
        f.mesh.position.x = f.bx + Math.sin(t * f.sx + f.phase) * f.amp
        f.mesh.position.y = f.by + Math.sin(t * 0.4  + f.phase) * 4
        f.mesh.position.z = f.bz + Math.cos(t * f.sz + f.phase) * f.amp
        // Orient toward velocity direction
        const vx = Math.cos(t * f.sx + f.phase) * f.sx * f.amp
        const vz = -Math.sin(t * f.sz + f.phase) * f.sz * f.amp
        f.mesh.rotation.y = Math.atan2(vx, vz)
      })

      // ── Coral / plant ambient sway ─────────────────────────────────────────
      swayAgents.forEach(s => {
        s.mesh.rotation.z = s.baseZ + Math.sin(t * s.freq + s.phase) * 0.06
        s.mesh.rotation.x =          Math.sin(t * s.freq * 0.7 + s.phase + 1) * 0.03
      })

      // ── Landing UI visibility ──────────────────────────────────────────────
      const shouldShow = p >= 0.88
      if (shouldShow !== showRef.current) {
        showRef.current = shouldShow
        setShowLanding(shouldShow)
      }

      composer.render()
    }

    animate()

    // ── Resize handler ─────────────────────────────────────────────────────────
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
    }
    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      scrollEl.removeEventListener('scroll', onScroll)
      cancelAnimationFrame(animId)
      composer.dispose()
      envRT.dispose()
      renderer.dispose()
      try {
        if (renderer.domElement.parentElement === container)
          container.removeChild(renderer.domElement)
      } catch { /* ignore */ }
    }
  }, [])

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      {/* 3D canvas — behind everything, no pointer events */}
      <div
        ref={mountRef}
        style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}
      />

      {/*
        Dedicated scroll container — sits above the canvas, owns the native scroll.
        Transparent background so the 3D scene shows through.
        The scrollbar appears on the right edge; styled below.
      */}
      <div
        ref={scrollElRef}
        className="ocean-scroll"
        style={{
          position: 'fixed', inset: 0,
          zIndex: 5,
          overflowY: 'scroll',
          overflowX: 'hidden',
          background: 'transparent',
        }}
      >
        {/* 500 vh spacer — pointer-events none so clicks pass through to canvas */}
        <div style={{ height: '500vh', width: '100%', pointerEvents: 'none' }} />
      </div>

      {/* Landing overlay — above scroll container */}
      {showLanding && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 20,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          pointerEvents: 'none',
          color: '#d8f0ff', textAlign: 'center',
          padding: '0 2rem',
        }}>
          <h1 style={{
            fontSize: 'clamp(2.2rem, 6vw, 5rem)',
            fontWeight: 700,
            letterSpacing: '-0.04em',
            margin: 0,
            textShadow: '0 0 50px rgba(50,140,255,0.9), 0 2px 4px rgba(0,0,0,0.6)',
            animation: 'bahaarFadeUp 1s cubic-bezier(.16,1,.3,1) both',
          }}>
            Bahaar Ocean
          </h1>
          <p style={{
            fontSize: 'clamp(1rem, 2.5vw, 1.4rem)',
            opacity: 0.75,
            marginTop: '1rem',
            maxWidth: 560,
            lineHeight: 1.6,
            textShadow: '0 1px 8px rgba(0,0,0,0.8)',
            animation: 'bahaarFadeUp 1s cubic-bezier(.16,1,.3,1) 0.15s both',
          }}>
            Exploring the depths — where light fades and life thrives.
          </p>
          <div style={{
            display: 'flex', gap: '0.75rem', marginTop: '2rem',
            animation: 'bahaarFadeUp 1s cubic-bezier(.16,1,.3,1) 0.3s both',
          }}>
            {['Vision', 'Our Team', 'Contact'].map(label => (
              <button key={label} style={{
                padding: '0.6rem 1.6rem',
                borderRadius: 9999,
                border: '1px solid rgba(80,170,255,0.35)',
                background: 'rgba(0,15,50,0.55)',
                color: '#a8d8ff',
                fontSize: '0.95rem',
                cursor: 'pointer',
                backdropFilter: 'blur(10px)',
                pointerEvents: 'auto',
                transition: 'background 0.2s, border-color 0.2s',
              }}
                onMouseEnter={e => {
                  const el = e.currentTarget as HTMLButtonElement
                  el.style.background = 'rgba(30,80,160,0.7)'
                  el.style.borderColor = 'rgba(100,190,255,0.6)'
                }}
                onMouseLeave={e => {
                  const el = e.currentTarget as HTMLButtonElement
                  el.style.background = 'rgba(0,15,50,0.55)'
                  el.style.borderColor = 'rgba(80,170,255,0.35)'
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @keyframes bahaarFadeUp {
          from { opacity: 0; transform: translateY(24px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        /* Custom scrollbar for the ocean scroll container */
        .ocean-scroll::-webkit-scrollbar {
          width: 6px;
        }
        .ocean-scroll::-webkit-scrollbar-track {
          background: rgba(0, 10, 30, 0.25);
          border-radius: 3px;
        }
        .ocean-scroll::-webkit-scrollbar-thumb {
          background: rgba(80, 170, 255, 0.55);
          border-radius: 3px;
          transition: background 0.2s;
        }
        .ocean-scroll::-webkit-scrollbar-thumb:hover {
          background: rgba(120, 200, 255, 0.85);
        }
        /* Firefox */
        .ocean-scroll {
          scrollbar-width: thin;
          scrollbar-color: rgba(80,170,255,0.55) rgba(0,10,30,0.25);
        }
      `}</style>
    </div>
  )
}
