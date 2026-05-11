import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { Water } from 'three/examples/jsm/objects/Water.js'
import { Sky } from 'three/examples/jsm/objects/Sky.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'

const SKY    = { elevation: 30.3, azimuth: 7.4, exposure: 0.1764 }
const WATER  = { distortionScale: 3.7, size: 0.4 }
const BLOOM  = { strength: 0.1, radius: 0, threshold: 0 }
const CLOUDS = { coverage: 0.41, density: 0.52, elevation: 0.72 }

// ── Cloud shader ──────────────────────────────────────────────────────────────

const CLOUD_VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const CLOUD_FRAG = /* glsl */`
uniform float time;
uniform float coverage;
uniform float density;
varying vec2 vUv;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 6; i++) { v += a * noise(p); p *= 2.0; a *= 0.5; }
  return v;
}

void main() {
  vec2 uv   = vUv - 0.5;
  float dist = length(uv) * 2.0;
  float fade = 1.0 - smoothstep(0.55, 1.0, dist);

  vec2  p     = uv * 5.0 + vec2(time * 0.005, time * 0.002);
  float cloud = fbm(p);
  cloud = smoothstep(1.0 - coverage, 1.0, cloud) * density * fade;

  gl_FragColor = vec4(1.0, 1.0, 1.0, cloud);
}
`

// ─────────────────────────────────────────────────────────────────────────────

export default function OceanScene() {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    // ── Renderer ─────────────────────────────────────────────────────────────
    // HalfFloatType is required so the HDR sky values are not clamped — this
    // is what makes the PMREM environment bake produce correct colours.
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ...({ outputBufferType: THREE.HalfFloatType } as any),
    })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = SKY.exposure
    container.appendChild(renderer.domElement)

    // ── Scene & Camera ───────────────────────────────────────────────────────
    const scene  = new THREE.Scene()
    scene.fog    = new THREE.Fog(0xa8c8dc, 2500, 9000)
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 1, 20000)
    camera.position.set(30, 30, 100)

    // ── Orbit Controls ───────────────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.maxPolarAngle = Math.PI * 0.495
    controls.target.set(0, 10, 0)
    controls.minDistance = 20
    controls.maxDistance = 500
    controls.update()

    // ── Sun position (elevation + azimuth → spherical coords) ─────────────
    const sun   = new THREE.Vector3()
    const phi   = THREE.MathUtils.degToRad(90 - SKY.elevation)
    const theta = THREE.MathUtils.degToRad(SKY.azimuth)
    sun.setFromSphericalCoords(1, phi, theta)

    // ── Sky ──────────────────────────────────────────────────────────────────
    const sky = new Sky()
    sky.scale.setScalar(10000)
    const skyUniforms = sky.material.uniforms
    // Exact values from the reference webgl_shaders_ocean example
    skyUniforms['turbidity'].value       = 10
    skyUniforms['rayleigh'].value        = 2
    skyUniforms['mieCoefficient'].value  = 0.005
    skyUniforms['mieDirectionalG'].value = 0.8
    skyUniforms['sunPosition'].value.copy(sun)

    // ── PMREM environment (matches the reference exactly) ────────────────────
    // Must come after renderer is created with HalfFloatType and after sky
    // uniforms are set. The generator renders the sky internally, compiling
    // the shader on first use, so the result is always correct here.
    const pmrem    = new THREE.PMREMGenerator(renderer)
    const sceneEnv = new THREE.Scene()
    sceneEnv.add(sky)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let envRT: THREE.WebGLRenderTarget = pmrem.fromScene(sceneEnv as any)
    scene.add(sky)
    scene.environment = envRT.texture
    pmrem.dispose()

    // ── Water ────────────────────────────────────────────────────────────────
    const waterGeometry = new THREE.PlaneGeometry(10000, 10000)
    const water = new Water(waterGeometry, {
      textureWidth:    512,
      textureHeight:   512,
      waterNormals:    new THREE.TextureLoader().load('/textures/waternormals.jpg', (tex) => {
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping
      }),
      sunDirection:    new THREE.Vector3(),
      sunColor:        0xffffff,
      waterColor:      0x001e0f,   // exact reference value
      distortionScale: WATER.distortionScale,
      fog:             scene.fog !== undefined,
    })
    water.rotation.x = -Math.PI / 2
    water.material.uniforms['size'].value = WATER.size
    water.material.uniforms['sunDirection'].value.copy(sun).normalize()
    scene.add(water)

    // ── Clouds ───────────────────────────────────────────────────────────────
    const cloudGeo = new THREE.CircleGeometry(7000, 64)
    const cloudMat = new THREE.ShaderMaterial({
      uniforms: {
        time:     { value: 0 },
        coverage: { value: CLOUDS.coverage },
        density:  { value: CLOUDS.density },
      },
      vertexShader:   CLOUD_VERT,
      fragmentShader: CLOUD_FRAG,
      transparent:    true,
      depthWrite:     false,
      side:           THREE.DoubleSide,
    })
    const cloudMesh = new THREE.Mesh(cloudGeo, cloudMat)
    cloudMesh.rotation.x = -Math.PI / 2
    cloudMesh.position.y = CLOUDS.elevation * 1000
    scene.add(cloudMesh)

    // ── Post-processing ──────────────────────────────────────────────────────
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    composer.addPass(new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      BLOOM.strength,
      BLOOM.radius,
      BLOOM.threshold,
    ))
    composer.addPass(new OutputPass())

    // ── Sail Ship ────────────────────────────────────────────────────────────
    let ship: THREE.Object3D | null = null
    new GLTFLoader().load(
      '/models/Sail Ship.glb',
      (gltf) => {
        ship = gltf.scene
        ship.scale.set(5, 5, 5)
        ship.position.set(0, 0, 0)
        scene.add(ship)
      },
      undefined,
      (err) => console.warn('Sail ship load error:', err),
    )

    // ── Animation loop ───────────────────────────────────────────────────────
    let animId = 0
    const clock = new THREE.Clock()

    const animate = () => {
      animId = requestAnimationFrame(animate)
      const t = clock.getElapsedTime()

      water.material.uniforms['time'].value += 1 / 60
      cloudMat.uniforms['time'].value = t

      if (ship) {
        ship.position.y = -1 + Math.sin(t) * 0.25
        ship.rotation.z =  Math.sin(t * 0.8) * 0.02
        ship.rotation.x =  Math.cos(t * 0.6) * 0.015
      }

      controls.update()
      composer.render()
    }

    animate()

    // ── Resize ───────────────────────────────────────────────────────────────
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(window.innerWidth, window.innerHeight)
      composer.setSize(window.innerWidth, window.innerHeight)
    }

    window.addEventListener('resize', onResize)

    return () => {
      window.removeEventListener('resize', onResize)
      cancelAnimationFrame(animId)
      controls.dispose()
      composer.dispose()
      envRT.dispose()
      renderer.dispose()
      try {
        if (renderer.domElement.parentElement === container) {
          container.removeChild(renderer.domElement)
        }
      } catch { /* ignore */ }
    }
  }, [])

  return (
    <div
      ref={mountRef}
      style={{ width: '100%', height: '100%', overflow: 'hidden' }}
    />
  )
}
