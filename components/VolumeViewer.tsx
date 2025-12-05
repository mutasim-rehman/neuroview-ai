import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VolumeData, VolumeRenderStyle, ColorMap, TissuePreset, RenderQuality } from '../types';
import { apply3DGaussianBlur } from '../utils/volumeBlur';

interface VolumeViewerProps {
  volumes: VolumeData[];
  threshold: number;
  brightness: number;
  renderStyle: VolumeRenderStyle;
  colorMap: ColorMap;
  slices: { x: number, y: number, z: number }; // Normalized 0-1 slice positions
  cutPlane: number; // -1 to 1 for "Lightsaber" cutting
  preset: TissuePreset;
  renderQuality?: RenderQuality;
}

// Helper to convert RenderQuality enum to shader value
const getQualityValue = (quality: RenderQuality): number => {
  switch (quality) {
    case RenderQuality.FAST: return 0.5;
    case RenderQuality.MEDIUM: return 1.0;
    case RenderQuality.HIGH: return 2.0;
    case RenderQuality.ULTRA: return 4.0;
    default: return 2.0;
  }
};

const VolumeViewer: React.FC<VolumeViewerProps> = ({ 
  volumes, 
  threshold, 
  brightness, 
  renderStyle,
  colorMap,
  slices,
  cutPlane,
  preset,
  renderQuality = RenderQuality.HIGH
}) => {
  // Use first visible volume as primary
  const primaryVolume = volumes.find(v => v.metadata.visible) || volumes[0];
  if (!primaryVolume) return null;

  // Auto-rotate state
  const [autoRotate, setAutoRotate] = useState(true);
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  
  // Plane Refs for visual slicing
  const axialPlaneRef = useRef<THREE.Mesh | null>(null);
  const sagittalPlaneRef = useRef<THREE.Mesh | null>(null);
  const coronalPlaneRef = useRef<THREE.Mesh | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // --- Scene Setup ---
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    
    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(2.0, 1.5, 2.0); 

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 0.5;
    controls.maxDistance = 10;
    controls.autoRotate = autoRotate; 
    controls.autoRotateSpeed = 1.0;
    controlsRef.current = controls;

    // --- Texture Preparation ---
    const { header, image, min, max } = primaryVolume;
    const { dims } = header;
    const xDim = dims[1];
    const yDim = dims[2];
    const zDim = dims[3];

    const count = xDim * yDim * zDim;
    const floatData = new Float32Array(count);
    const range = max - min;
    const rawData = image as any; 

    for (let i = 0; i < count; i++) {
        floatData[i] = (rawData[i] - min) / range;
    }

    // --- Volume Box ---
    const pixDims = primaryVolume.header.pixDims;
    const spacingX = pixDims[1] || 1;
    const spacingY = pixDims[2] || 1;
    const spacingZ = pixDims[3] || 1;
    const physX = xDim * spacingX;
    const physY = yDim * spacingY;
    const physZ = zDim * spacingZ;
    const maxDim = Math.max(physX, Math.max(physY, physZ));
    const scale = new THREE.Vector3(physX / maxDim, physY / maxDim, physZ / maxDim);

    // Apply 3D Gaussian blur to reduce staircasing artifacts
    // The blur accounts for anisotropic spacing between slices
    const blurredData = apply3DGaussianBlur(
      floatData,
      xDim,
      yDim,
      zDim,
      spacingX,
      spacingY,
      spacingZ,
      1.0 // sigma - adjust this (0.5-2.0) to control blur amount
    );

    const texture = new THREE.Data3DTexture(blurredData, xDim, yDim, zDim);
    texture.format = THREE.RedFormat;
    texture.type = THREE.FloatType;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.unpackAlignment = 1;
    texture.needsUpdate = true;

    // Helpers
    const gridHelper = new THREE.GridHelper(3, 20, 0x10b981, 0x064e3b);
    gridHelper.position.y = -0.6;
    scene.add(gridHelper);

    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
    const edges = new THREE.EdgesGeometry(boxGeometry);
    const boxLine = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.15 }));
    boxLine.scale.copy(scale);
    scene.add(boxLine);

    // --- Slicing Planes (Visual Indicators) ---
    const planeMat = new THREE.MeshBasicMaterial({ 
      color: 0xffff00, 
      side: THREE.DoubleSide, 
      transparent: true, 
      opacity: 0.2, 
      depthTest: false 
    });
    
    // Axial (Z)
    const axP = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ ...planeMat, color: 0x10b981 })); // Green
    axP.rotation.x = -Math.PI / 2;
    axP.scale.set(scale.x, scale.y, 1);
    scene.add(axP);
    axialPlaneRef.current = axP;

    // Sagittal (X)
    const sagP = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ ...planeMat, color: 0xf472b6 })); // Pink
    sagP.rotation.y = -Math.PI / 2;
    sagP.scale.set(scale.z, scale.y, 1);
    scene.add(sagP);
    sagittalPlaneRef.current = sagP;

    // Coronal (Y - but usually Z in Nifti depending on view, let's map to Y box coord)
    const corP = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ ...planeMat, color: 0x3b82f6 })); // Blue
    corP.scale.set(scale.x, scale.z, 1); // Face Z
    scene.add(corP);
    coronalPlaneRef.current = corP;


    // --- Advanced Raymarching Shader ---
    const vertexShader = `
      varying vec3 vOrigin;
      varying vec3 vDirection;
      void main() {
        vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
        vDirection = position - vOrigin;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;

    const fragmentShader = `
      precision highp float;
      precision highp sampler3D;

      uniform sampler3D uVolume;
      uniform float uThreshold;
      uniform float uBrightness;
      uniform int uRenderStyle; // 0=MIP, 1=ISO, 2=VOL
      uniform int uColorMap; // 0=Gray, 1=Hot, 2=Cool, 3=Rainbow, 4=Anatomy, 5=Density
      uniform float uCutPlane; // -1.0 to 1.0, slices along X axis for demo
      uniform vec3 uLightDir; // Light direction (normalized)
      uniform float uRenderQuality; // 0.5=Fast, 1.0=Medium, 2.0=High, 4.0=Ultra
      uniform vec3 uVolumeDims; // Texture dimensions (xDim, yDim, zDim) for accurate gradient calculation

      varying vec3 vOrigin;
      varying vec3 vDirection;

      // Hash-based random for noise / dithering
      float rand(vec2 co) {
        return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
      }

      vec2 hitBox(vec3 orig, vec3 dir) {
        const vec3 boxMin = vec3(-0.5);
        const vec3 boxMax = vec3(0.5);
        vec3 invDir = 1.0 / dir;
        vec3 tmin = (boxMin - orig) * invDir;
        vec3 tmax = (boxMax - orig) * invDir;
        vec3 t1 = min(tmin, tmax);
        vec3 t2 = max(tmin, tmax);
        float tNear = max(max(t1.x, t1.y), t1.z);
        float tFar = min(min(t2.x, t2.y), t2.z);
        return vec2(tNear, tFar);
      }

      vec3 applyColormap(float t) {
        if (uColorMap == 0) return vec3(t); // Gray
        
        if (uColorMap == 1) { // Hot
          return vec3(t * 2.0, t * t * 2.0, t * t * t);
        }
        if (uColorMap == 2) { // Cool
           return vec3(t, t, 1.0);
        }
        if (uColorMap == 3) { // Rainbow
           float r = max(0.0, sin(t * 6.28));
           float g = max(0.0, sin(t * 6.28 + 2.0));
           float b = max(0.0, sin(t * 6.28 + 4.0));
           return vec3(r, g, b);
        }
        if (uColorMap == 4) { // Anatomy (Blue to Bone White)
           vec3 bone = vec3(0.95, 0.90, 0.85);
           vec3 tissue = vec3(0.8, 0.4, 0.4);
           vec3 fluid = vec3(0.0, 0.1, 0.2);
           
           if (t < 0.3) return mix(fluid, tissue, t * 3.0);
           return mix(tissue, bone, (t - 0.3) * 1.4);
        }
        if (uColorMap == 5) { // Density Heatmap (Blue=Low, Red=High, Orange=Mid)
            // Low density -> Blue
            // Mid density -> Orange/Yellow
            // High density -> Red
            vec3 low = vec3(0.0, 0.0, 1.0); // Blue
            vec3 mid = vec3(1.0, 0.6, 0.0); // Orange
            vec3 high = vec3(1.0, 0.0, 0.0); // Red
            
            if (t < 0.5) return mix(low, mid, t * 2.0);
            return mix(mid, high, (t - 0.5) * 2.0);
        }
        return vec3(t);
      }

      // Calculate gradient for normal estimation with proper linear interpolation
      vec3 calculateGradient(vec3 pos) {
        // Use texture dimensions to calculate accurate epsilon for each axis
        // This ensures proper gradient calculation accounting for anisotropic spacing
        vec3 eps = vec3(1.0) / uVolumeDims;
        
        // Sample with proper spacing - using linear interpolation from the texture
        float val = texture(uVolume, pos).r;
        float dx = texture(uVolume, pos + vec3(eps.x, 0.0, 0.0)).r - texture(uVolume, pos - vec3(eps.x, 0.0, 0.0)).r;
        float dy = texture(uVolume, pos + vec3(0.0, eps.y, 0.0)).r - texture(uVolume, pos - vec3(0.0, eps.y, 0.0)).r;
        float dz = texture(uVolume, pos + vec3(0.0, 0.0, eps.z)).r - texture(uVolume, pos - vec3(0.0, 0.0, eps.z)).r;
        
        // Normalize by the epsilon values to get proper gradient magnitude
        vec3 gradient = vec3(dx / (2.0 * eps.x), dy / (2.0 * eps.y), dz / (2.0 * eps.z));
        return normalize(gradient);
      }

      // Phong lighting model
      vec3 phongLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 color) {
        vec3 ambient = color * 0.3; // Ambient component
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = color * diff * 0.7;
        
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = vec3(1.0) * spec * 0.3;
        
        return ambient + diffuse + specular;
      }

      // Simple ambient occlusion approximation
      float ambientOcclusion(vec3 pos, vec3 normal) {
        // Lightweight AO: sample a few points along the normal direction
        float ao = 0.0;
        float sampleRadius = 0.03;
        const int samples = 3;
        for (int i = 1; i <= samples; i++) {
          float s = float(i) / float(samples + 1);
          vec3 samplePos = pos + normal * sampleRadius * s;
          if (samplePos.x >= 0.0 && samplePos.x <= 1.0 &&
              samplePos.y >= 0.0 && samplePos.y <= 1.0 &&
              samplePos.z >= 0.0 && samplePos.z <= 1.0) {
            float sampleVal = texture(uVolume, samplePos).r;
            if (sampleVal > uThreshold) {
              ao += 1.0 / float(samples);
            }
          }
        }
        return 1.0 - ao;
      }

      void main() {
        vec3 rayDir = normalize(vDirection);
        vec2 bounds = hitBox(vOrigin, rayDir);
        if (bounds.x > bounds.y) discard;
        bounds.x = max(bounds.x, 0.0);

        // Jitter starting point per-pixel to reduce banding ("stack of slices" look)
        float noise = rand(gl_FragCoord.xy);

        vec3 p = vOrigin + bounds.x * rayDir;
        vec3 inc = 1.0 / abs(rayDir);
        
        // Adaptive step sizing based on quality
        float baseSteps = 220.0 * uRenderQuality;
        float delta = min(inc.x, min(inc.y, inc.z)) / baseSteps;
        // Apply jitter so adjacent rays hit different depths inside a voxel
        float jitter = noise * delta;
        p += rayDir * jitter;
        float t = bounds.x + jitter;

        vec4 col = vec4(0.0);
        float maxVal = 0.0;
        vec3 viewDir = normalize(-rayDir);
        vec3 lightDir = normalize(uLightDir);

        for (float i = 0.0; i < baseSteps; i++) {
           if (t > bounds.y) break;
           
           // Clipping Logic (Lightsaber)
           if (p.z > uCutPlane * 0.5) {
               p += rayDir * delta;
               t += delta;
               continue;
           }

           vec3 uv = p + 0.5;
           float val = texture(uVolume, uv).r;
           
           // Apply Brightness
           val = clamp(val + uBrightness * 0.2, 0.0, 1.0);

           if (uRenderStyle == 0) { // MIP (X-Ray)
              maxVal = max(maxVal, val);
           } 
           else if (uRenderStyle == 1) { // ISO Surface
              if (val > uThreshold) {
                  vec3 normal = calculateGradient(uv);
                  vec3 baseColor = applyColormap(val);
                  
                  // Apply Phong lighting
                  vec3 litColor = phongLighting(normal, viewDir, lightDir, baseColor);
                  
                  // Apply ambient occlusion
                  float ao = ambientOcclusion(uv, normal);
                  litColor *= ao;
                  
                  // Depth darkening
                  float depthFactor = 1.0 - (t - bounds.x) * 0.3;
                  col = vec4(litColor * depthFactor, 1.0);
                  break; 
              }
           }
           else { // Volumetric (Cloud)
               float alpha = smoothstep(uThreshold, uThreshold + 0.1, val);
               if (alpha > 0.0) {
                   vec3 rgb = applyColormap(val);
                   
                   // Apply lighting for volumetric rendering
                   vec3 normal = calculateGradient(uv);
                   rgb = phongLighting(normal, viewDir, lightDir, rgb);
                   
                   // Depth-based alpha falloff
                   float depthAlpha = 1.0 - (t - bounds.x) * 0.2;
                   alpha *= depthAlpha;
                   
                   col.rgb += (1.0 - col.a) * alpha * 0.5 * rgb;
                   col.a += (1.0 - col.a) * alpha * 0.5;
               }
           }
           
           if (col.a >= 0.99) break;

           // Adaptive step size - larger steps in empty space
           float adaptiveDelta = delta;
           if (val < uThreshold * 0.5) {
               adaptiveDelta = delta * 2.0; // Skip faster through empty space
           }
           
           p += rayDir * adaptiveDelta;
           t += adaptiveDelta;
        }

        if (uRenderStyle == 0) {
            float alpha = smoothstep(uThreshold, 1.0, maxVal);
            col = vec4(vec3(maxVal), alpha);
        }

        gl_FragColor = col;
      }
    `;

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uVolume: { value: texture },
        uThreshold: { value: 0.1 },
        uBrightness: { value: 0.0 },
        uRenderStyle: { value: 2 },
        uColorMap: { value: 4 },
        uCutPlane: { value: 1.0 },
        uLightDir: { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
        uRenderQuality: { value: getQualityValue(renderQuality) },
        uVolumeDims: { value: new THREE.Vector3(xDim, yDim, zDim) }
      },
      vertexShader,
      fragmentShader,
      side: THREE.BackSide,
      transparent: true,
      depthTest: false
    });
    materialRef.current = material;

    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.copy(scale);
    scene.add(mesh);

    // --- Animation ---
    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      if (controlsRef.current) controlsRef.current.update();
      renderer.render(scene, camera);
    };
    animate();

    // Clean up
    return () => {
      cancelAnimationFrame(animationId);
      if (mountRef.current && rendererRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
      geometry.dispose();
      material.dispose();
      texture.dispose();
      renderer.dispose();
    };
  }, [primaryVolume, autoRotate]);

  // Keyboard event listener for spacebar to toggle rotation
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Only toggle if not typing in an input field
      if (event.code === 'Space' && event.target instanceof HTMLElement && event.target.tagName !== 'INPUT' && event.target.tagName !== 'TEXTAREA') {
        event.preventDefault();
        setAutoRotate(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, []);

  // Update controls when autoRotate state changes
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = autoRotate;
    }
  }, [autoRotate]);

  // Update Uniforms & Plane Positions
  useEffect(() => {
      if (materialRef.current) {
          // Adjust threshold based on preset override or manual slider
          let finalThreshold = threshold;
          let finalColorMap = colorMap;
          
          if (preset === TissuePreset.SKIN) {
            finalThreshold = 0.05; // Skin shows up early
          } else if (preset === TissuePreset.SOFT_TISSUE) {
            finalThreshold = 0.25;
          } else if (preset === TissuePreset.BONE) {
            finalThreshold = 0.55; // Bone is dense
          } else if (preset === TissuePreset.VESSELS) {
             finalThreshold = 0.35;
          }

          // If Density ColorMap is requested, override
          let mapIdx = 0;
          if (finalColorMap === ColorMap.HOT) mapIdx = 1;
          if (finalColorMap === ColorMap.COOL) mapIdx = 2;
          if (finalColorMap === ColorMap.RAINBOW) mapIdx = 3;
          if (finalColorMap === ColorMap.ANATOMY) mapIdx = 4;
          if (finalColorMap === ColorMap.DENSITY) mapIdx = 5;

          materialRef.current.uniforms.uThreshold.value = finalThreshold;
          materialRef.current.uniforms.uBrightness.value = brightness / 50.0;
          materialRef.current.uniforms.uCutPlane.value = cutPlane;
          
          let styleIdx = 2; // Vol
          if (renderStyle === VolumeRenderStyle.MIP) styleIdx = 0;
          if (renderStyle === VolumeRenderStyle.ISO) styleIdx = 1;
          materialRef.current.uniforms.uRenderStyle.value = styleIdx;

          materialRef.current.uniforms.uColorMap.value = mapIdx;
          materialRef.current.uniforms.uRenderQuality.value = getQualityValue(renderQuality);
      }

      // Update Slice Plane Positions
      if (primaryVolume) {
           const { pixDims, dims } = primaryVolume.header;
           
           const physX = dims[1] * (pixDims[1] || 1);
           const physY = dims[2] * (pixDims[2] || 1);
           const physZ = dims[3] * (pixDims[3] || 1);
           const maxDim = Math.max(physX, physY, physZ);
           
           const scaleX = physX / maxDim;
           const scaleY = physY / maxDim;
           const scaleZ = physZ / maxDim;

           if (axialPlaneRef.current) {
               axialPlaneRef.current.position.z = (slices.z - 0.5) * scaleZ;
               axialPlaneRef.current.visible = renderStyle === VolumeRenderStyle.VOL;
           }
           if (sagittalPlaneRef.current) {
               sagittalPlaneRef.current.position.x = (slices.x - 0.5) * scaleX;
               sagittalPlaneRef.current.visible = renderStyle === VolumeRenderStyle.VOL;
           }
           if (coronalPlaneRef.current) {
               coronalPlaneRef.current.position.y = (slices.y - 0.5) * scaleY;
               coronalPlaneRef.current.visible = renderStyle === VolumeRenderStyle.VOL;
           }
      }

  }, [threshold, brightness, renderStyle, colorMap, slices, primaryVolume, cutPlane, preset, renderQuality]);

  // Re-use box geometry outside effect to prevent recreation
  const geometry = new THREE.BoxGeometry(1, 1, 1);

  return (
    <div className="w-full h-full bg-gradient-to-b from-zinc-900 via-black to-zinc-900 overflow-hidden relative" ref={mountRef}>
       <div className="absolute top-4 left-4 text-xs font-mono text-emerald-400 bg-black/80 p-3 rounded-lg border border-emerald-500/30 backdrop-blur-md pointer-events-none select-none z-10 flex flex-col gap-1">
          <div className="font-bold flex items-center gap-2">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
            3D DIGITAL TWIN
          </div>
          <div className="opacity-80 flex justify-between">
              <span>Mode:</span>
              <span className="text-white">{renderStyle}</span>
          </div>
          {preset !== TissuePreset.CUSTOM && (
              <div className="opacity-80 flex justify-between text-yellow-400">
                  <span>Layer:</span>
                  <span className="font-bold">{preset}</span>
              </div>
          )}
        </div>
        
        {cutPlane < 0.95 && (
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none z-0 opacity-20">
                <div className="w-64 h-1 bg-red-500 shadow-[0_0_20px_rgba(255,0,0,0.8)] animate-pulse"></div>
            </div>
        )}
    </div>
  );
};

export default VolumeViewer;
