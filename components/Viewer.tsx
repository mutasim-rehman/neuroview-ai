import React, { useEffect, useRef, useState, useCallback } from 'react';
import { VolumeData, ViewType, TypedArray, ToolMode, Measurement, Annotation, ROIStats, OverlayMode } from '../types';
import { Brain, Ruler, MapPin } from 'lucide-react';
import { analyzeMedicalSlice } from '../services/geminiService';

interface ViewerProps {
  volumes: VolumeData[];
  activeView: ViewType;
  sliceIndex: number;
  onSliceChange: (index: number) => void;
  brightness: number;
  contrast: number;
  toolMode: ToolMode;
  measurements: Measurement[];
  onAddMeasurement: (m: Measurement) => void;
  annotations: Annotation[];
  onAddAnnotation: (a: Annotation) => void;
  onROIStatsUpdate: (stats: ROIStats | null) => void;
  overlayMode?: OverlayMode;
}

const Viewer: React.FC<ViewerProps> = ({ 
  volumes, 
  activeView, 
  sliceIndex,
  onSliceChange,
  brightness,
  contrast,
  toolMode,
  measurements,
  onAddMeasurement,
  annotations,
  onAddAnnotation,
  onROIStatsUpdate,
  overlayMode = OverlayMode.BLEND
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<string | null>(null);
  
  // Interaction State
  const [isDragging, setIsDragging] = useState(false);
  const [startPos, setStartPos] = useState<{x: number, y: number} | null>(null);
  const [currentPos, setCurrentPos] = useState<{x: number, y: number} | null>(null);

  // Use first volume for dimensions (all volumes should have same dimensions for overlay)
  const primaryVolume = volumes[0];
  if (!primaryVolume) return null;

  const { header, image, min: dataMin, max: dataMax } = primaryVolume;
  const dims = header.dims;
  const pixDims = header.pixDims; // [ndim, dx, dy, dz, dt]

  // Calculate current dimensions based on view
  const getSliceDims = useCallback(() => {
    switch (activeView) {
      case ViewType.AXIAL:
        // x, y
        return { 
          width: dims[1], height: dims[2], depth: dims[3],
          pixelW: pixDims[1], pixelH: pixDims[2] 
        };
      case ViewType.CORONAL:
        // x, z
        return { 
          width: dims[1], height: dims[3], depth: dims[2],
          pixelW: pixDims[1], pixelH: pixDims[3] 
        };
      case ViewType.SAGITTAL:
        // y, z
        return { 
          width: dims[2], height: dims[3], depth: dims[1],
          pixelW: pixDims[2], pixelH: pixDims[3] 
        };
      default:
        return { width: 0, height: 0, depth: 0, pixelW: 1, pixelH: 1 };
    }
  }, [activeView, dims, pixDims]);

  const { width, height, depth, pixelW, pixelH } = getSliceDims();

  // Normalize max slice index when view changes
  useEffect(() => {
    const maxIndex = depth - 1;
    if (sliceIndex > maxIndex) {
      onSliceChange(Math.floor(maxIndex / 2));
    }
  }, [activeView, depth, sliceIndex, onSliceChange]);

  // Helper to get voxel index for a given position
  const getVoxelIndex = useCallback((c: number, r: number, xDim: number, yDim: number) => {
    if (activeView === ViewType.AXIAL) {
      const x = c;
      const y = height - 1 - r;
      const z = sliceIndex;
      return x + (y * xDim) + (z * xDim * yDim);
    } else if (activeView === ViewType.CORONAL) {
      const x = c;
      const y = sliceIndex;
      const z = height - 1 - r;
      return x + (y * xDim) + (z * xDim * yDim);
    } else { // SAGITTAL
      const x = sliceIndex;
      const y = c;
      const z = height - 1 - r;
      return x + (y * xDim) + (z * xDim * yDim);
    }
  }, [activeView, sliceIndex, height]);

  // Apply window/level to a value
  const applyWindowLevel = useCallback((val: number, window: number, level: number, min: number, max: number) => {
    const range = max - min;
    if (range === 0) return 0.5;
    
    // Normalize to 0-1
    let normalized = (val - min) / range;
    
    // Apply window/level
    const windowMin = level - window / 2;
    const windowMax = level + window / 2;
    const windowRange = windowMax - windowMin;
    
    if (windowRange > 0) {
      normalized = (normalized * (max - min) + min - windowMin) / windowRange;
      normalized = Math.max(0, Math.min(1, normalized));
    } else {
      normalized = normalized * (max - min) + min >= level ? 1 : 0;
    }
    
    return normalized;
  }, []);

  // Convert normalized value to RGB based on color map
  const applyColorMap = useCallback((normalized: number, colorMap: string): [number, number, number] => {
    // Simple color map implementations
    if (colorMap === 'Grayscale') {
      return [normalized, normalized, normalized];
    } else if (colorMap === 'Hot Iron') {
      return [normalized * 2, normalized * normalized * 2, normalized * normalized * normalized];
    } else if (colorMap === 'Cool Blue') {
      return [normalized, normalized, 1.0];
    } else if (colorMap === 'Rainbow') {
      const r = Math.max(0, Math.sin(normalized * 6.28));
      const g = Math.max(0, Math.sin(normalized * 6.28 + 2.0));
      const b = Math.max(0, Math.sin(normalized * 6.28 + 4.0));
      return [r, g, b];
    } else {
      // Default grayscale
      return [normalized, normalized, normalized];
    }
  }, []);

  // Blend two colors based on overlay mode
  const blendColors = useCallback((
    base: [number, number, number], 
    overlay: [number, number, number], 
    opacity: number,
    mode: OverlayMode
  ): [number, number, number] => {
    const a = opacity;
    const b = 1 - a;
    
    switch (mode) {
      case OverlayMode.BLEND:
        return [
          base[0] * b + overlay[0] * a,
          base[1] * b + overlay[1] * a,
          base[2] * b + overlay[2] * a
        ];
      case OverlayMode.ADDITIVE:
        return [
          Math.min(1, base[0] + overlay[0] * a),
          Math.min(1, base[1] + overlay[1] * a),
          Math.min(1, base[2] + overlay[2] * a)
        ];
      case OverlayMode.MULTIPLY:
        return [
          base[0] * (1 - a) + base[0] * overlay[0] * a,
          base[1] * (1 - a) + base[1] * overlay[1] * a,
          base[2] * (1 - a) + base[2] * overlay[2] * a
        ];
      case OverlayMode.DIFFERENCE:
        return [
          Math.abs(base[0] - overlay[0]) * a + base[0] * b,
          Math.abs(base[1] - overlay[1]) * a + base[1] * b,
          Math.abs(base[2] - overlay[2]) * a + base[2] * b
        ];
      case OverlayMode.MAXIMUM:
        return [
          Math.max(base[0], overlay[0] * a + base[0] * b),
          Math.max(base[1], overlay[1] * a + base[1] * b),
          Math.max(base[2], overlay[2] * a + base[2] * b)
        ];
      case OverlayMode.MINIMUM:
        return [
          Math.min(base[0], overlay[0] * a + base[0] * b),
          Math.min(base[1], overlay[1] * a + base[1] * b),
          Math.min(base[2], overlay[2] * a + base[2] * b)
        ];
      default:
        return [
          base[0] * b + overlay[0] * a,
          base[1] * b + overlay[1] * a,
          base[2] * b + overlay[2] * a
        ];
    }
  }, []);

  const drawSlice = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || volumes.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size (logical pixels)
    canvas.width = width;
    canvas.height = height;

    const imgData = ctx.createImageData(width, height);
    const pixelData = imgData.data;
    
    const xDim = dims[1];
    const yDim = dims[2];
    
    const contrastFactor = Math.max(0, (259 * (contrast + 255)) / (255 * (259 - contrast)));
    
    // Get visible volumes sorted by opacity (render base first)
    const visibleVols = volumes.filter(v => v.metadata.visible).sort((a, b) => 
      a.metadata.opacity - b.metadata.opacity
    );

    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const pixelIndex = (r * width + c) * 4;
        let finalColor: [number, number, number] = [0, 0, 0];
        
        // Render each volume and blend
        for (let volIdx = 0; volIdx < visibleVols.length; volIdx++) {
          const volume = visibleVols[volIdx];
          const volData = volume.image as TypedArray;
          const volDims = volume.header.dims;
          const volXDim = volDims[1];
          const volYDim = volDims[2];
          
          const volIndex = getVoxelIndex(c, r, volXDim, volYDim);
          
          let val = 0;
          if (volIndex >= 0 && volIndex < volData.length) {
            val = volData[volIndex];
          }
          
          // Apply window/level
          const normalized = applyWindowLevel(val, volume.metadata.window, volume.metadata.level, volume.min, volume.max);
          
          // Apply brightness/contrast
          let adjusted = contrastFactor * (normalized - 0.5) + 0.5 + (brightness / 255);
          adjusted = Math.max(0, Math.min(1, adjusted));
          
          // Apply color map
          const rgb = applyColorMap(adjusted, volume.metadata.colorMap);
          
          // Blend with existing color
          if (volIdx === 0) {
            finalColor = rgb;
          } else {
            finalColor = blendColors(finalColor, rgb, volume.metadata.opacity, overlayMode);
          }
        }
        
        // Convert to 0-255 range
        pixelData[pixelIndex] = Math.round(finalColor[0] * 255);
        pixelData[pixelIndex + 1] = Math.round(finalColor[1] * 255);
        pixelData[pixelIndex + 2] = Math.round(finalColor[2] * 255);
        pixelData[pixelIndex + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [width, height, volumes, activeView, sliceIndex, dims, contrast, brightness, getVoxelIndex, applyWindowLevel, applyColorMap, blendColors, overlayMode]);

  useEffect(() => {
    drawSlice();
  }, [drawSlice]);

  const handleAnalyze = async () => {
      if (!canvasRef.current) return;
      setAnalyzing(true);
      setAnalysis(null);
      const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.8);
      const result = await analyzeMedicalSlice(dataUrl);
      setAnalysis(result);
      setAnalyzing(false);
  };

  // --- Interaction Handlers ---

  const getCanvasCoordinates = (e: React.MouseEvent) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const coords = getCanvasCoordinates(e);
    
    if (toolMode === ToolMode.MEASURE) {
      setIsDragging(true);
      setStartPos(coords);
      setCurrentPos(coords);
    } else if (toolMode === ToolMode.ANNOTATE) {
      const text = prompt("Enter annotation label:");
      if (text) {
        onAddAnnotation({
          id: Date.now().toString(),
          viewType: activeView,
          sliceIndex: sliceIndex,
          x: coords.x,
          y: coords.y,
          text: text
        });
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const coords = getCanvasCoordinates(e);

    // ROI Stats Extraction
    const pxX = Math.floor(coords.x);
    const pxY = Math.floor(coords.y);
    
    if (pxX >= 0 && pxX < width && pxY >= 0 && pxY < height) {
         const xDim = dims[1];
         const yDim = dims[2];
         
         let volIndex = 0;
         let realX=0, realY=0, realZ=0;

         if (activeView === ViewType.AXIAL) {
           realX = pxX;
           realY = height - 1 - pxY;
           realZ = sliceIndex;
           volIndex = realX + (realY * xDim) + (realZ * xDim * yDim);
         } else if (activeView === ViewType.CORONAL) {
           realX = pxX;
           realY = sliceIndex;
           realZ = height - 1 - pxY;
           volIndex = realX + (realY * xDim) + (realZ * xDim * yDim);
         } else if (activeView === ViewType.SAGITTAL) {
           realX = sliceIndex;
           realY = pxX;
           realZ = height - 1 - pxY;
           volIndex = realX + (realY * xDim) + (realZ * xDim * yDim);
         }
         
         const volData = image as TypedArray;
         if (volIndex >= 0 && volIndex < volData.length) {
             const val = volData[volIndex];
             // Simple density classifier based on range (normalized approx)
             const range = dataMax - dataMin;
             const norm = (val - dataMin) / range;
             let label = "Soft Tissue";
             if (norm < 0.1) label = "Air/Background";
             else if (norm < 0.3) label = "Fluid/Fat";
             else if (norm > 0.6) label = "Bone/Calcification";
             else if (norm > 0.45) label = "High Density";
             
             onROIStatsUpdate({
                 x: realX, y: realY, z: realZ,
                 value: val,
                 densityLabel: label
             });
         }
    } else {
        onROIStatsUpdate(null);
    }

    if (toolMode === ToolMode.MEASURE && isDragging) {
      setCurrentPos(coords);
    }
  };

  const handleMouseUp = () => {
    if (toolMode === ToolMode.MEASURE && isDragging && startPos && currentPos) {
      // Calculate real-world distance in mm
      const dx = (currentPos.x - startPos.x) * pixelW;
      const dy = (currentPos.y - startPos.y) * pixelH;
      const dist = Math.sqrt(dx*dx + dy*dy);

      if (dist > 0.5) { // Only add if line has length
        onAddMeasurement({
          id: Date.now().toString(),
          viewType: activeView,
          sliceIndex: sliceIndex,
          startX: startPos.x,
          startY: startPos.y,
          endX: currentPos.x,
          endY: currentPos.y,
          lengthMm: dist
        });
      }
    }
    setIsDragging(false);
    setStartPos(null);
    setCurrentPos(null);
  };
  
  const handleMouseLeave = () => {
      handleMouseUp();
      onROIStatsUpdate(null);
  }

  return (
    <div className="flex flex-col h-full bg-zinc-900 rounded-xl overflow-hidden shadow-2xl border border-zinc-800 relative group">
      
      {/* Canvas Area */}
      <div 
        className="flex-1 relative bg-black flex items-center justify-center p-4 overflow-hidden" 
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: toolMode === ToolMode.POINTER ? 'default' : 'crosshair' }}
      >
        <div className="relative">
          <canvas 
            ref={canvasRef} 
            className="block max-w-full max-h-full object-contain image-pixelated"
            style={{ imageRendering: 'pixelated' }} 
          />
          
          {/* SVG Overlay for Tools */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${width} ${height}`}>
            
            {/* Existing Measurements */}
            {measurements.filter(m => m.viewType === activeView && m.sliceIndex === sliceIndex).map(m => (
              <g key={m.id}>
                <line 
                  x1={m.startX} y1={m.startY} 
                  x2={m.endX} y2={m.endY} 
                  stroke="#34d399" strokeWidth={Math.max(1, width * 0.005)} 
                />
                <circle cx={m.startX} cy={m.startY} r={Math.max(2, width * 0.01)} fill="#34d399" />
                <circle cx={m.endX} cy={m.endY} r={Math.max(2, width * 0.01)} fill="#34d399" />
                <text 
                  x={(m.startX + m.endX)/2} 
                  y={(m.startY + m.endY)/2 - 5} 
                  fill="#34d399" 
                  fontSize={Math.max(12, width * 0.05)}
                  fontWeight="bold"
                  textAnchor="middle"
                  style={{ textShadow: '1px 1px 2px black' }}
                >
                  {m.lengthMm.toFixed(1)} mm
                </text>
              </g>
            ))}

            {/* Existing Annotations */}
            {annotations.filter(a => a.viewType === activeView && a.sliceIndex === sliceIndex).map(a => (
               <g key={a.id}>
                 <circle cx={a.x} cy={a.y} r={Math.max(3, width * 0.015)} fill="#f472b6" stroke="white" strokeWidth="1" />
                 <text 
                   x={a.x + 8} y={a.y + 4} 
                   fill="#f472b6" 
                   fontSize={Math.max(12, width * 0.05)}
                   fontWeight="bold"
                   style={{ textShadow: '1px 1px 2px black' }}
                 >
                   {a.text}
                 </text>
               </g>
            ))}

            {/* Active Drawing Line */}
            {isDragging && startPos && currentPos && (
               <line 
               x1={startPos.x} y1={startPos.y} 
               x2={currentPos.x} y2={currentPos.y} 
               stroke="#fbbf24" strokeWidth={Math.max(1, width * 0.005)} 
               strokeDasharray="4 2"
             />
            )}
          </svg>
        </div>

        {/* Overlay Info */}
        <div className="absolute top-4 left-4 text-xs font-mono text-emerald-400 bg-black/70 p-2 rounded border border-emerald-500/30 backdrop-blur-sm pointer-events-none">
          <div className="font-bold">{activeView.toUpperCase()}</div>
          <div>Slice: {sliceIndex + 1} / {depth}</div>
          <div className="text-zinc-500">{width}x{height} px</div>
        </div>
        
        {/* Gemini Panel */}
        {(analyzing || analysis) && (
             <div className="absolute bottom-4 right-4 max-w-sm w-full bg-zinc-900/95 border border-purple-500/50 p-4 rounded-lg shadow-2xl backdrop-blur text-sm text-zinc-300 z-10">
             <div className="flex justify-between items-center mb-2 border-b border-zinc-700 pb-2">
                <h3 className="font-semibold text-purple-400 flex items-center gap-2">
                    <Brain size={16} /> Gemini Analysis
                </h3>
                <button 
                    onClick={() => setAnalysis(null)} 
                    className="text-zinc-500 hover:text-white"
                >
                    &times;
                </button>
             </div>
             {analyzing ? (
                 <div className="flex items-center gap-2 animate-pulse text-zinc-400">
                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"/>
                    Analyzing anatomy...
                 </div>
             ) : (
                 <div className="prose prose-invert prose-sm max-h-60 overflow-y-auto">
                     <p className="whitespace-pre-line leading-relaxed">{analysis}</p>
                 </div>
             )}
         </div>
        )}
      </div>

      {/* Mini Toolbar */}
      <div className="h-10 bg-zinc-950 border-t border-zinc-800 flex items-center justify-between px-3 gap-2">
          <input 
            type="range" 
            min="0" 
            max={depth - 1} 
            value={sliceIndex} 
            onChange={(e) => onSliceChange(parseInt(e.target.value))}
            className="flex-1 h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400"
          />
          <span className="text-xs font-mono w-8 text-right text-zinc-500">{sliceIndex + 1}</span>
          
          <div className="w-px h-4 bg-zinc-800 mx-1"></div>
          
          <button 
            onClick={handleAnalyze}
            className="text-purple-400 hover:text-purple-300 transition"
            title="Analyze with AI"
          >
            <Brain size={14} />
          </button>
      </div>
    </div>
  );
};

export default Viewer;
