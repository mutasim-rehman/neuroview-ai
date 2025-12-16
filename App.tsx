import React, { useState, useMemo, useEffect, useRef } from 'react';
import { parseNifti, createVolumeData, extractTimePoint } from './utils/niftiLoader';
import { VolumeData, ViewType, RenderMode, VolumeRenderStyle, ColorMap, ToolMode, Measurement, Annotation, TissuePreset, ROIStats, OverlayMode, TimeSeriesState, RenderQuality } from './types';
import Viewer from './components/Viewer';
import VolumeViewer from './components/VolumeViewer';
import FileUpload from './components/FileUpload';
import HistogramPanel from './components/HistogramPanel';
import BrainHealthPanel from './components/BrainHealthPanel';
import { predictFromVolumeData, checkApiHealth, BrainHealthPrediction } from './services/brainHealthService';
import { 
  Activity, Layers, Sliders, Info, AlertTriangle, 
  Box, Grid, LayoutDashboard, Ruler, MousePointer2, 
  Type, Droplet, Sun, Eye, Cuboid, Scan, Scissors,
  Thermometer, User, Plus, X, Play, Pause, Volume2,
  ChevronLeft, ChevronRight, Menu, Brain
} from 'lucide-react';

const App: React.FC = () => {
  const [volumes, setVolumes] = useState<VolumeData[]>([]);
  const [activeVolumeId, setActiveVolumeId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Viewer State
  const [renderMode, setRenderMode] = useState<RenderMode>(RenderMode.QUAD);
  const [activeView2D, setActiveView2D] = useState<ViewType>(ViewType.AXIAL);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>(OverlayMode.BLEND);
  
  // Slice Indices
  const [sliceAxial, setSliceAxial] = useState(0);
  const [sliceSagittal, setSliceSagittal] = useState(0);
  const [sliceCoronal, setSliceCoronal] = useState(0);

  // Appearance & Analysis
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
  const [threshold, setThreshold] = useState(0.1);
  const [volumeStyle, setVolumeStyle] = useState<VolumeRenderStyle>(VolumeRenderStyle.VOL);
  const [colorMap, setColorMap] = useState<ColorMap>(ColorMap.ANATOMY);
  const [tissuePreset, setTissuePreset] = useState<TissuePreset>(TissuePreset.CUSTOM);
  const [roiStats, setRoiStats] = useState<ROIStats | null>(null);
  const [renderQuality, setRenderQuality] = useState<RenderQuality>(RenderQuality.HIGH);

  // Tools
  const [toolMode, setToolMode] = useState<ToolMode>(ToolMode.POINTER);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [cutPlane, setCutPlane] = useState(1.0); // 1.0 = No cut, -1.0 = Full cut

  // Time-series state
  const [timeSeriesState, setTimeSeriesState] = useState<TimeSeriesState>({
    currentFrame: 0,
    totalFrames: 1,
    isPlaying: false,
    playbackSpeed: 2,
    loop: true
  });

  // UI State
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(true);

  // Brain Health Prediction State
  const [prediction, setPrediction] = useState<BrainHealthPrediction | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [apiHealth, setApiHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);

  // Get active volume or first volume
  const activeVolume = useMemo(() => {
    if (activeVolumeId) {
      return volumes.find(v => v.metadata.id === activeVolumeId) || volumes[0] || null;
    }
    return volumes[0] || null;
  }, [volumes, activeVolumeId]);

  // Get visible volumes for overlay
  const visibleVolumes = useMemo(() => {
    return volumes.filter(v => v.metadata.visible);
  }, [volumes]);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth()
      .then(health => {
        setApiHealth(health);
        console.log('API Health:', health);
      })
      .catch(err => {
        console.warn('API health check failed:', err);
        setApiHealth({ status: 'error', model_loaded: false });
      });
  }, []);

  // Handle brain health prediction
  const handlePredictHealth = async () => {
    if (!activeVolume) {
      setPredictionError('No volume loaded');
      return;
    }

    setIsPredicting(true);
    setPredictionError(null);
    setPrediction(null);

    try {
      const result = await predictFromVolumeData(activeVolume);
      setPrediction(result);
    } catch (error: any) {
      console.error('Prediction error:', error);
      setPredictionError(error.message || 'Failed to predict brain health');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleFileSelect = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const niftiData = parseNifti(arrayBuffer);
      if (!niftiData) throw new Error("Invalid NIfTI file format.");

      const volumeData = createVolumeData(niftiData, {
        name: file.name.replace(/\.(nii|nii\.gz)$/i, ''),
      });

      setVolumes(prev => {
        const updated = [...prev, volumeData];
        if (updated.length === 1) {
          setActiveVolumeId(volumeData.metadata.id);
          // Reset view defaults for first volume
          const dims = volumeData.header.dims;
          setSliceSagittal(Math.floor(dims[1] / 2));
          setSliceCoronal(Math.floor(dims[2] / 2));
          setSliceAxial(Math.floor(dims[3] / 2));
        }
        return updated;
      });
      
      // Update time-series state if this is a 4D volume
      if (volumeData.isTimeSeries && volumeData.timePoints) {
        setTimeSeriesState(prev => ({
          ...prev,
          totalFrames: volumeData.timePoints,
          currentFrame: 0
        }));
      }
      
      setBrightness(0);
      setContrast(0);
      setThreshold(0.15);
      setRenderMode(RenderMode.QUAD);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to load file.");
    } finally {
      setLoading(false);
    }
  };

  const removeVolume = (volumeId: string) => {
    setVolumes(prev => {
      const filtered = prev.filter(v => v.metadata.id !== volumeId);
      if (activeVolumeId === volumeId && filtered.length > 0) {
        setActiveVolumeId(filtered[0].metadata.id);
      } else if (filtered.length === 0) {
        setActiveVolumeId(null);
      }
      return filtered;
    });
  };

  const updateVolumeMetadata = (volumeId: string, updates: Partial<VolumeData['metadata']>) => {
    setVolumes(prev => prev.map(v => 
      v.metadata.id === volumeId 
        ? { ...v, metadata: { ...v.metadata, ...updates } }
        : v
    ));
  };

  // Get current volume data (handling time-series)
  const getCurrentVolumeData = (volume: VolumeData): VolumeData => {
    if (volume.isTimeSeries && timeSeriesState.currentFrame >= 0) {
      const timePointData = extractTimePoint(volume, timeSeriesState.currentFrame);
      if (timePointData) {
        return createVolumeData(timePointData, volume.metadata);
      }
    }
    return volume;
  };

  // Time-series playback effect
  const playbackIntervalRef = useRef<number | null>(null);
  useEffect(() => {
    if (timeSeriesState.isPlaying && activeVolume?.isTimeSeries && activeVolume.timePoints) {
      playbackIntervalRef.current = window.setInterval(() => {
        setTimeSeriesState(prev => {
          let nextFrame = prev.currentFrame + 1;
          if (nextFrame >= activeVolume.timePoints!) {
            if (prev.loop) {
              nextFrame = 0;
            } else {
              return { ...prev, isPlaying: false, currentFrame: activeVolume.timePoints! - 1 };
            }
          }
          return { ...prev, currentFrame: nextFrame };
        });
      }, 1000 / timeSeriesState.playbackSpeed);
    } else {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
        playbackIntervalRef.current = null;
      }
    }
    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, [timeSeriesState.isPlaying, timeSeriesState.playbackSpeed, activeVolume?.isTimeSeries, activeVolume?.timePoints, timeSeriesState.loop]);

  return (
    <div className="h-screen flex flex-col bg-black text-zinc-200 font-sans overflow-hidden">
      {/* Header */}
      <header className="h-12 border-b border-zinc-800 bg-zinc-950 flex items-center justify-between px-3 flex-shrink-0 z-50">
        <div className="flex items-center gap-3">
            <div className="bg-emerald-600 p-1.5 rounded-lg">
                <Activity size={18} className="text-white" />
            </div>
            <h1 className="text-base font-bold tracking-tight text-white">NeuroView <span className="text-emerald-500">AI</span></h1>
        </div>
        
        {/* Render Mode Switcher */}
        {activeVolume && (
            <div className="flex bg-zinc-900 p-0.5 rounded-lg border border-zinc-800">
                {[RenderMode.SLICE, RenderMode.QUAD, RenderMode.VOLUME].map((mode) => (
                    <button
                        key={mode}
                        onClick={() => setRenderMode(mode)}
                        className={`px-2.5 py-1 text-xs font-medium rounded-md flex items-center gap-1.5 transition-all ${
                            renderMode === mode 
                            ? 'bg-zinc-800 text-white shadow-sm' 
                            : 'text-zinc-500 hover:text-zinc-300'
                        }`}
                    >
                        {mode === RenderMode.SLICE && <Grid size={12} />}
                        {mode === RenderMode.QUAD && <LayoutDashboard size={12} />}
                        {mode === RenderMode.VOLUME && <Box size={12} />}
                        <span className="hidden sm:inline">{mode}</span>
                    </button>
                ))}
            </div>
        )}

        <div className="flex items-center gap-2 text-xs text-zinc-400">
             {activeVolume && (
                 <div className="flex items-center gap-2 bg-zinc-900 py-0.5 px-2 rounded-full border border-zinc-800">
                     <span className="flex items-center gap-1"><Scan size={11}/> {activeVolume.header.dims.slice(1, 4).join('×')}</span>
                     {activeVolume.isTimeSeries && (
                         <span className="flex items-center gap-1 text-emerald-400">
                             <Volume2 size={11}/> {activeVolume.timePoints}
                         </span>
                     )}
                 </div>
             )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden flex flex-row relative min-h-0">
        {!activeVolume ? (
            <div className="flex-1 flex flex-col items-center justify-center p-6">
                {loading ? (
                    <div className="text-center">
                        <div className="w-16 h-16 border-4 border-emerald-600 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
                        <h2 className="text-2xl font-semibold text-white mb-2">Processing Volume...</h2>
                        <p className="text-zinc-500">Decompressing and analyzing voxel data</p>
                    </div>
                ) : (
                    <div className="max-w-xl w-full">
                        <FileUpload onFileSelect={handleFileSelect} />
                        {error && (
                             <div className="mt-6 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-3 text-red-200">
                                 <AlertTriangle size={20} />
                                 {error}
                             </div>
                        )}
                        {volumes.length > 0 && (
                            <div className="mt-4 text-sm text-zinc-400 text-center">
                                {volumes.length} volume{volumes.length > 1 ? 's' : ''} loaded. Add more to enable overlay mode.
                            </div>
                        )}
                    </div>
                )}
            </div>
        ) : (
            <div className="flex-1 flex h-full min-w-0">
                
                {/* Left Toolbar */}
                {leftSidebarOpen && (
                <aside className="w-14 bg-zinc-950 border-r border-zinc-800 flex flex-col items-center py-2 gap-2 z-20 overflow-y-auto scrollbar-hide flex-shrink-0">
                     <div className="flex flex-col gap-1.5 w-full px-1.5">
                        <div className="text-[9px] text-zinc-600 font-bold uppercase text-center mb-0.5">Tools</div>
                        {[
                          { mode: ToolMode.POINTER, icon: MousePointer2 },
                          { mode: ToolMode.MEASURE, icon: Ruler },
                          { mode: ToolMode.ANNOTATE, icon: Type }
                        ].map(tool => (
                            <button
                                key={tool.mode}
                                onClick={() => setToolMode(tool.mode)}
                                className={`p-2 rounded-lg flex items-center justify-center transition-all ${
                                    toolMode === tool.mode 
                                    ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/20' 
                                    : 'text-zinc-500 hover:bg-zinc-900 hover:text-zinc-200'
                                }`}
                                title={tool.mode}
                            >
                                <tool.icon size={18} />
                            </button>
                        ))}
                     </div>
                     
                     <div className="w-6 h-px bg-zinc-800 my-1"></div>

                     <div className="flex flex-col gap-1.5 w-full px-1.5">
                        <div className="text-[9px] text-zinc-600 font-bold uppercase text-center mb-0.5">3D Style</div>
                        <button onClick={() => setVolumeStyle(VolumeRenderStyle.MIP)} className={`p-1.5 rounded-lg ${volumeStyle === VolumeRenderStyle.MIP ? 'text-emerald-400 bg-zinc-900' : 'text-zinc-500'}`} title="X-Ray">
                            <Sun size={18} />
                        </button>
                        <button onClick={() => setVolumeStyle(VolumeRenderStyle.ISO)} className={`p-1.5 rounded-lg ${volumeStyle === VolumeRenderStyle.ISO ? 'text-emerald-400 bg-zinc-900' : 'text-zinc-500'}`} title="Surface">
                            <Cuboid size={18} />
                        </button>
                        <button onClick={() => setVolumeStyle(VolumeRenderStyle.VOL)} className={`p-1.5 rounded-lg ${volumeStyle === VolumeRenderStyle.VOL ? 'text-emerald-400 bg-zinc-900' : 'text-zinc-500'}`} title="Volume">
                            <Droplet size={18} />
                        </button>
                     </div>
                     
                     <div className="w-6 h-px bg-zinc-800 my-1"></div>
                     
                     {/* Tissue Presets (Peeling) */}
                     <div className="flex flex-col gap-1.5 w-full px-1.5">
                         <div className="text-[9px] text-zinc-600 font-bold uppercase text-center mb-0.5">Peel</div>
                         <button 
                            onClick={() => setTissuePreset(TissuePreset.SKIN)} 
                            className={`p-1.5 rounded-lg ${tissuePreset === TissuePreset.SKIN ? 'text-emerald-400 bg-zinc-900' : 'text-zinc-500'}`} 
                            title="Skin"
                         >
                            <User size={16} />
                         </button>
                         <button 
                            onClick={() => setTissuePreset(TissuePreset.BONE)} 
                            className={`p-1.5 rounded-lg ${tissuePreset === TissuePreset.BONE ? 'text-emerald-400 bg-zinc-900' : 'text-zinc-500'}`} 
                            title="Bone"
                         >
                            <Layers size={16} />
                         </button>
                     </div>
                </aside>
                )}
                
                {/* Left Sidebar Toggle */}
                <button
                    onClick={() => setLeftSidebarOpen(!leftSidebarOpen)}
                    className="absolute left-0 top-1/2 -translate-y-1/2 z-30 bg-zinc-900 border border-zinc-800 border-l-0 rounded-r-lg p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all"
                    title={leftSidebarOpen ? "Hide toolbar" : "Show toolbar"}
                >
                    {leftSidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
                </button>

                {/* Viewport Area */}
                <div className="flex-1 bg-black overflow-hidden relative min-w-0">
                    
                    {/* Render Modes */}
                    {renderMode === RenderMode.QUAD && activeVolume && (
                        <div className="grid grid-cols-2 grid-rows-2 w-full h-full gap-0.5 bg-zinc-900 p-0.5">
                            <Viewer 
                                volumes={visibleVolumes.map(v => getCurrentVolumeData(v))}
                                activeView={ViewType.AXIAL} 
                                sliceIndex={sliceAxial} 
                                onSliceChange={setSliceAxial}
                                brightness={brightness} contrast={contrast}
                                toolMode={toolMode}
                                measurements={measurements} onAddMeasurement={m => setMeasurements([...measurements, m])}
                                annotations={annotations} onAddAnnotation={a => setAnnotations([...annotations, a])}
                                onROIStatsUpdate={setRoiStats}
                                overlayMode={overlayMode}
                            />
                            <Viewer 
                                volumes={visibleVolumes.map(v => getCurrentVolumeData(v))}
                                activeView={ViewType.SAGITTAL} 
                                sliceIndex={sliceSagittal} 
                                onSliceChange={setSliceSagittal}
                                brightness={brightness} contrast={contrast}
                                toolMode={toolMode}
                                measurements={measurements} onAddMeasurement={m => setMeasurements([...measurements, m])}
                                annotations={annotations} onAddAnnotation={a => setAnnotations([...annotations, a])}
                                onROIStatsUpdate={setRoiStats}
                                overlayMode={overlayMode}
                            />
                            <Viewer 
                                volumes={visibleVolumes.map(v => getCurrentVolumeData(v))}
                                activeView={ViewType.CORONAL} 
                                sliceIndex={sliceCoronal} 
                                onSliceChange={setSliceCoronal}
                                brightness={brightness} contrast={contrast}
                                toolMode={toolMode}
                                measurements={measurements} onAddMeasurement={m => setMeasurements([...measurements, m])}
                                annotations={annotations} onAddAnnotation={a => setAnnotations([...annotations, a])}
                                onROIStatsUpdate={setRoiStats}
                                overlayMode={overlayMode}
                            />
                            <div className="relative rounded-xl overflow-hidden border border-zinc-800">
                                <VolumeViewer 
                                    volumes={visibleVolumes.map(v => getCurrentVolumeData(v))}
                                    threshold={threshold} 
                                    brightness={brightness}
                                    renderStyle={volumeStyle}
                                    colorMap={colorMap}
                                    slices={{
                                        x: sliceSagittal / activeVolume.header.dims[1],
                                        y: sliceCoronal / activeVolume.header.dims[2],
                                        z: sliceAxial / activeVolume.header.dims[3]
                                    }}
                                    cutPlane={cutPlane}
                                    preset={tissuePreset}
                                    renderQuality={renderQuality}
                                    isolateBrain={tissuePreset === TissuePreset.BRAIN}
                                />
                            </div>
                        </div>
                    )}

                    {renderMode === RenderMode.SLICE && activeVolume && (
                         <div className="w-full h-full p-4 relative">
                             <Viewer 
                                volumes={visibleVolumes.map(v => getCurrentVolumeData(v))}
                                activeView={activeView2D} 
                                sliceIndex={activeView2D === ViewType.AXIAL ? sliceAxial : activeView2D === ViewType.SAGITTAL ? sliceSagittal : sliceCoronal}
                                onSliceChange={(idx) => {
                                    if(activeView2D === ViewType.AXIAL) setSliceAxial(idx);
                                    else if(activeView2D === ViewType.SAGITTAL) setSliceSagittal(idx);
                                    else setSliceCoronal(idx);
                                }}
                                brightness={brightness} contrast={contrast}
                                toolMode={toolMode}
                                measurements={measurements} onAddMeasurement={m => setMeasurements([...measurements, m])}
                                annotations={annotations} onAddAnnotation={a => setAnnotations([...annotations, a])}
                                onROIStatsUpdate={setRoiStats}
                                overlayMode={overlayMode}
                            />
                            <div className="absolute top-6 right-6 flex flex-col gap-2">
                                {Object.values(ViewType).map(v => (
                                    <button 
                                        key={v}
                                        onClick={() => setActiveView2D(v)}
                                        className={`px-3 py-1 text-xs font-bold uppercase rounded border ${activeView2D === v ? 'bg-emerald-600 border-emerald-500' : 'bg-black/50 border-zinc-700'}`}
                                    >
                                        {v}
                                    </button>
                                ))}
                            </div>
                         </div>
                    )}

                    {renderMode === RenderMode.VOLUME && activeVolume && (
                        <div className="w-full h-full p-1 relative">
                             <VolumeViewer 
                                volumes={visibleVolumes.map(v => getCurrentVolumeData(v))}
                                threshold={threshold} 
                                brightness={brightness}
                                renderStyle={volumeStyle}
                                colorMap={colorMap}
                                slices={{
                                    x: sliceSagittal / activeVolume.header.dims[1],
                                    y: sliceCoronal / activeVolume.header.dims[2],
                                    z: sliceAxial / activeVolume.header.dims[3]
                                }}
                                cutPlane={cutPlane}
                                preset={tissuePreset}
                                renderQuality={renderQuality}
                                isolateBrain={tissuePreset === TissuePreset.BRAIN}
                            />
                        </div>
                    )}
                </div>

                {/* Right Sidebar Toggle */}
                <button
                    onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
                    className="absolute right-0 top-1/2 -translate-y-1/2 z-30 bg-zinc-900 border border-zinc-800 border-r-0 rounded-l-lg p-1.5 text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all"
                    title={rightSidebarOpen ? "Hide settings" : "Show settings"}
                >
                    {rightSidebarOpen ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                </button>

                {/* Right Settings Panel */}
                {rightSidebarOpen && (
                <aside className="w-72 bg-zinc-950 border-l border-zinc-800 flex flex-col p-3 overflow-y-auto flex-shrink-0">
                    
                    {/* Brain Health Prediction */}
                    <div className="mb-4 p-3 bg-gradient-to-br from-emerald-950/30 to-zinc-900 rounded-lg border border-emerald-800/30">
                        <div className="flex items-center gap-2 mb-3">
                            <Brain className="text-emerald-400" size={18} />
                            <h3 className="text-sm font-semibold text-white">AI Health Prediction</h3>
                        </div>
                        {apiHealth && (
                            <div className="flex items-center gap-2 mb-3 text-xs">
                                <div className={`w-2 h-2 rounded-full ${apiHealth.model_loaded ? 'bg-emerald-500' : 'bg-red-500'}`} />
                                <span className="text-zinc-400">
                                    {apiHealth.model_loaded ? 'Model Ready' : 'Model Not Loaded'}
                                </span>
                            </div>
                        )}
                        <button
                            onClick={handlePredictHealth}
                            disabled={isPredicting || !activeVolume || (apiHealth && !apiHealth.model_loaded)}
                            className={`w-full px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                                isPredicting || !activeVolume || (apiHealth && !apiHealth.model_loaded)
                                    ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                                    : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                            }`}
                        >
                            {isPredicting ? 'Predicting...' : 'Predict Health Status'}
                        </button>
                        {prediction && (
                            <button
                                onClick={() => setPrediction(null)}
                                className="w-full mt-2 px-3 py-1.5 text-xs text-zinc-400 hover:text-white transition"
                            >
                                Clear Results
                            </button>
                        )}
                    </div>

                    {/* Volume Management */}
                    {volumes.length > 0 && (
                        <div className="mb-4 bg-zinc-900 rounded-lg p-2 border border-zinc-800">
                            <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                <Layers size={12}/> Volumes ({volumes.length})
                            </h3>
                            <div className="space-y-2 max-h-48 overflow-y-auto">
                                {volumes.map(volume => (
                                    <div key={volume.metadata.id} className={`p-2 rounded border ${activeVolumeId === volume.metadata.id ? 'border-emerald-500 bg-emerald-900/20' : 'border-zinc-800 bg-zinc-950'}`}>
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center gap-2 flex-1 min-w-0">
                                                <input
                                                    type="checkbox"
                                                    checked={volume.metadata.visible}
                                                    onChange={(e) => updateVolumeMetadata(volume.metadata.id, { visible: e.target.checked })}
                                                    className="w-3 h-3 accent-emerald-600"
                                                />
                                                <span 
                                                    className="text-xs font-medium text-zinc-300 truncate cursor-pointer"
                                                    onClick={() => setActiveVolumeId(volume.metadata.id)}
                                                    title={volume.metadata.name}
                                                >
                                                    {volume.metadata.name}
                                                </span>
                                            </div>
                                            {volumes.length > 1 && (
                                                <button
                                                    onClick={() => removeVolume(volume.metadata.id)}
                                                    className="text-zinc-500 hover:text-red-400 transition"
                                                >
                                                    <X size={14} />
                                                </button>
                                            )}
                                        </div>
                                        <div className="flex items-center gap-2 text-[10px] text-zinc-500">
                                            <input
                                                type="range"
                                                min="0"
                                                max="1"
                                                step="0.01"
                                                value={volume.metadata.opacity}
                                                onChange={(e) => updateVolumeMetadata(volume.metadata.id, { opacity: Number(e.target.value) })}
                                                className="flex-1 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                                            />
                                            <span className="w-8 text-right">{Math.round(volume.metadata.opacity * 100)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            {volumes.length > 1 && (
                                <div className="mt-3 pt-3 border-t border-zinc-800">
                                    <label className="text-[10px] text-zinc-500 uppercase mb-1 block">Overlay Mode</label>
                                    <select
                                        value={overlayMode}
                                        onChange={(e) => setOverlayMode(e.target.value as OverlayMode)}
                                        className="w-full text-xs bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-zinc-300"
                                    >
                                        {Object.values(OverlayMode).map(mode => (
                                            <option key={mode} value={mode}>{mode}</option>
                                        ))}
                                    </select>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Time-Series Controls */}
                    {activeVolume?.isTimeSeries && activeVolume.timePoints && activeVolume.timePoints > 1 && (
                        <div className="mb-4 bg-zinc-900 rounded-lg p-2 border border-zinc-800">
                            <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                                <Volume2 size={12}/> Time Series
                            </h3>
                            <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => setTimeSeriesState(prev => ({ ...prev, isPlaying: !prev.isPlaying }))}
                                        className="p-2 rounded bg-zinc-800 hover:bg-zinc-700 text-emerald-400"
                                    >
                                        {timeSeriesState.isPlaying ? <Pause size={16} /> : <Play size={16} />}
                                    </button>
                                    <div className="flex-1">
                                        <input
                                            type="range"
                                            min="0"
                                            max={activeVolume.timePoints - 1}
                                            value={timeSeriesState.currentFrame}
                                            onChange={(e) => setTimeSeriesState(prev => ({ ...prev, currentFrame: Number(e.target.value) }))}
                                            className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                                        />
                                    </div>
                                    <span className="text-xs text-zinc-400 w-12 text-right">
                                        {timeSeriesState.currentFrame + 1}/{activeVolume.timePoints}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between text-[10px] text-zinc-500">
                                    <label>Speed:</label>
                                    <input
                                        type="range"
                                        min="0.5"
                                        max="5"
                                        step="0.5"
                                        value={timeSeriesState.playbackSpeed}
                                        onChange={(e) => setTimeSeriesState(prev => ({ ...prev, playbackSpeed: Number(e.target.value) }))}
                                        className="w-24 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                                    />
                                    <span className="text-zinc-400">{timeSeriesState.playbackSpeed}x</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* ROI Stats Widget */}
                    <div className="mb-4 bg-zinc-900 rounded-lg p-2 border border-zinc-800 shadow-inner">
                        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Activity size={12}/> Region Analysis
                        </h3>
                        {roiStats ? (
                            <div className="space-y-2">
                                <div className="flex justify-between items-center">
                                    <span className="text-zinc-400 text-xs">Voxel Intensity</span>
                                    <span className="text-emerald-400 font-mono font-bold text-lg">{roiStats.value.toFixed(0)}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-zinc-400 text-xs">Density Class</span>
                                    <span className="px-2 py-0.5 rounded bg-emerald-900/40 text-emerald-300 text-xs border border-emerald-800">
                                        {roiStats.densityLabel}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center text-xs font-mono text-zinc-500 pt-1 border-t border-zinc-800 mt-2">
                                    <span>X:{roiStats.x}</span>
                                    <span>Y:{roiStats.y}</span>
                                    <span>Z:{roiStats.z}</span>
                                </div>
                            </div>
                        ) : (
                            <div className="h-20 flex flex-col items-center justify-center text-zinc-600 text-xs italic text-center">
                                <MousePointer2 size={16} className="mb-1 opacity-50"/>
                                Hover over slice to inspect tissue
                            </div>
                        )}
                    </div>

                    <div className="mb-4">
                        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Scissors size={12}/> Lightsaber Cut
                        </h3>
                        <div className="space-y-2">
                            <div className="flex justify-between text-xs">
                                <span className="text-zinc-400">Cut Plane Position</span>
                                <span className="text-red-400 font-mono">{(cutPlane * 100).toFixed(0)}%</span>
                            </div>
                            <input 
                                type="range" 
                                min="-1.0" 
                                max="1.0" 
                                step="0.01" 
                                value={cutPlane} 
                                onChange={(e) => setCutPlane(Number(e.target.value))} 
                                className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-red-500 hover:accent-red-400"
                            />
                            <div className="flex justify-between text-[10px] text-zinc-600">
                                <span>Front</span>
                                <span>Back</span>
                            </div>
                        </div>
                    </div>

                    {/* Histogram Panel */}
                    {activeVolume && (
                        <div className="mb-6">
                            <HistogramPanel
                                volume={activeVolume}
                                window={activeVolume.metadata.window}
                                level={activeVolume.metadata.level}
                                onWindowChange={(w) => updateVolumeMetadata(activeVolume.metadata.id, { window: w })}
                                onLevelChange={(l) => updateVolumeMetadata(activeVolume.metadata.id, { level: l })}
                                onPresetChange={(preset) => {
                                    // Preset change handled by HistogramPanel
                                }}
                            />
                        </div>
                    )}

                    <div className="mb-4">
                        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Sliders size={12}/> Adjustments
                        </h3>
                        <div className="space-y-5">
                             {/* Quick brain isolation helper */}
                             <div className="flex items-center justify-between text-xs">
                                <span className="text-zinc-400 flex items-center gap-1.5">
                                    <Brain size={12} className="text-emerald-400" />
                                    Isolate
                                </span>
                                <button
                                    onClick={() => {
                                        // Configure view to highlight brain only
                                        setRenderMode(RenderMode.VOLUME);
                                        setVolumeStyle(VolumeRenderStyle.ISO);
                                        setTissuePreset(TissuePreset.BRAIN);
                                        setThreshold(0.4);
                                        setColorMap(ColorMap.ANATOMY);
                                        setCutPlane(1.0);
                                    }}
                                    className="px-2 py-1 rounded-full border border-emerald-500/60 text-emerald-300 hover:bg-emerald-600/20 hover:text-emerald-100 transition text-[11px] font-medium"
                                >
                                    Isolate Brain
                                </button>
                             </div>
                             <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                    <span className="text-zinc-400">Density Threshold</span>
                                    <span className="text-emerald-400 font-mono">{threshold.toFixed(2)}</span>
                                </div>
                                <input type="range" min="0" max="1" step="0.01" value={threshold} onChange={(e) => { setThreshold(Number(e.target.value)); setTissuePreset(TissuePreset.CUSTOM); }} className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"/>
                            </div>
                             <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                    <span className="text-zinc-400">Brightness</span>
                                    <span className="text-zinc-300">{brightness}</span>
                                </div>
                                <input type="range" min="-100" max="100" value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"/>
                            </div>
                        </div>
                    </div>

                    <div className="mb-4 border-t border-zinc-900 pt-3">
                        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Eye size={12}/> 3D Color Map
                        </h3>
                        <div className="grid grid-cols-2 gap-2">
                            {Object.values(ColorMap).map(cm => (
                                <button 
                                    key={cm}
                                    onClick={() => setColorMap(cm)}
                                    className={`text-xs py-2 rounded border ${colorMap === cm ? 'bg-emerald-900/30 border-emerald-500 text-emerald-400' : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:border-zinc-600'}`}
                                >
                                    {cm}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="mb-4 border-t border-zinc-900 pt-3">
                        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Box size={12}/> Render Quality
                        </h3>
                        <div className="grid grid-cols-2 gap-2">
                            {Object.values(RenderQuality).map(quality => (
                                <button 
                                    key={quality}
                                    onClick={() => setRenderQuality(quality)}
                                    className={`text-xs py-2 rounded border ${renderQuality === quality ? 'bg-emerald-900/30 border-emerald-500 text-emerald-400' : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:border-zinc-600'}`}
                                >
                                    {quality}
                                </button>
                            ))}
                        </div>
                        <p className="text-[10px] text-zinc-600 mt-2">
                            {renderQuality === RenderQuality.FAST && 'Fast rendering, lower quality'}
                            {renderQuality === RenderQuality.MEDIUM && 'Balanced quality and performance'}
                            {renderQuality === RenderQuality.HIGH && 'High quality, good performance'}
                            {renderQuality === RenderQuality.ULTRA && 'Ultra quality, may be slower'}
                        </p>
                    </div>

                    <div className="mb-4 border-t border-zinc-900 pt-3 flex-1">
                        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                            <Ruler size={12}/> Measurements
                        </h3>
                        {measurements.length === 0 && <p className="text-xs text-zinc-600 italic">No measurements yet. Select Ruler tool and drag on slice.</p>}
                        <div className="space-y-2 max-h-40 overflow-y-auto pr-2">
                            {measurements.map((m, i) => (
                                <div key={m.id} className="flex justify-between items-center text-xs bg-zinc-900 p-2 rounded border border-zinc-800">
                                    <span className="text-zinc-400">#{i+1} {m.viewType}</span>
                                    <span className="text-emerald-400 font-mono font-bold">{m.lengthMm.toFixed(1)} mm</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </aside>
                )}
            </div>
        )}

        {/* Brain Health Prediction Panel */}
        {(prediction || isPredicting || predictionError) && (
            <BrainHealthPanel
                prediction={prediction}
                isPredicting={isPredicting}
                error={predictionError}
                onClose={() => {
                    setPrediction(null);
                    setPredictionError(null);
                }}
            />
        )}
      </main>
    </div>
  );
};

export default App;
