import React, { useMemo } from 'react';
import { VolumeData, WindowLevelPreset, HistogramData } from '../types';
import { BarChart3 } from 'lucide-react';

interface HistogramPanelProps {
  volume: VolumeData;
  window: number;
  level: number;
  onWindowChange: (window: number) => void;
  onLevelChange: (level: number) => void;
  onPresetChange?: (preset: WindowLevelPreset) => void;
}

// Window/Level presets
const PRESETS: Record<WindowLevelPreset, { window: number; level: number }> = {
  [WindowLevelPreset.BRAIN]: { window: 80, level: 40 },
  [WindowLevelPreset.BONE]: { window: 2000, level: 400 },
  [WindowLevelPreset.LUNG]: { window: 1500, level: -600 },
  [WindowLevelPreset.SOFT_TISSUE]: { window: 400, level: 50 },
  [WindowLevelPreset.ABDOMEN]: { window: 400, level: 50 },
  [WindowLevelPreset.CUSTOM]: { window: 0, level: 0 }, // Not used
};

// Calculate histogram from volume data
const calculateHistogram = (volume: VolumeData, bins: number = 256): HistogramData => {
  const data = volume.image as any;
  const min = volume.min;
  const max = volume.max;
  const range = max - min;
  
  const histogram = new Array(bins).fill(0);
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  
  // Sample data for performance (every Nth voxel)
  const step = Math.max(1, Math.floor(data.length / 100000));
  
  for (let i = 0; i < data.length; i += step) {
    const val = data[i];
    if (val >= min && val <= max) {
      const bin = Math.floor(((val - min) / range) * (bins - 1));
      if (bin >= 0 && bin < bins) {
        histogram[bin]++;
        sum += val;
        sumSq += val * val;
        count++;
      }
    }
  }
  
  const mean = count > 0 ? sum / count : 0;
  const variance = count > 0 ? (sumSq / count) - (mean * mean) : 0;
  const stdDev = Math.sqrt(Math.max(0, variance));
  
  // Find median
  const sorted = [...data].sort((a, b) => a - b);
  const median = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : 0;
  
  return {
    bins: histogram,
    min,
    max,
    mean,
    median,
    stdDev,
  };
};

const HistogramPanel: React.FC<HistogramPanelProps> = ({
  volume,
  window,
  level,
  onWindowChange,
  onLevelChange,
  onPresetChange,
}) => {
  const histogram = useMemo(() => calculateHistogram(volume), [volume]);
  
  const maxBinValue = Math.max(...histogram.bins);
  const windowMin = level - window / 2;
  const windowMax = level + window / 2;
  const range = histogram.max - histogram.min;
  const windowMinBin = Math.floor(((windowMin - histogram.min) / range) * (histogram.bins.length - 1));
  const windowMaxBin = Math.floor(((windowMax - histogram.min) / range) * (histogram.bins.length - 1));
  
  // Determine current preset
  const currentPreset = useMemo(() => {
    for (const [preset, values] of Object.entries(PRESETS)) {
      if (preset === WindowLevelPreset.CUSTOM) continue;
      if (Math.abs(values.window - window) < 10 && Math.abs(values.level - level) < 10) {
        return preset as WindowLevelPreset;
      }
    }
    return WindowLevelPreset.CUSTOM;
  }, [window, level]);

  return (
    <div className="bg-zinc-900 rounded-lg p-2 border border-zinc-800">
      <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
        <BarChart3 size={12}/> Histogram & Window/Level
      </h3>
      
      {/* Histogram Visualization */}
      <div className="mb-3">
        <div className="h-24 bg-black rounded border border-zinc-800 p-1.5 relative overflow-hidden">
          <svg width="100%" height="100%" viewBox="0 0 256 120" preserveAspectRatio="none">
            {/* Histogram bars */}
            {histogram.bins.map((value, index) => {
              const height = (value / maxBinValue) * 100;
              const isInWindow = index >= windowMinBin && index <= windowMaxBin;
              const color = isInWindow ? '#10b981' : '#3f3f46';
              
              return (
                <rect
                  key={index}
                  x={index}
                  y={120 - height}
                  width={1}
                  height={height}
                  fill={color}
                  opacity={0.8}
                />
              );
            })}
            
            {/* Window/Level indicators */}
            <line
              x1={windowMinBin}
              y1={0}
              x2={windowMinBin}
              y2={120}
              stroke="#ef4444"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
            <line
              x1={windowMaxBin}
              y1={0}
              x2={windowMaxBin}
              y2={120}
              stroke="#ef4444"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          </svg>
        </div>
        
        {/* Stats */}
        <div className="flex justify-between text-[10px] text-zinc-500 mt-1">
          <span>Min: {histogram.min.toFixed(0)}</span>
          <span>Mean: {histogram.mean.toFixed(0)}</span>
          <span>Max: {histogram.max.toFixed(0)}</span>
        </div>
      </div>

      {/* Presets */}
      {onPresetChange && (
        <div className="mb-3">
          <label className="text-[9px] text-zinc-500 uppercase mb-1.5 block">Presets</label>
          <div className="grid grid-cols-2 gap-1">
            {Object.values(WindowLevelPreset)
              .filter(p => p !== WindowLevelPreset.CUSTOM)
              .map(preset => (
                <button
                  key={preset}
                  onClick={() => {
                    const values = PRESETS[preset];
                    onWindowChange(values.window);
                    onLevelChange(values.level);
                    onPresetChange(preset);
                  }}
                  className={`text-xs py-1.5 px-2 rounded border transition ${
                    currentPreset === preset
                      ? 'bg-emerald-900/30 border-emerald-500 text-emerald-400'
                      : 'bg-zinc-800 border-zinc-700 text-zinc-500 hover:border-zinc-600'
                  }`}
                >
                  {preset}
                </button>
              ))}
          </div>
        </div>
      )}

      {/* Window/Level Controls */}
      <div className="space-y-2">
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-zinc-400">Window</span>
            <span className="text-emerald-400 font-mono">{window.toFixed(0)}</span>
          </div>
          <input
            type="range"
            min={1}
            max={volume.max - volume.min}
            step={1}
            value={window}
            onChange={(e) => onWindowChange(Number(e.target.value))}
            className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-zinc-400">Level</span>
            <span className="text-emerald-400 font-mono">{level.toFixed(0)}</span>
          </div>
          <input
            type="range"
            min={volume.min}
            max={volume.max}
            step={1}
            value={level}
            onChange={(e) => onLevelChange(Number(e.target.value))}
            className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
          />
        </div>
      </div>
    </div>
  );
};

export default HistogramPanel;

