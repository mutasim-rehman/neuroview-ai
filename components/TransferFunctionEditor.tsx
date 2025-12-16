import React, { useCallback, useRef, useState } from 'react';
import { TransferFunction, TransferFunctionPoint } from '../types';
import { X, Plus } from 'lucide-react';

interface TransferFunctionEditorProps {
  transferFunction: TransferFunction;
  onTransferFunctionChange: (tf: TransferFunction) => void;
}

const TransferFunctionEditor: React.FC<TransferFunctionEditorProps> = ({
  transferFunction,
  onTransferFunctionChange
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const drawTransferFunction = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      
      const y = (i / 10) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Sort points by value
    const sortedPoints = [...transferFunction.points].sort((a, b) => a.value - b.value);

    if (sortedPoints.length === 0) return;

    // Draw opacity curve
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < sortedPoints.length; i++) {
      const x = sortedPoints[i].value * width;
      const y = (1 - sortedPoints[i].opacity) * height;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw color gradient
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    for (let i = 0; i < sortedPoints.length; i++) {
      const point = sortedPoints[i];
      const color = `rgb(${Math.round(point.color[0] * 255)}, ${Math.round(point.color[1] * 255)}, ${Math.round(point.color[2] * 255)})`;
      gradient.addColorStop(point.value, color);
    }
    ctx.fillStyle = gradient;
    ctx.fillRect(0, height - 20, width, 20);

    // Draw control points
    sortedPoints.forEach((point, index) => {
      const x = point.value * width;
      const y = (1 - point.opacity) * height;
      
      const isSelected = selectedPoint === index;
      ctx.fillStyle = isSelected ? '#00ff00' : '#ff00ff';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.stroke();
    });
  }, [transferFunction, selectedPoint]);

  React.useEffect(() => {
    drawTransferFunction();
  }, [drawTransferFunction]);

  const getPointAtPosition = (x: number, y: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const canvasX = ((x - rect.left) / rect.width) * canvas.width;
    const canvasY = ((y - rect.top) / rect.height) * canvas.height;

    const value = Math.max(0, Math.min(1, canvasX / canvas.width));
    const opacity = Math.max(0, Math.min(1, 1 - (canvasY / canvas.height)));

    // Find nearest point
    const sortedPoints = [...transferFunction.points].sort((a, b) => a.value - b.value);
    let nearestIdx = -1;
    let minDist = Infinity;

    sortedPoints.forEach((point, idx) => {
      const dist = Math.abs(point.value - value) * canvas.width;
      if (dist < 20 && dist < minDist) {
        minDist = dist;
        nearestIdx = idx;
      }
    });

    return { value, opacity, nearestIdx };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const point = getPointAtPosition(e.clientX, e.clientY);
    if (!point) return;

    if (point.nearestIdx >= 0) {
      // Select existing point
      setSelectedPoint(point.nearestIdx);
      setIsDragging(true);
    } else {
      // Create new point
      const sortedPoints = [...transferFunction.points].sort((a, b) => a.value - b.value);
      const newPoint: TransferFunctionPoint = {
        value: point.value,
        opacity: point.opacity,
        color: [1, 1, 1] // Default white
      };
      
      const newPoints = [...transferFunction.points, newPoint];
      onTransferFunctionChange({
        ...transferFunction,
        points: newPoints
      });
      
      // Select the new point
      const newIdx = newPoints.length - 1;
      setSelectedPoint(newIdx);
      setIsDragging(true);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || selectedPoint === null) return;

    const point = getPointAtPosition(e.clientX, e.clientY);
    if (!point) return;

    const newPoints = [...transferFunction.points];
    newPoints[selectedPoint] = {
      ...newPoints[selectedPoint],
      value: point.value,
      opacity: point.opacity
    };

    onTransferFunctionChange({
      ...transferFunction,
      points: newPoints
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleDeletePoint = () => {
    if (selectedPoint === null || transferFunction.points.length <= 1) return;

    const newPoints = transferFunction.points.filter((_, idx) => idx !== selectedPoint);
    onTransferFunctionChange({
      ...transferFunction,
      points: newPoints
    });
    setSelectedPoint(null);
  };

  const handleColorChange = (color: [number, number, number]) => {
    if (selectedPoint === null) return;

    const newPoints = [...transferFunction.points];
    newPoints[selectedPoint] = {
      ...newPoints[selectedPoint],
      color
    };

    onTransferFunctionChange({
      ...transferFunction,
      points: newPoints
    });
  };

  const selectedPointData = selectedPoint !== null ? transferFunction.points[selectedPoint] : null;

  return (
    <div className="bg-zinc-900 rounded-lg p-3 border border-zinc-800">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold text-zinc-400 uppercase">Transfer Function</h3>
        <label className="flex items-center gap-2 text-xs text-zinc-500">
          <input
            type="checkbox"
            checked={transferFunction.enabled}
            onChange={(e) => onTransferFunctionChange({
              ...transferFunction,
              enabled: e.target.checked
            })}
            className="w-3 h-3 accent-emerald-600"
          />
          Enable
        </label>
      </div>

      <canvas
        ref={canvasRef}
        width={256}
        height={128}
        className="w-full h-32 bg-zinc-950 rounded border border-zinc-800 cursor-crosshair"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {selectedPointData && (
        <div className="mt-3 p-2 bg-zinc-950 rounded border border-zinc-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-zinc-400">Selected Point</span>
            <button
              onClick={handleDeletePoint}
              className="text-red-400 hover:text-red-300 text-xs"
              disabled={transferFunction.points.length <= 1}
            >
              <X size={14} />
            </button>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 w-16">Value:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={selectedPointData.value}
                onChange={(e) => {
                  const newPoints = [...transferFunction.points];
                  newPoints[selectedPoint!] = {
                    ...newPoints[selectedPoint!],
                    value: Number(e.target.value)
                  };
                  onTransferFunctionChange({
                    ...transferFunction,
                    points: newPoints
                  });
                }}
                className="flex-1 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <span className="text-zinc-300 w-12 text-right">{selectedPointData.value.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 w-16">Opacity:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={selectedPointData.opacity}
                onChange={(e) => {
                  const newPoints = [...transferFunction.points];
                  newPoints[selectedPoint!] = {
                    ...newPoints[selectedPoint!],
                    opacity: Number(e.target.value)
                  };
                  onTransferFunctionChange({
                    ...transferFunction,
                    points: newPoints
                  });
                }}
                className="flex-1 h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <span className="text-zinc-300 w-12 text-right">{selectedPointData.opacity.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-zinc-500 w-16">Color:</span>
              <input
                type="color"
                value={`#${Math.round(selectedPointData.color[0] * 255).toString(16).padStart(2, '0')}${Math.round(selectedPointData.color[1] * 255).toString(16).padStart(2, '0')}${Math.round(selectedPointData.color[2] * 255).toString(16).padStart(2, '0')}`}
                onChange={(e) => {
                  const hex = e.target.value;
                  const r = parseInt(hex.slice(1, 3), 16) / 255;
                  const g = parseInt(hex.slice(3, 5), 16) / 255;
                  const b = parseInt(hex.slice(5, 7), 16) / 255;
                  handleColorChange([r, g, b]);
                }}
                className="w-12 h-6 rounded border border-zinc-700 cursor-pointer"
              />
              <div className="flex-1 text-zinc-400 text-[10px]">
                RGB: ({Math.round(selectedPointData.color[0] * 255)}, {Math.round(selectedPointData.color[1] * 255)}, {Math.round(selectedPointData.color[2] * 255)})
              </div>
            </div>
          </div>
        </div>
      )}

      {transferFunction.points.length === 0 && (
        <div className="mt-2 text-xs text-zinc-600 text-center">
          Click on the canvas to add control points
        </div>
      )}
    </div>
  );
};

export default TransferFunctionEditor;

