import React from 'react';
import { CheckCircle2, XCircle, AlertCircle, Loader2, Activity } from 'lucide-react';
import { BrainHealthPrediction } from '../services/brainHealthService';

interface BrainHealthPanelProps {
  prediction: BrainHealthPrediction | null;
  isPredicting: boolean;
  error: string | null;
  onClose: () => void;
}

const BrainHealthPanel: React.FC<BrainHealthPanelProps> = ({
  prediction,
  isPredicting,
  error,
  onClose,
}) => {
  if (!prediction && !isPredicting && !error) {
    return null;
  }

  const getPredictionColor = () => {
    if (!prediction) return 'text-zinc-400';
    return prediction.prediction === 'healthy'
      ? 'text-emerald-400'
      : 'text-red-400';
  };

  const getPredictionBg = () => {
    if (!prediction) return 'bg-zinc-900';
    return prediction.prediction === 'healthy'
      ? 'bg-emerald-950/50 border-emerald-800'
      : 'bg-red-950/50 border-red-800';
  };

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.5) return 'Medium';
    return 'Low';
  };

  return (
    <div className="absolute top-4 right-4 w-80 bg-zinc-950 border border-zinc-800 rounded-lg shadow-2xl z-50 max-h-[80vh] overflow-y-auto">
      <div className={`p-4 border-b ${getPredictionBg()}`}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Activity className="text-emerald-400" size={20} />
            <h3 className="text-sm font-semibold text-white">Brain Health Prediction</h3>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-400 hover:text-white transition"
          >
            <XCircle size={18} />
          </button>
        </div>

        {isPredicting && (
          <div className="flex items-center gap-3 py-4">
            <Loader2 className="animate-spin text-emerald-400" size={20} />
            <div>
              <p className="text-sm text-white font-medium">Analyzing brain scan...</p>
              <p className="text-xs text-zinc-400 mt-1">
                Running AI model inference
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-3 py-4">
            <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <p className="text-sm text-red-400 font-medium">Prediction Failed</p>
              <p className="text-xs text-zinc-400 mt-1">{error}</p>
            </div>
          </div>
        )}

        {prediction && !prediction.error && (
          <div className="space-y-4">
            {/* Prediction Result */}
            <div className="flex items-center gap-3">
              {prediction.prediction === 'healthy' ? (
                <CheckCircle2 className="text-emerald-400" size={32} />
              ) : (
                <XCircle className="text-red-400" size={32} />
              )}
              <div className="flex-1">
                <p className={`text-lg font-bold ${getPredictionColor()}`}>
                  {prediction.prediction === 'healthy' ? 'Healthy' : 'Defect Detected'}
                </p>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Confidence: {getConfidenceLevel(prediction.confidence)} ({Math.round(prediction.confidence * 100)}%)
                </p>
              </div>
            </div>

            {/* Confidence Bar */}
            <div>
              <div className="flex justify-between text-xs text-zinc-400 mb-1">
                <span>Confidence</span>
                <span>{Math.round(prediction.confidence * 100)}%</span>
              </div>
              <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    prediction.prediction === 'healthy'
                      ? 'bg-emerald-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${prediction.confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Anomaly Score */}
            <div className="pt-2 border-t border-zinc-800">
              <p className="text-xs text-zinc-400 mb-2">Anomaly Score</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-500 transition-all"
                    style={{
                      width: `${Math.min(prediction.anomaly_score * 10000, 100)}%`,
                    }}
                  />
                </div>
                <span className="text-xs font-mono text-zinc-300">
                  {prediction.anomaly_score.toFixed(4)}
                </span>
              </div>
              <p className="text-xs text-zinc-500 mt-1">
                Lower is better (healthy scans typically &lt; 0.01)
              </p>
            </div>

            {/* Error Metrics */}
            {prediction.error_metrics && (
              <div className="pt-2 border-t border-zinc-800 space-y-2">
                <p className="text-xs text-zinc-400 mb-2">Reconstruction Metrics</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-zinc-500">MSE:</span>
                    <span className="ml-2 font-mono text-zinc-300">
                      {prediction.error_metrics.mse.toFixed(6)}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500">MAE:</span>
                    <span className="ml-2 font-mono text-zinc-300">
                      {prediction.error_metrics.mae.toFixed(6)}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Medical Disclaimer */}
            <div className="pt-2 border-t border-zinc-800">
              <p className="text-xs text-zinc-500 leading-relaxed">
                <strong className="text-zinc-400">⚠️ Disclaimer:</strong> This prediction is for
                research and educational purposes only. It is not a medical diagnosis. Always consult
                a qualified healthcare professional for medical advice.
              </p>
            </div>
          </div>
        )}

        {prediction?.error && (
          <div className="py-4">
            <p className="text-sm text-red-400">{prediction.error}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default BrainHealthPanel;
