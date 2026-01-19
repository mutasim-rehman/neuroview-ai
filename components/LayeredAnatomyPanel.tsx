import React from 'react';
import { Brain, X, Info, Focus } from 'lucide-react';
import { BrainPart, LayeredAnatomyState } from '../types';

interface LayeredAnatomyPanelProps {
  state: LayeredAnatomyState;
  onToggleMode: (enabled: boolean) => void;
  onTogglePart: (partId: string, visible: boolean) => void;
  onShowAll: () => void;
  onHideAll: () => void;
  onIsolatePart: (partId: string | null) => void;
  selectedPart: BrainPart | null;
  onClearSelection: () => void;
}

const LayeredAnatomyPanel: React.FC<LayeredAnatomyPanelProps> = ({
  state,
  onToggleMode,
  onTogglePart,
  onShowAll,
  onHideAll,
  onIsolatePart,
  selectedPart,
  onClearSelection
}) => {
  if (!state.enabled) {
    return (
      <div className="mb-4 bg-zinc-900 rounded-lg p-3 border border-zinc-800">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Brain size={16} className="text-emerald-400" />
            <h3 className="text-xs font-semibold text-white">Layered Anatomy</h3>
          </div>
          <button
            onClick={() => onToggleMode(true)}
            className="text-xs px-2 py-1 rounded bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/30 transition"
          >
            Enable
          </button>
        </div>
        <p className="text-[10px] text-zinc-500 leading-relaxed">
          Explore brain structures layer by layer. Toggle visibility of cortex, cerebellum, brainstem, ventricles, and lobes.
        </p>
      </div>
    );
  }

  const majorStructures = state.parts.filter(p => 
    ['Cortex', 'Cerebellum', 'Brainstem', 'Ventricles'].includes(p.id)
  );
  const lobes = state.parts.filter(p => 
    ['Frontal', 'Parietal', 'Temporal', 'Occipital'].includes(p.id)
  );
  const functionalAreas = state.parts.filter(p => 
    ['VisualArea', 'MotorArea', 'BrocaArea', 'AuditoryArea', 'WernickeArea', 
     'SensoryArea', 'AssociationArea', 'EmotionalArea', 'OlfactoryArea', 
     'HigherMentalFunctions'].includes(p.id)
  );

  return (
    <div className="mb-4 bg-zinc-900 rounded-lg p-3 border border-zinc-800">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Brain size={16} className="text-emerald-400" />
          <h3 className="text-xs font-semibold text-white">Layered Anatomy</h3>
        </div>
        <button
          onClick={() => onToggleMode(false)}
          className="text-zinc-500 hover:text-white transition"
          title="Disable layered mode"
        >
          <X size={14} />
        </button>
      </div>

      {/* Quick Actions */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={onShowAll}
          className="flex-1 text-[10px] px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition"
        >
          Show All
        </button>
        <button
          onClick={onHideAll}
          className="flex-1 text-[10px] px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition"
        >
          Hide All
        </button>
      </div>

      {/* Isolation Mode Indicator */}
      {state.isolatedPartId && (
        <div className="mb-3 p-2 bg-emerald-900/30 border border-emerald-700/50 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Focus size={14} className="text-emerald-400" />
              <span className="text-xs text-emerald-300">
                Isolated: {state.parts.find(p => p.id === state.isolatedPartId)?.name || 'Unknown'}
              </span>
            </div>
            <button
              onClick={() => onIsolatePart(null)}
              className="text-xs px-2 py-1 rounded bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/30 transition"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Major Structures */}
      <div className="mb-3">
        <div className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider mb-2">Major Structures</div>
        <div className="space-y-1.5">
          {majorStructures.map(part => (
            <label
              key={part.id}
              className="flex items-center gap-2 p-1.5 rounded hover:bg-zinc-800/50 cursor-pointer transition group"
            >
              <input
                type="checkbox"
                checked={part.visible}
                onChange={(e) => onTogglePart(part.id, e.target.checked)}
                className="w-3 h-3 accent-emerald-600 cursor-pointer"
              />
              <div
                className="w-3 h-3 rounded border border-zinc-700"
                style={{ backgroundColor: part.color }}
              />
              <span className="flex-1 text-xs text-zinc-300 group-hover:text-white transition">
                {part.name}
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onIsolatePart(state.isolatedPartId === part.id ? null : part.id);
                  }}
                  className={`opacity-0 group-hover:opacity-100 transition ${
                    state.isolatedPartId === part.id
                      ? 'opacity-100 text-emerald-400'
                      : 'text-zinc-500 hover:text-emerald-400'
                  }`}
                  title={state.isolatedPartId === part.id ? 'Clear isolation' : 'Isolate this part'}
                >
                  <Focus size={12} />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    // Info will be shown via hover/click on 3D model
                  }}
                  className="opacity-0 group-hover:opacity-100 transition text-zinc-500 hover:text-emerald-400"
                  title="Click brain part in 3D view for details"
                >
                  <Info size={12} />
                </button>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Lobes */}
      {lobes.length > 0 && (
        <div className="mb-3">
          <div className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider mb-2">Lobes</div>
          <div className="space-y-1.5">
            {lobes.map(part => (
              <label
                key={part.id}
                className="flex items-center gap-2 p-1.5 rounded hover:bg-zinc-800/50 cursor-pointer transition group"
              >
                <input
                  type="checkbox"
                  checked={part.visible}
                  onChange={(e) => onTogglePart(part.id, e.target.checked)}
                  className="w-3 h-3 accent-emerald-600 cursor-pointer"
                />
                <div
                  className="w-3 h-3 rounded border border-zinc-700"
                  style={{ backgroundColor: part.color }}
                />
                <span className="flex-1 text-xs text-zinc-300 group-hover:text-white transition">
                  {part.name}
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                  className="opacity-0 group-hover:opacity-100 transition text-zinc-500 hover:text-emerald-400"
                  title="Click brain part in 3D view for details"
                >
                  <Info size={12} />
                </button>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Functional Areas */}
      {functionalAreas.length > 0 && (
        <div className="mb-3">
          <div className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider mb-2">Functional Areas</div>
          <div className="space-y-1.5 max-h-48 overflow-y-auto">
            {functionalAreas.map(part => (
              <label
                key={part.id}
                className="flex items-center gap-2 p-1.5 rounded hover:bg-zinc-800/50 cursor-pointer transition group"
              >
                <input
                  type="checkbox"
                  checked={part.visible}
                  onChange={(e) => onTogglePart(part.id, e.target.checked)}
                  className="w-3 h-3 accent-emerald-600 cursor-pointer"
                />
                <div
                  className="w-3 h-3 rounded border border-zinc-700"
                  style={{ backgroundColor: part.color }}
                />
                <span className="flex-1 text-xs text-zinc-300 group-hover:text-white transition">
                  {part.name}
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                  }}
                  className="opacity-0 group-hover:opacity-100 transition text-zinc-500 hover:text-emerald-400"
                  title="Click brain part in 3D view for details"
                >
                  <Info size={12} />
                </button>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Selected Part Info Card */}
      {selectedPart && (
        <div className="mt-3 p-2.5 bg-emerald-950/30 border border-emerald-800/50 rounded-lg">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded border border-emerald-600"
                style={{ backgroundColor: selectedPart.color }}
              />
              <h4 className="text-xs font-semibold text-emerald-300">{selectedPart.name}</h4>
            </div>
            <button
              onClick={onClearSelection}
              className="text-zinc-500 hover:text-white transition"
            >
              <X size={12} />
            </button>
          </div>
          <p className="text-[10px] text-zinc-300 leading-relaxed mb-1">
            {selectedPart.description}
          </p>
          {selectedPart.function && (
            <p className="text-[10px] text-zinc-400 leading-relaxed">
              <span className="font-medium text-zinc-500">Function:</span> {selectedPart.function}
            </p>
          )}
          {selectedPart.location && (
            <p className="text-[10px] text-zinc-400 leading-relaxed mt-1">
              <span className="font-medium text-zinc-500">Location:</span> {selectedPart.location}
            </p>
          )}
        </div>
      )}

      {/* Info Message */}
      {!selectedPart && (
        <div className="mt-3 p-2 bg-zinc-800/50 rounded text-[10px] text-zinc-500 italic text-center">
          Click or hover over brain parts in the 3D view to learn more
        </div>
      )}
    </div>
  );
};

export default LayeredAnatomyPanel;
