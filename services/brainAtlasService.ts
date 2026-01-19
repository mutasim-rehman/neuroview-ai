import { VolumeData, TypedArray } from '../types';

/**
 * Brain Atlas Service
 * 
 * This service provides anatomical region mapping using standard brain atlases.
 * For MVP, we use simplified spatial heuristics, but this can be extended to:
 * - AAL (Automated Anatomical Labeling) atlas
 * - FreeSurfer parcellation
 * - Custom segmentation masks
 * - AI-based segmentation
 */

export interface AtlasRegion {
  id: string;
  name: string;
  label: number; // Voxel label value in mask
  color: string;
  bounds?: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    minZ: number;
    maxZ: number;
  };
}

/**
 * Generate a segmentation mask for brain regions based on anatomical atlases
 * This is a simplified version - in production, you'd load actual atlas files
 * (like AAL, FreeSurfer, or custom NIfTI label files)
 */
export const generateBrainAtlasMask = (
  volume: VolumeData,
  regionIds: string[]
): { mask: TypedArray; regions: AtlasRegion[] } => {
  const { dims } = volume.header;
  const xDim = dims[1];
  const yDim = dims[2];
  const zDim = dims[3];
  const totalVoxels = xDim * yDim * zDim;
  
  // Create mask (Uint16Array supports up to 65535 regions)
  const mask = new Uint16Array(totalVoxels);
  
  // Define anatomical regions with their approximate spatial boundaries
  // These are based on standard brain anatomy, not perfect but much better than straight lines
  const regionDefinitions: Record<string, AtlasRegion> = {
    'Cortex': {
      id: 'Cortex',
      name: 'Cortex',
      label: 1,
      color: '#10b981',
      bounds: { minX: 0.1, maxX: 0.9, minY: 0.3, maxY: 0.9, minZ: 0.1, maxZ: 0.9 }
    },
    'Cerebellum': {
      id: 'Cerebellum',
      name: 'Cerebellum',
      label: 2,
      color: '#3b82f6',
      bounds: { minX: 0.2, maxX: 0.8, minY: 0.0, maxY: 0.35, minZ: 0.5, maxZ: 0.95 }
    },
    'Brainstem': {
      id: 'Brainstem',
      name: 'Brainstem',
      label: 3,
      color: '#f59e0b',
      bounds: { minX: 0.4, maxX: 0.6, minY: 0.0, maxY: 0.25, minZ: 0.4, maxZ: 0.6 }
    },
    'Ventricles': {
      id: 'Ventricles',
      name: 'Ventricles',
      label: 4,
      color: '#06b6d4',
      bounds: { minX: 0.35, maxX: 0.65, minY: 0.4, maxY: 0.6, minZ: 0.4, maxZ: 0.6 }
    },
    'Frontal': {
      id: 'Frontal',
      name: 'Frontal Lobe',
      label: 5,
      color: '#ef4444',
      bounds: { minX: 0.1, maxX: 0.9, minY: 0.4, maxY: 0.9, minZ: 0.0, maxZ: 0.45 }
    },
    'Parietal': {
      id: 'Parietal',
      name: 'Parietal Lobe',
      label: 6,
      color: '#8b5cf6',
      bounds: { minX: 0.1, maxX: 0.9, minY: 0.5, maxY: 0.9, minZ: 0.4, maxZ: 0.65 }
    },
    'Temporal': {
      id: 'Temporal',
      name: 'Temporal Lobe',
      label: 7,
      color: '#ec4899',
      bounds: { minX: 0.0, maxX: 0.35, minY: 0.2, maxY: 0.6, minZ: 0.3, maxZ: 0.7 }
    },
    'Occipital': {
      id: 'Occipital',
      name: 'Occipital Lobe',
      label: 8,
      color: '#14b8a6',
      bounds: { minX: 0.1, maxX: 0.9, minY: 0.3, maxY: 0.7, minZ: 0.6, maxZ: 0.95 }
    },
    'VisualArea': {
      id: 'VisualArea',
      name: 'Visual Area',
      label: 9,
      color: '#60a5fa',
      bounds: { minX: 0.2, maxX: 0.8, minY: 0.4, maxY: 0.6, minZ: 0.65, maxZ: 0.9 }
    },
    'MotorArea': {
      id: 'MotorArea',
      name: 'Motor Function Area',
      label: 10,
      color: '#fbbf24',
      bounds: { minX: 0.3, maxX: 0.7, minY: 0.5, maxY: 0.65, minZ: 0.1, maxZ: 0.4 }
    },
    'BrocaArea': {
      id: 'BrocaArea',
      name: 'Broca\'s Area',
      label: 11,
      color: '#f97316',
      bounds: { minX: 0.1, maxX: 0.4, minY: 0.35, maxY: 0.5, minZ: 0.15, maxZ: 0.35 }
    },
    'AuditoryArea': {
      id: 'AuditoryArea',
      name: 'Auditory Area',
      label: 12,
      color: '#a855f7',
      bounds: { minX: 0.0, maxX: 0.3, minY: 0.4, maxY: 0.6, minZ: 0.4, maxZ: 0.6 }
    },
    'WernickeArea': {
      id: 'WernickeArea',
      name: 'Wernicke\'s Area',
      label: 13,
      color: '#ec4899',
      bounds: { minX: 0.0, maxX: 0.35, minY: 0.35, maxY: 0.5, minZ: 0.45, maxZ: 0.65 }
    },
    'SensoryArea': {
      id: 'SensoryArea',
      name: 'Sensory Area',
      label: 14,
      color: '#22d3ee',
      bounds: { minX: 0.3, maxX: 0.7, minY: 0.5, maxY: 0.65, minZ: 0.45, maxZ: 0.6 }
    },
    'AssociationArea': {
      id: 'AssociationArea',
      name: 'Association Area',
      label: 15,
      color: '#34d399',
      bounds: { minX: 0.2, maxX: 0.8, minY: 0.4, maxY: 0.6, minZ: 0.3, maxZ: 0.6 }
    },
    'EmotionalArea': {
      id: 'EmotionalArea',
      name: 'Emotional Area',
      label: 16,
      color: '#f43f5e',
      bounds: { minX: 0.4, maxX: 0.6, minY: 0.3, maxY: 0.5, minZ: 0.4, maxZ: 0.6 }
    },
    'OlfactoryArea': {
      id: 'OlfactoryArea',
      name: 'Olfactory Area',
      label: 17,
      color: '#84cc16',
      bounds: { minX: 0.4, maxX: 0.6, minY: 0.2, maxY: 0.4, minZ: 0.3, maxZ: 0.5 }
    },
    'HigherMentalFunctions': {
      id: 'HigherMentalFunctions',
      name: 'Higher Mental Functions',
      label: 18,
      color: '#dc2626',
      bounds: { minX: 0.2, maxX: 0.8, minY: 0.6, maxY: 0.9, minZ: 0.0, maxZ: 0.35 }
    }
  };
  
  // Get regions to include
  const activeRegions = regionIds
    .map(id => regionDefinitions[id])
    .filter(r => r !== undefined);
  
  // Fill mask based on region boundaries
  // Use distance-based weighting for smoother boundaries
  for (let z = 0; z < zDim; z++) {
    for (let y = 0; y < yDim; y++) {
      for (let x = 0; x < xDim; x++) {
        const idx = x + (y * xDim) + (z * xDim * yDim);
        const nx = x / (xDim - 1);
        const ny = y / (yDim - 1);
        const nz = z / (zDim - 1);
        
        // Find which region this voxel belongs to
        let bestRegion: AtlasRegion | null = null;
        let bestScore = 0;
        
        for (const region of activeRegions) {
          if (!region.bounds) continue;
          
          const { minX, maxX, minY, maxY, minZ, maxZ } = region.bounds;
          
          // Check if voxel is within bounds
          if (nx >= minX && nx <= maxX && ny >= minY && ny <= maxY && nz >= minZ && nz <= maxZ) {
            // Calculate distance from center of region (for priority when overlapping)
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const centerZ = (minZ + maxZ) / 2;
            const dist = Math.sqrt(
              Math.pow(nx - centerX, 2) + 
              Math.pow(ny - centerY, 2) + 
              Math.pow(nz - centerZ, 2)
            );
            
            // Closer to center = higher priority
            const score = 1 / (1 + dist * 10);
            if (score > bestScore) {
              bestScore = score;
              bestRegion = region;
            }
          }
        }
        
        if (bestRegion) {
          mask[idx] = bestRegion.label;
        }
      }
    }
  }
  
  return {
    mask,
    regions: activeRegions
  };
};

/**
 * Load a brain atlas from a NIfTI label file
 * This would be used with actual atlas files (AAL, FreeSurfer, etc.)
 */
export const loadAtlasFromFile = async (
  file: File
): Promise<{ mask: TypedArray; regions: AtlasRegion[] } | null> => {
  // TODO: Implement NIfTI label file loading
  // This would parse a standard brain atlas file format
  console.warn('Atlas file loading not yet implemented. Using spatial heuristics.');
  return null;
};

/**
 * Check if a voxel belongs to a specific region
 */
export const isVoxelInRegion = (
  mask: TypedArray,
  x: number,
  y: number,
  z: number,
  xDim: number,
  yDim: number,
  regionLabel: number
): boolean => {
  const idx = x + (y * xDim) + (z * xDim * yDim);
  if (idx < 0 || idx >= mask.length) return false;
  return mask[idx] === regionLabel;
};
