import * as THREE from 'three';
import { VolumeData, TypedArray } from '../types';

/**
 * Tissue types based on MRI intensity values (T1-weighted)
 */
export enum TissueType {
  BACKGROUND = 0,
  CSF = 1,           // Cerebrospinal fluid (dark/low intensity)
  GRAY_MATTER = 2,   // Gray matter (medium intensity)
  WHITE_MATTER = 3,  // White matter (bright/high intensity)
  BONE = 4,          // Bone/skull (very bright)
  UNKNOWN = 5
}

/**
 * Anatomical region IDs for segmentation mask
 */
export enum RegionId {
  BACKGROUND = 0,
  VENTRICLES = 1,      // CSF-filled ventricles
  CORTEX = 2,          // Gray matter cortex
  WHITE_MATTER = 3,    // White matter tracts
  CEREBELLUM = 4,
  BRAINSTEM = 5,
  FRONTAL_LOBE = 6,
  PARIETAL_LOBE = 7,
  TEMPORAL_LOBE = 8,
  OCCIPITAL_LOBE = 9
}

/**
 * Intensity-based tissue segmentation for T1-weighted MRI
 * Uses histogram analysis to determine tissue type thresholds
 */
export interface TissueSegmentationResult {
  mask: Uint8Array; // Segmentation mask (RegionId values)
  tissueMask: Uint8Array; // Tissue type mask (TissueType values)
  thresholds: {
    csf: number;
    grayMatter: number;
    whiteMatter: number;
  };
  stats: {
    csfVoxels: number;
    grayMatterVoxels: number;
    whiteMatterVoxels: number;
    totalBrainVoxels: number;
  };
}

/**
 * Perform intensity-based tissue segmentation
 * Uses histogram analysis to find optimal thresholds for tissue types
 */
export const segmentTissues = (
  volume: VolumeData,
  options?: {
    csfThreshold?: number;      // Manual CSF threshold (0-1 normalized)
    grayMatterThreshold?: number; // Manual gray matter threshold
    whiteMatterThreshold?: number; // Manual white matter threshold
    useHistogramAnalysis?: boolean; // Auto-detect thresholds from histogram
  }
): TissueSegmentationResult => {
  const { header, image, min, max } = volume;
  const { dims } = header;
  const xDim = dims[1];
  const yDim = dims[2];
  const zDim = dims[3];
  const totalVoxels = xDim * yDim * zDim;

  const rawData = image as TypedArray;
  const range = max - min;

  // Normalize data to 0-1
  const normalizedData = new Float32Array(totalVoxels);
  for (let i = 0; i < totalVoxels; i++) {
    normalizedData[i] = (rawData[i] - min) / range;
  }

  // Determine thresholds
  let csfThreshold: number;
  let grayMatterThreshold: number;
  let whiteMatterThreshold: number;

  if (options?.useHistogramAnalysis !== false) {
    // Auto-detect thresholds using histogram analysis
    const thresholds = analyzeHistogram(normalizedData);
    csfThreshold = options?.csfThreshold ?? thresholds.csf;
    grayMatterThreshold = options?.grayMatterThreshold ?? thresholds.grayMatter;
    whiteMatterThreshold = options?.whiteMatterThreshold ?? thresholds.whiteMatter;
  } else {
    // Use manual thresholds or defaults
    csfThreshold = options?.csfThreshold ?? 0.25;
    grayMatterThreshold = options?.grayMatterThreshold ?? 0.45;
    whiteMatterThreshold = options?.whiteMatterThreshold ?? 0.65;
  }

  // Create tissue type mask
  const tissueMask = new Uint8Array(totalVoxels);
  let csfVoxels = 0;
  let grayMatterVoxels = 0;
  let whiteMatterVoxels = 0;

  for (let i = 0; i < totalVoxels; i++) {
    const intensity = normalizedData[i];
    
    if (intensity < csfThreshold) {
      tissueMask[i] = TissueType.CSF;
      csfVoxels++;
    } else if (intensity < grayMatterThreshold) {
      tissueMask[i] = TissueType.GRAY_MATTER;
      grayMatterVoxels++;
    } else if (intensity < whiteMatterThreshold) {
      tissueMask[i] = TissueType.WHITE_MATTER;
      whiteMatterVoxels++;
    } else {
      tissueMask[i] = TissueType.BONE; // Very bright = bone/skull
    }
  }

  // Create anatomical region mask based on tissue types + spatial location
  const regionMask = createAnatomicalRegions(
    tissueMask,
    normalizedData,
    xDim,
    yDim,
    zDim
  );

  return {
    mask: regionMask,
    tissueMask,
    thresholds: {
      csf: csfThreshold,
      grayMatter: grayMatterThreshold,
      whiteMatter: whiteMatterThreshold,
    },
    stats: {
      csfVoxels,
      grayMatterVoxels,
      whiteMatterVoxels,
      totalBrainVoxels: csfVoxels + grayMatterVoxels + whiteMatterVoxels,
    },
  };
};

/**
 * Analyze histogram to find optimal tissue type thresholds
 * Uses Otsu-like method to find peaks and valleys
 */
function analyzeHistogram(data: Float32Array): {
  csf: number;
  grayMatter: number;
  whiteMatter: number;
} {
  // Create histogram (256 bins)
  const bins = 256;
  const histogram = new Array(bins).fill(0);
  const step = Math.max(1, Math.floor(data.length / 100000)); // Sample for performance

  for (let i = 0; i < data.length; i += step) {
    const bin = Math.floor(data[i] * (bins - 1));
    histogram[Math.max(0, Math.min(bins - 1, bin))]++;
  }

  // Find peaks (modes) for each tissue type
  // CSF: low intensity peak
  // Gray matter: medium intensity peak
  // White matter: high intensity peak

  // Find CSF threshold (first significant peak in low range)
  let csfThreshold = 0.2;
  let maxCount = 0;
  for (let i = 0; i < bins * 0.3; i++) {
    if (histogram[i] > maxCount) {
      maxCount = histogram[i];
      csfThreshold = i / bins;
    }
  }
  // Add margin - CSF is typically darker than the peak
  csfThreshold = Math.max(0.15, csfThreshold - 0.05);

  // Find white matter threshold (peak in high range)
  let whiteMatterThreshold = 0.7;
  maxCount = 0;
  for (let i = Math.floor(bins * 0.5); i < bins; i++) {
    if (histogram[i] > maxCount) {
      maxCount = histogram[i];
      whiteMatterThreshold = i / bins;
    }
  }
  // White matter is typically brighter than the peak
  whiteMatterThreshold = Math.min(0.85, whiteMatterThreshold + 0.1);

  // Gray matter threshold is between CSF and white matter
  const grayMatterThreshold = (csfThreshold + whiteMatterThreshold) / 2;

  return {
    csf: csfThreshold,
    grayMatter: grayMatterThreshold,
    whiteMatter: whiteMatterThreshold,
  };
}

/**
 * Create anatomical region mask from tissue types and spatial location
 */
function createAnatomicalRegions(
  tissueMask: Uint8Array,
  intensityData: Float32Array,
  xDim: number,
  yDim: number,
  zDim: number
): Uint8Array {
  const totalVoxels = xDim * yDim * zDim;
  const regionMask = new Uint8Array(totalVoxels);

  const getIndex = (x: number, y: number, z: number) =>
    x + y * xDim + z * xDim * yDim;

  for (let z = 0; z < zDim; z++) {
    for (let y = 0; y < yDim; y++) {
      for (let x = 0; x < xDim; x++) {
        const idx = getIndex(x, y, z);
        const tissueType = tissueMask[idx];
        const intensity = intensityData[idx];

        // Normalized position (0-1)
        const nx = x / (xDim - 1);
        const ny = y / (yDim - 1);
        const nz = z / (zDim - 1);

        const center = { x: 0.5, y: 0.5, z: 0.5 };
        const distFromCenter = Math.sqrt(
          Math.pow(nx - center.x, 2) +
          Math.pow(ny - center.y, 2) +
          Math.pow(nz - center.z, 2)
        );

        // VENTRICLES: CSF in central regions
        if (tissueType === TissueType.CSF && distFromCenter < 0.25) {
          regionMask[idx] = RegionId.VENTRICLES;
        }
        // CORTEX: Gray matter in outer shell
        else if (tissueType === TissueType.GRAY_MATTER && distFromCenter > 0.35) {
          regionMask[idx] = RegionId.CORTEX;
        }
        // WHITE MATTER: White matter tracts
        else if (tissueType === TissueType.WHITE_MATTER) {
          regionMask[idx] = RegionId.WHITE_MATTER;
        }
        // CEREBELLUM: Posterior inferior, medium intensity
        else if (nz > 0.55 && ny < 0.35 && distFromCenter > 0.15) {
          regionMask[idx] = RegionId.CEREBELLUM;
        }
        // BRAINSTEM: Inferior center, small structure
        else if (ny < 0.25 && nz > 0.35 && nz < 0.65 && distFromCenter < 0.15) {
          regionMask[idx] = RegionId.BRAINSTEM;
        }
        // LOBES based on spatial location + tissue type
        else if (tissueType === TissueType.GRAY_MATTER || tissueType === TissueType.WHITE_MATTER) {
          // FRONTAL LOBE (anterior)
          if (nz < 0.45 && ny > 0.35) {
            regionMask[idx] = RegionId.FRONTAL_LOBE;
          }
          // OCCIPITAL LOBE (posterior)
          else if (nz > 0.6 && ny > 0.3 && ny < 0.7) {
            regionMask[idx] = RegionId.OCCIPITAL_LOBE;
          }
          // TEMPORAL LOBE (lateral)
          else if ((nx < 0.3 || nx > 0.7) && ny > 0.3) {
            regionMask[idx] = RegionId.TEMPORAL_LOBE;
          }
          // PARIETAL LOBE (superior, middle)
          else if (ny > 0.5 && nz > 0.4 && nz < 0.6) {
            regionMask[idx] = RegionId.PARIETAL_LOBE;
          }
          // Default to cortex if gray matter
          else if (tissueType === TissueType.GRAY_MATTER) {
            regionMask[idx] = RegionId.CORTEX;
          }
          // Default to white matter
          else {
            regionMask[idx] = RegionId.WHITE_MATTER;
          }
        }
        // Background/unknown
        else {
          regionMask[idx] = RegionId.BACKGROUND;
        }
      }
    }
  }

  return regionMask;
}

/**
 * Convert segmentation mask to 3D texture data for shader
 */
export const createSegmentationTexture = (
  mask: Uint8Array,
  xDim: number,
  yDim: number,
  zDim: number
): THREE.Data3DTexture => {
  const texture = new THREE.Data3DTexture(mask, xDim, yDim, zDim);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.NearestFilter; // No interpolation for discrete labels
  texture.magFilter = THREE.NearestFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
};
