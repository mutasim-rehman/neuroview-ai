export enum ViewType {
  AXIAL = 'Axial',
  CORONAL = 'Coronal',
  SAGITTAL = 'Sagittal',
}

export enum RenderMode {
  SLICE = '2D Slice',
  VOLUME = '3D Volume',
  QUAD = 'Quad View'
}

export enum VolumeRenderStyle {
  MIP = 'X-Ray (MIP)',
  ISO = 'Surface (Iso)',
  VOL = 'Volumetric'
}

export enum ColorMap {
  GRAY = 'Grayscale',
  HOT = 'Hot Iron',
  COOL = 'Cool Blue',
  RAINBOW = 'Rainbow',
  ANATOMY = 'Anatomy',
  DENSITY = 'Density Heatmap'
}

export enum ToolMode {
  POINTER = 'Pointer',
  MEASURE = 'Ruler',
  ANNOTATE = 'Marker'
}

export enum TissuePreset {
  CUSTOM = 'Custom',
  SKIN = 'Skin',
  SOFT_TISSUE = 'Soft Tissue',
  BONE = 'Bone',
  VESSELS = 'Vessels'
}

export interface ROIStats {
  x: number;
  y: number;
  z: number;
  value: number; // Raw intensity
  densityLabel: string; // "High", "Low", "Soft Tissue" based on value
}

export interface Measurement {
  id: string;
  viewType: ViewType;
  sliceIndex: number;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  lengthMm: number;
}

export interface Annotation {
  id: string;
  viewType: ViewType;
  sliceIndex: number;
  x: number;
  y: number;
  text: string;
}

export interface NiftiHeader {
  dims: number[];
  dataType: number;
  littleEndian: boolean;
  pixDims: number[];
  affine?: number[][];
}

export interface NiftiData {
  header: NiftiHeader;
  image: ArrayBuffer | TypedArray;
  min: number;
  max: number;
}

// Helper type for TypedArrays
export type TypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export interface AnalysisResult {
  text: string;
  loading: boolean;
  error?: string;
}

// Multi-Volume Support
export enum OverlayMode {
  BLEND = 'Blend',
  DIFFERENCE = 'Difference',
  MULTIPLY = 'Multiply',
  ADDITIVE = 'Additive',
  MAXIMUM = 'Maximum',
  MINIMUM = 'Minimum'
}

export interface VolumeMetadata {
  id: string;
  name: string;
  color: string; // Hex color for this volume
  visible: boolean;
  opacity: number; // 0-1
  colorMap: ColorMap;
  window: number; // Window width for window/level
  level: number; // Window level
}

export interface VolumeData extends NiftiData {
  metadata: VolumeMetadata;
  isTimeSeries?: boolean;
  timePoints?: number; // For 4D volumes
}

// Time-Series Support
export interface TimeSeriesState {
  currentFrame: number;
  totalFrames: number;
  isPlaying: boolean;
  playbackSpeed: number; // frames per second
  loop: boolean;
}

// Segmentation & AI Features
export interface SegmentationMask {
  id: string;
  volumeId: string;
  name: string;
  color: string;
  opacity: number;
  visible: boolean;
  data: TypedArray; // Mask data (same dimensions as volume)
  regions: SegmentationRegion[];
}

export interface SegmentationRegion {
  id: string;
  label: string;
  color: string;
  voxelCount: number;
  volumeMm3: number;
  centroid: { x: number; y: number; z: number };
}

export interface AnomalyDetection {
  id: string;
  volumeId: string;
  sliceIndex: number;
  viewType: ViewType;
  regions: AnomalyRegion[];
  confidence: number; // Overall confidence 0-1
  timestamp: number;
}

export interface AnomalyRegion {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  description: string;
  severity: 'low' | 'medium' | 'high';
}

// Enhanced Analysis
export interface EnhancedAnalysisResult extends AnalysisResult {
  structures: IdentifiedStructure[];
  measurements: AutomatedMeasurement[];
  anomalyScore?: number;
  confidence: number;
  recommendations?: string[];
}

export interface IdentifiedStructure {
  name: string;
  confidence: number;
  location: { x: number; y: number; z: number };
  description: string;
}

export interface AutomatedMeasurement {
  id: string;
  type: 'distance' | 'area' | 'volume' | 'intensity';
  value: number;
  unit: string;
  structure: string;
  location: { x: number; y: number; z: number };
}

// Window/Level Presets
export enum WindowLevelPreset {
  BRAIN = 'Brain',
  BONE = 'Bone',
  LUNG = 'Lung',
  SOFT_TISSUE = 'Soft Tissue',
  ABDOMEN = 'Abdomen',
  CUSTOM = 'Custom'
}

export interface WindowLevel {
  window: number;
  level: number;
  preset: WindowLevelPreset;
}

// Render Quality
export enum RenderQuality {
  FAST = 'Fast',
  MEDIUM = 'Medium',
  HIGH = 'High',
  ULTRA = 'Ultra'
}

// Histogram Data
export interface HistogramData {
  bins: number[];
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
}
