import * as nifti from 'nifti-reader-js';
import pako from 'pako';
import { NiftiData, TypedArray, VolumeData, VolumeMetadata } from '../types';

export const isCompressed = (buffer: ArrayBuffer): boolean => {
  const header = new Uint8Array(buffer.slice(0, 2));
  return header[0] === 0x1f && header[1] === 0x8b;
};

export const decompress = (buffer: ArrayBuffer): ArrayBuffer => {
  try {
    const inflated = pako.inflate(new Uint8Array(buffer));
    return inflated.buffer;
  } catch (err) {
    console.error("Decompression failed", err);
    throw new Error("Failed to decompress GZIP file.");
  }
};

export const parseNifti = (data: ArrayBuffer): NiftiData | null => {
  let rawData = data;

  if (isCompressed(data)) {
    rawData = decompress(data);
  }

  if (!nifti.isNIFTI(rawData)) {
    return null;
  }

  const header = nifti.readHeader(rawData);
  let image = nifti.readImage(header, rawData);

  // Determine min/max for contrast scaling
  let min = Infinity;
  let max = -Infinity;
  
  // Convert ArrayBuffer to TypedArray for iteration if needed, 
  // though readImage usually returns ArrayBuffer.
  // We need to cast it to the correct type based on header.datatypeCode
  
  let typedImage: TypedArray;

  // Basic type mapping based on NIfTI standard
  switch (header.datatypeCode) {
    case nifti.NIFTI1.TYPE_UINT8:
      typedImage = new Uint8Array(image);
      break;
    case nifti.NIFTI1.TYPE_INT16:
      typedImage = new Int16Array(image);
      break;
    case nifti.NIFTI1.TYPE_INT32:
      typedImage = new Int32Array(image);
      break;
    case nifti.NIFTI1.TYPE_FLOAT32:
      typedImage = new Float32Array(image);
      break;
    case nifti.NIFTI1.TYPE_FLOAT64:
      typedImage = new Float64Array(image);
      break;
    case nifti.NIFTI1.TYPE_INT8:
      typedImage = new Int8Array(image);
      break;
    case nifti.NIFTI1.TYPE_UINT16:
      typedImage = new Uint16Array(image);
      break;
    case nifti.NIFTI1.TYPE_UINT32:
      typedImage = new Uint32Array(image);
      break;
    default:
        // Fallback or unsupported
        console.warn("Unsupported data type code:", header.datatypeCode);
        typedImage = new Uint8Array(image); // Attempt to treat as bytes
  }

  // Calculate stats for auto-contrast
  // Sampling strategy for performance on large files
  const step = Math.max(1, Math.floor(typedImage.length / 10000));
  for (let i = 0; i < typedImage.length; i += step) {
    const val = typedImage[i];
    if (val < min) min = val;
    if (val > max) max = val;
  }

  // Safety check if range is zero
  if (min === max) {
      max = min + 255; 
  }

  return {
    header: {
      dims: header.dims,
      dataType: header.datatypeCode,
      littleEndian: header.littleEndian,
      pixDims: header.pixDims,
      affine: header.affine,
    },
    image: typedImage,
    min,
    max,
  };
};

// Check if volume is 4D (time-series)
export const isTimeSeries = (data: NiftiData): boolean => {
  return data.header.dims.length >= 5 && data.header.dims[4] > 1;
};

// Get time point count for 4D volumes
export const getTimePointCount = (data: NiftiData): number => {
  if (data.header.dims.length >= 5) {
    return data.header.dims[4];
  }
  return 1;
};

// Extract a single time point from 4D volume
export const extractTimePoint = (data: NiftiData, timePoint: number): NiftiData | null => {
  if (!isTimeSeries(data)) {
    return data; // Return original if not time-series
  }

  const dims = data.header.dims;
  const xDim = dims[1];
  const yDim = dims[2];
  const zDim = dims[3];
  const tDim = dims[4] || 1;
  
  if (timePoint < 0 || timePoint >= tDim) {
    return null;
  }

  const voxelsPerTimePoint = xDim * yDim * zDim;
  const startIndex = timePoint * voxelsPerTimePoint;
  const endIndex = startIndex + voxelsPerTimePoint;

  const typedImage = data.image as TypedArray;
  const timePointData = typedImage.slice(startIndex, endIndex);

  // Calculate min/max for this time point
  let min = Infinity;
  let max = -Infinity;
  const step = Math.max(1, Math.floor(timePointData.length / 10000));
  for (let i = 0; i < timePointData.length; i += step) {
    const val = timePointData[i];
    if (val < min) min = val;
    if (val > max) max = val;
  }

  if (min === max) {
    max = min + 255;
  }

  return {
    header: {
      dims: [dims[0], xDim, yDim, zDim, 1], // Remove time dimension
      dataType: data.header.dataType,
      littleEndian: data.header.littleEndian,
      pixDims: data.header.pixDims.slice(0, 4), // Remove time pixel dimension
      affine: data.header.affine,
    },
    image: timePointData,
    min,
    max,
  };
};

// Create VolumeData with metadata
export const createVolumeData = (
  niftiData: NiftiData,
  metadata?: Partial<VolumeMetadata>
): VolumeData => {
  const is4D = isTimeSeries(niftiData);
  const defaultMetadata: VolumeMetadata = {
    id: `volume-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    name: metadata?.name || `Volume ${Date.now()}`,
    color: metadata?.color || '#10b981',
    visible: metadata?.visible !== undefined ? metadata.visible : true,
    opacity: metadata?.opacity !== undefined ? metadata.opacity : 1.0,
    colorMap: metadata?.colorMap || 'Anatomy' as any,
    window: niftiData.max - niftiData.min,
    level: (niftiData.max + niftiData.min) / 2,
  };

  return {
    ...niftiData,
    metadata: { ...defaultMetadata, ...metadata },
    isTimeSeries: is4D,
    timePoints: is4D ? getTimePointCount(niftiData) : undefined,
  };
};
