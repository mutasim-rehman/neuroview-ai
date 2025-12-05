/**
 * 3D Gaussian blur utility for medical volume data
 * 
 * This function applies a 3D Gaussian blur to volume data to reduce staircasing
 * (terracing) artifacts caused by anisotropic spacing between slices in medical scans.
 * The blur accounts for different spacing in X, Y, and Z directions.
 */

/**
 * Applies 3D Gaussian blur to volume data
 * @param data - Input volume data as Float32Array
 * @param xDim - X dimension size
 * @param yDim - Y dimension size
 * @param zDim - Z dimension size
 * @param spacingX - Physical spacing in X direction (default: 1.0)
 * @param spacingY - Physical spacing in Y direction (default: 1.0)
 * @param spacingZ - Physical spacing in Z direction (default: 1.0)
 * @param sigma - Blur radius in voxels (default: 1.0, typically 0.5-2.0)
 * @returns Blurred volume data as Float32Array
 */
export function apply3DGaussianBlur(
  data: Float32Array,
  xDim: number,
  yDim: number,
  zDim: number,
  spacingX: number = 1.0,
  spacingY: number = 1.0,
  spacingZ: number = 1.0,
  sigma: number = 1.0
): Float32Array {
  const output = new Float32Array(data.length);
  
  // Calculate sigma in each direction, accounting for spacing
  // Larger spacing means we need more blur in that direction
  const sigmaX = sigma * Math.max(1.0, spacingZ / spacingX);
  const sigmaY = sigma * Math.max(1.0, spacingZ / spacingY);
  const sigmaZ = sigma * Math.max(1.0, Math.max(spacingX, spacingY) / spacingZ);
  
  // Calculate kernel radius (3-sigma rule for 99.7% coverage)
  const radiusX = Math.ceil(3 * sigmaX);
  const radiusY = Math.ceil(3 * sigmaY);
  const radiusZ = Math.ceil(3 * sigmaZ);
  
  // Pre-compute 1D Gaussian kernels for each axis
  const kernelX = computeGaussianKernel1D(radiusX, sigmaX);
  const kernelY = computeGaussianKernel1D(radiusY, sigmaY);
  const kernelZ = computeGaussianKernel1D(radiusZ, sigmaZ);
  
  // Temporary buffer for intermediate results
  const temp = new Float32Array(data.length);
  
  // First pass: blur along X axis
  for (let z = 0; z < zDim; z++) {
    for (let y = 0; y < yDim; y++) {
      for (let x = 0; x < xDim; x++) {
        let sum = 0;
        let weightSum = 0;
        
        for (let kx = -radiusX; kx <= radiusX; kx++) {
          const nx = x + kx;
          if (nx >= 0 && nx < xDim) {
            const idx = getIndex(nx, y, z, xDim, yDim);
            const weight = kernelX[kx + radiusX];
            sum += data[idx] * weight;
            weightSum += weight;
          }
        }
        
        const idx = getIndex(x, y, z, xDim, yDim);
        temp[idx] = weightSum > 0 ? sum / weightSum : data[idx];
      }
    }
  }
  
  // Second pass: blur along Y axis (using temp as input)
  for (let z = 0; z < zDim; z++) {
    for (let y = 0; y < yDim; y++) {
      for (let x = 0; x < xDim; x++) {
        let sum = 0;
        let weightSum = 0;
        
        for (let ky = -radiusY; ky <= radiusY; ky++) {
          const ny = y + ky;
          if (ny >= 0 && ny < yDim) {
            const idx = getIndex(x, ny, z, xDim, yDim);
            const weight = kernelY[ky + radiusY];
            sum += temp[idx] * weight;
            weightSum += weight;
          }
        }
        
        const idx = getIndex(x, y, z, xDim, yDim);
        output[idx] = weightSum > 0 ? sum / weightSum : temp[idx];
      }
    }
  }
  
  // Third pass: blur along Z axis (using output as input, write back to temp)
  for (let z = 0; z < zDim; z++) {
    for (let y = 0; y < yDim; y++) {
      for (let x = 0; x < xDim; x++) {
        let sum = 0;
        let weightSum = 0;
        
        for (let kz = -radiusZ; kz <= radiusZ; kz++) {
          const nz = z + kz;
          if (nz >= 0 && nz < zDim) {
            const idx = getIndex(x, y, nz, xDim, yDim);
            const weight = kernelZ[kz + radiusZ];
            sum += output[idx] * weight;
            weightSum += weight;
          }
        }
        
        const idx = getIndex(x, y, z, xDim, yDim);
        temp[idx] = weightSum > 0 ? sum / weightSum : output[idx];
      }
    }
  }
  
  // Copy final result from temp to output
  output.set(temp);
  
  return output;
}

/**
 * Computes a 1D Gaussian kernel
 */
function computeGaussianKernel1D(radius: number, sigma: number): Float32Array {
  const size = 2 * radius + 1;
  const kernel = new Float32Array(size);
  const twoSigmaSq = 2 * sigma * sigma;
  let sum = 0;
  
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    const value = Math.exp(-(x * x) / twoSigmaSq);
    kernel[i] = value;
    sum += value;
  }
  
  // Normalize
  for (let i = 0; i < size; i++) {
    kernel[i] /= sum;
  }
  
  return kernel;
}

/**
 * Gets the linear index from 3D coordinates
 */
function getIndex(x: number, y: number, z: number, xDim: number, yDim: number): number {
  return z * xDim * yDim + y * xDim + x;
}
