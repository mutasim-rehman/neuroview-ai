/**
 * Service for brain health prediction using the trained AI model.
 * Connects to the Python Flask backend API for inference.
 */

export interface BrainHealthPrediction {
  prediction: 'healthy' | 'defect';
  confidence: number;
  anomaly_score: number;
  error_metrics?: {
    mse: number;
    mae: number;
    max_error: number;
  };
  feature_vector?: number[];
  error?: string;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Check if the API server is healthy and model is loaded.
 */
export const checkApiHealth = async (): Promise<{
  status: string;
  model_loaded: boolean;
  device?: string;
}> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`API health check failed: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('API health check failed:', error);
    throw new Error('Unable to connect to prediction server. Please ensure the API server is running.');
  }
};

/**
 * Predict brain health from a NIfTI file.
 * @param file The NIfTI file to analyze
 * @returns Prediction results
 */
export const predictFromFile = async (file: File): Promise<BrainHealthPrediction> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `Prediction failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error: any) {
    console.error('Prediction error:', error);
    throw error;
  }
};

/**
 * Predict brain health from volume data array.
 * @param volumeData 3D array of volume data
 * @returns Prediction results
 */
export const predictFromVolume = async (
  volumeData: Float32Array | number[][][],
  shape: [number, number, number]
): Promise<BrainHealthPrediction> => {
  try {
    // Convert volume data to nested array if needed
    let volumeArray: number[][][];
    
    if (volumeData instanceof Float32Array) {
      // Reshape flat array to 3D
      const [depth, height, width] = shape;
      volumeArray = [];
      
      for (let d = 0; d < depth; d++) {
        volumeArray[d] = [];
        for (let h = 0; h < height; h++) {
          volumeArray[d][h] = [];
          for (let w = 0; w < width; w++) {
            const idx = d * height * width + h * width + w;
            volumeArray[d][h][w] = volumeData[idx];
          }
        }
      }
    } else {
      volumeArray = volumeData as number[][][];
    }

    const response = await fetch(`${API_BASE_URL}/predict_from_array`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        volume: volumeArray,
        shape: shape,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `Prediction failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result;
  } catch (error: any) {
    console.error('Prediction error:', error);
    throw error;
  }
};

/**
 * Predict brain health from NIfTI volume data (VolumeData type).
 * @param volumeData Volume data object from niftiLoader
 * @returns Prediction results
 */
export const predictFromVolumeData = async (
  volumeData: { image: ArrayLike<number>; header: { dims: number[] } }
): Promise<BrainHealthPrediction> => {
  try {
    const { image, header } = volumeData;
    const dims = header.dims;
    
    // Extract shape (skip first dim which is usually metadata)
    const shape: [number, number, number] = [
      dims[1] || 128,
      dims[2] || 128,
      dims[3] || 128,
    ];

    // Convert to Float32Array
    const imageArray = new Float32Array(image.length);
    for (let i = 0; i < image.length; i++) {
      imageArray[i] = Number(image[i]);
    }

    // Resize if necessary (model expects 128x128x128)
    // For now, we'll let the backend handle resizing
    // But we could do it here for better performance

    return await predictFromVolume(imageArray, shape);
  } catch (error: any) {
    console.error('Prediction error:', error);
    throw error;
  }
};
