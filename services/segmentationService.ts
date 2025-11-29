import { GoogleGenAI } from "@google/genai";
import { SegmentationMask, SegmentationRegion, VolumeData, TypedArray } from '../types';

const getClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey });
};

// Segment structures in a medical image using AI
export const segmentStructures = async (
  base64Image: string,
  viewType: string,
  sliceIndex: number
): Promise<SegmentationRegion[]> => {
  try {
    const ai = getClient();
    const cleanBase64 = base64Image.split(',')[1] || base64Image;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: cleanBase64
            }
          },
          {
            text: `Analyze this medical imaging slice and identify anatomical structures or regions of interest.

View: ${viewType}, Slice: ${sliceIndex}

Provide a JSON array of identified structures with:
{
  "regions": [
    {
      "label": "structure name (e.g., 'Brain', 'Ventricle', 'Lesion')",
      "description": "brief description",
      "approximateLocation": {
        "x": approximate X coordinate (0-100%),
        "y": approximate Y coordinate (0-100%),
        "z": slice index
      },
      "estimatedSize": "small|medium|large",
      "confidence": 0.0-1.0
    }
  ]
}

Focus on:
- Major anatomical structures (brain regions, organs, vessels)
- Any visible lesions or abnormalities
- Regions with distinct intensity patterns
- Symmetrical structures (left/right)

Return only valid JSON.`
          }
        ]
      }
    });

    const text = response.text || "";
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    
    if (jsonMatch) {
      const data = JSON.parse(jsonMatch[0]);
      const regions: SegmentationRegion[] = (data.regions || []).map((r: any, idx: number) => {
        // Estimate voxel count based on size
        let voxelCount = 1000;
        if (r.estimatedSize === 'small') voxelCount = 500;
        else if (r.estimatedSize === 'medium') voxelCount = 2000;
        else if (r.estimatedSize === 'large') voxelCount = 5000;

        // Estimate volume (assuming 1mm³ per voxel for simplicity)
        const volumeMm3 = voxelCount;

        return {
          id: `region-${Date.now()}-${idx}`,
          label: r.label || `Region ${idx + 1}`,
          color: getColorForLabel(r.label),
          voxelCount,
          volumeMm3,
          centroid: {
            x: r.approximateLocation?.x || 50,
            y: r.approximateLocation?.y || 50,
            z: r.approximateLocation?.z || sliceIndex
          }
        };
      });
      
      return regions;
    }
    
    return [];
  } catch (error) {
    console.error("Segmentation Error:", error);
    return [];
  }
};

// Generate a segmentation mask from regions
export const createSegmentationMask = (
  volume: VolumeData,
  regions: SegmentationRegion[],
  name: string
): SegmentationMask => {
  const dims = volume.header.dims;
  const xDim = dims[1];
  const yDim = dims[2];
  const zDim = dims[3];
  const totalVoxels = xDim * yDim * zDim;
  
  // Create mask data (Uint8Array where 0 = background, >0 = region ID)
  const maskData = new Uint8Array(totalVoxels);
  
  // For each region, mark approximate area in mask
  // This is a simplified implementation - in production, you'd use actual segmentation algorithms
  regions.forEach((region, regionIdx) => {
    const regionId = regionIdx + 1;
    const centerX = Math.floor((region.centroid.x / 100) * xDim);
    const centerY = Math.floor((region.centroid.y / 100) * yDim);
    const centerZ = Math.floor(region.centroid.z);
    
    // Create a simple circular/spherical region around centroid
    const radius = Math.ceil(Math.sqrt(region.voxelCount / Math.PI));
    
    for (let z = Math.max(0, centerZ - radius); z <= Math.min(zDim - 1, centerZ + radius); z++) {
      for (let y = Math.max(0, centerY - radius); y <= Math.min(yDim - 1, centerY + radius); y++) {
        for (let x = Math.max(0, centerX - radius); x <= Math.min(xDim - 1, centerX + radius); x++) {
          const dx = x - centerX;
          const dy = y - centerY;
          const dz = z - centerZ;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          
          if (dist <= radius) {
            const idx = x + (y * xDim) + (z * xDim * yDim);
            if (idx >= 0 && idx < totalVoxels) {
              maskData[idx] = regionId;
            }
          }
        }
      }
    }
  });
  
  return {
    id: `mask-${Date.now()}`,
    volumeId: volume.metadata.id,
    name,
    color: '#10b981',
    opacity: 0.5,
    visible: true,
    data: maskData,
    regions
  };
};

// Get color for a structure label
const getColorForLabel = (label: string): string => {
  const labelLower = label.toLowerCase();
  
  // Brain regions
  if (labelLower.includes('ventricle')) return '#3b82f6'; // Blue
  if (labelLower.includes('brain') || labelLower.includes('cortex')) return '#10b981'; // Green
  if (labelLower.includes('white matter')) return '#fbbf24'; // Yellow
  if (labelLower.includes('gray matter')) return '#ef4444'; // Red
  
  // Organs
  if (labelLower.includes('heart')) return '#ef4444'; // Red
  if (labelLower.includes('lung')) return '#60a5fa'; // Light blue
  if (labelLower.includes('liver')) return '#34d399'; // Teal
  if (labelLower.includes('kidney')) return '#a78bfa'; // Purple
  
  // Pathologies
  if (labelLower.includes('lesion') || labelLower.includes('tumor')) return '#f472b6'; // Pink
  if (labelLower.includes('edema')) return '#fbbf24'; // Yellow
  if (labelLower.includes('hemorrhage')) return '#dc2626'; // Dark red
  
  // Default
  return '#10b981'; // Green
};

// Analyze slice and create segmentation mask
export const analyzeAndSegment = async (
  base64Image: string,
  volume: VolumeData,
  viewType: string,
  sliceIndex: number
): Promise<SegmentationMask | null> => {
  try {
    const regions = await segmentStructures(base64Image, viewType, sliceIndex);
    
    if (regions.length === 0) {
      return null;
    }
    
    return createSegmentationMask(volume, regions, `Segmentation - ${viewType} Slice ${sliceIndex}`);
  } catch (error) {
    console.error("Analyze and Segment Error:", error);
    return null;
  }
};

