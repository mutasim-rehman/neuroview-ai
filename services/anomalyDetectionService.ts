import { GoogleGenAI } from "@google/genai";
import { AnomalyDetection, AnomalyRegion, ViewType } from '../types';

const getClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey });
};

// Detect anomalies in a medical image slice
export const detectAnomalies = async (
  base64Image: string,
  viewType: ViewType,
  sliceIndex: number
): Promise<AnomalyDetection> => {
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
            text: `Analyze this medical imaging slice for potential anomalies, abnormalities, or areas of concern.

View: ${viewType}, Slice: ${sliceIndex}

IMPORTANT: This is for visualization and educational purposes only. This is NOT a medical diagnosis.

Provide a JSON response with:
{
  "overallConfidence": 0.0-1.0,
  "anomalies": [
    {
      "description": "brief description of the finding",
      "location": {
        "x": approximate X position (0-100%),
        "y": approximate Y position (0-100%),
        "width": approximate width (0-100%),
        "height": approximate height (0-100%)
      },
      "confidence": 0.0-1.0,
      "severity": "low|medium|high",
      "category": "anatomical variant|pathology|artifact|normal variant"
    }
  ]
}

Focus on:
- Asymmetries
- Unusual intensity patterns
- Structural abnormalities
- Masses or lesions
- Edema or fluid collections
- Hemorrhage
- Calcifications
- Artifacts

Be conservative - only flag findings with reasonable confidence. Return only valid JSON.`
          }
        ]
      }
    });

    const text = response.text || "";
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    
    if (jsonMatch) {
      const data = JSON.parse(jsonMatch[0]);
      const regions: AnomalyRegion[] = (data.anomalies || []).map((a: any, idx: number) => ({
        id: `anomaly-${Date.now()}-${idx}`,
        x: a.location?.x || 50,
        y: a.location?.y || 50,
        width: a.location?.width || 10,
        height: a.location?.height || 10,
        confidence: a.confidence || 0.5,
        description: a.description || "Anomaly detected",
        severity: (a.severity || 'medium') as 'low' | 'medium' | 'high'
      }));
      
      return {
        id: `detection-${Date.now()}`,
        volumeId: '', // Will be set by caller
        sliceIndex,
        viewType,
        regions,
        confidence: data.overallConfidence || 0.5,
        timestamp: Date.now()
      };
    }
    
    // No anomalies detected
    return {
      id: `detection-${Date.now()}`,
      volumeId: '',
      sliceIndex,
      viewType,
      regions: [],
      confidence: 0.9,
      timestamp: Date.now()
    };
  } catch (error) {
    console.error("Anomaly Detection Error:", error);
    return {
      id: `detection-${Date.now()}`,
      volumeId: '',
      sliceIndex,
      viewType,
      regions: [],
      confidence: 0,
      timestamp: Date.now()
    };
  }
};

// Get color for anomaly severity
export const getAnomalyColor = (severity: 'low' | 'medium' | 'high'): string => {
  switch (severity) {
    case 'low':
      return '#fbbf24'; // Yellow
    case 'medium':
      return '#f97316'; // Orange
    case 'high':
      return '#ef4444'; // Red
    default:
      return '#fbbf24';
  }
};

// Batch anomaly detection for multiple slices
export const detectAnomaliesBatch = async (
  base64Images: string[],
  viewType: ViewType,
  onProgress?: (current: number, total: number) => void
): Promise<AnomalyDetection[]> => {
  const results: AnomalyDetection[] = [];
  
  for (let i = 0; i < base64Images.length; i++) {
    if (onProgress) {
      onProgress(i + 1, base64Images.length);
    }
    
    try {
      const detection = await detectAnomalies(base64Images[i], viewType, i);
      results.push(detection);
      
      // Small delay to avoid rate limiting
      if (i < base64Images.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (error) {
      console.error(`Error detecting anomalies in slice ${i}:`, error);
      results.push({
        id: `detection-${Date.now()}-${i}`,
        volumeId: '',
        sliceIndex: i,
        viewType,
        regions: [],
        confidence: 0,
        timestamp: Date.now()
      });
    }
  }
  
  return results;
};

