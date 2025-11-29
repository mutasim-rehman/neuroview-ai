import { GoogleGenAI } from "@google/genai";
import { EnhancedAnalysisResult, IdentifiedStructure, AutomatedMeasurement } from '../types';

const getClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found");
  }
  return new GoogleGenAI({ apiKey });
};

// Enhanced analysis with detailed prompts
export const analyzeMedicalSlice = async (
  base64Image: string,
  viewType?: string,
  sliceIndex?: number
): Promise<string> => {
  try {
    const ai = getClient();
    
    // Remove header from base64 string if present (data:image/jpeg;base64,...)
    const cleanBase64 = base64Image.split(',')[1] || base64Image;

    const viewInfo = viewType ? `View plane: ${viewType}. ` : '';
    const sliceInfo = sliceIndex !== undefined ? `Slice index: ${sliceIndex}. ` : '';

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
            text: `You are a medical imaging analysis assistant. Analyze this medical imaging slice in detail.

${viewInfo}${sliceInfo}

Provide a comprehensive analysis in the following format:

## Anatomical Identification
- **View Plane**: Identify if this is axial, sagittal, or coronal
- **Anatomical Region**: Specify the body region (e.g., brain, chest, abdomen)
- **Key Structures**: List 3-5 major anatomical structures visible

## Structural Analysis
- **Normal Anatomy**: Describe the normal anatomical structures visible
- **Spatial Relationships**: Note the relative positions of key structures
- **Tissue Characteristics**: Describe density/intensity patterns

## Quantitative Observations
- **Symmetry**: Assess left-right symmetry if applicable
- **Size/Proportions**: Note if structures appear normal in size
- **Intensity Patterns**: Describe any notable intensity variations

## Potential Findings
- **Anomalies**: Describe any visible anomalies, abnormalities, or areas of concern (with disclaimer: this is not a medical diagnosis)
- **Confidence Level**: Rate your confidence in observations (High/Medium/Low)
- **Recommendations**: Suggest what additional views or studies might be helpful

Format your response in clear Markdown with proper headings. Be professional, concise, and include appropriate medical disclaimers.`
          }
        ]
      }
    });

    return response.text || "No analysis generated.";
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    return "Failed to analyze the image. Please check your API key or try again.";
  }
};

// Enhanced analysis with structured results
export const analyzeMedicalSliceEnhanced = async (
  base64Image: string,
  viewType?: string,
  sliceIndex?: number
): Promise<EnhancedAnalysisResult> => {
  try {
    const ai = getClient();
    
    const cleanBase64 = base64Image.split(',')[1] || base64Image;
    const viewInfo = viewType ? `View plane: ${viewType}. ` : '';
    const sliceInfo = sliceIndex !== undefined ? `Slice index: ${sliceIndex}. ` : '';

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
            text: `Analyze this medical imaging slice and provide a structured JSON response along with a detailed text analysis.

${viewInfo}${sliceInfo}

Provide:
1. A detailed text analysis in Markdown format covering:
   - Anatomical identification (view plane, region, key structures)
   - Structural analysis (normal anatomy, spatial relationships)
   - Quantitative observations (symmetry, sizes, intensity patterns)
   - Potential findings (anomalies with confidence levels)
   - Recommendations

2. A JSON object with this structure:
{
  "structures": [
    {
      "name": "structure name",
      "confidence": 0.0-1.0,
      "location": {"x": 0, "y": 0, "z": 0},
      "description": "brief description"
    }
  ],
  "measurements": [
    {
      "type": "distance|area|volume|intensity",
      "value": 0,
      "unit": "mm|mm²|mm³|HU",
      "structure": "structure name",
      "location": {"x": 0, "y": 0, "z": 0}
    }
  ],
  "anomalyScore": 0.0-1.0,
  "confidence": 0.0-1.0,
  "recommendations": ["recommendation 1", "recommendation 2"]
}

Format your response as:
TEXT ANALYSIS:
[your markdown text here]

JSON DATA:
[your json object here]`
          }
        ]
      }
    });

    const text = response.text || "";
    
    // Parse structured data from response
    let structures: IdentifiedStructure[] = [];
    let measurements: AutomatedMeasurement[] = [];
    let anomalyScore: number | undefined;
    let confidence = 0.7;
    let recommendations: string[] = [];

    try {
      const jsonMatch = text.match(/JSON DATA:\s*(\{[\s\S]*\})/);
      if (jsonMatch) {
        const jsonData = JSON.parse(jsonMatch[1]);
        structures = jsonData.structures || [];
        measurements = (jsonData.measurements || []).map((m: any, idx: number) => ({
          ...m,
          id: `auto-measure-${Date.now()}-${idx}`
        }));
        anomalyScore = jsonData.anomalyScore;
        confidence = jsonData.confidence || 0.7;
        recommendations = jsonData.recommendations || [];
      }
    } catch (parseError) {
      console.warn("Failed to parse JSON from response", parseError);
    }

    const textAnalysis = text.split('JSON DATA:')[0].replace('TEXT ANALYSIS:', '').trim();

    return {
      text: textAnalysis || text,
      loading: false,
      structures,
      measurements,
      anomalyScore,
      confidence,
      recommendations
    };
  } catch (error) {
    console.error("Enhanced Analysis Error:", error);
    return {
      text: "Failed to analyze the image. Please check your API key or try again.",
      loading: false,
      error: error instanceof Error ? error.message : "Unknown error",
      structures: [],
      measurements: [],
      confidence: 0
    };
  }
};

// Batch analysis for multiple slices
export const analyzeBatchSlices = async (
  base64Images: string[],
  viewType?: string,
  onProgress?: (current: number, total: number) => void
): Promise<EnhancedAnalysisResult[]> => {
  const results: EnhancedAnalysisResult[] = [];
  
  for (let i = 0; i < base64Images.length; i++) {
    if (onProgress) {
      onProgress(i + 1, base64Images.length);
    }
    
    try {
      const result = await analyzeMedicalSliceEnhanced(
        base64Images[i],
        viewType,
        i
      );
      results.push(result);
      
      // Small delay to avoid rate limiting
      if (i < base64Images.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (error) {
      console.error(`Error analyzing slice ${i}:`, error);
      results.push({
        text: `Failed to analyze slice ${i + 1}`,
        loading: false,
        error: error instanceof Error ? error.message : "Unknown error",
        structures: [],
        measurements: [],
        confidence: 0
      });
    }
  }
  
  return results;
};

// AI-powered window/level suggestions
export const suggestWindowLevel = async (
  base64Image: string,
  currentWindow: number,
  currentLevel: number
): Promise<{ window: number; level: number; reason: string }> => {
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
            text: `Analyze this medical image and suggest optimal window/level settings.

Current settings: Window=${currentWindow}, Level=${currentLevel}

Provide a JSON response with:
{
  "window": suggested window value,
  "level": suggested level value,
  "reason": "brief explanation of why these settings are optimal for this image type"
}

Consider:
- The tissue types visible
- The contrast needed for diagnostic quality
- Standard window/level presets for this anatomy
- Optimal visualization of key structures`
          }
        ]
      }
    });

    const text = response.text || "";
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    
    if (jsonMatch) {
      const data = JSON.parse(jsonMatch[0]);
      return {
        window: data.window || currentWindow,
        level: data.level || currentLevel,
        reason: data.reason || "AI-suggested optimization"
      };
    }
    
    return {
      window: currentWindow,
      level: currentLevel,
      reason: "Unable to generate suggestion"
    };
  } catch (error) {
    console.error("Window/Level Suggestion Error:", error);
    return {
      window: currentWindow,
      level: currentLevel,
      reason: "Error generating suggestion"
    };
  }
};

