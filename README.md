# NeuroView AI

<div align="center">

**Advanced Medical Imaging Visualization Platform with AI-Powered Analysis**

[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19.2-blue)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-0.181-green)](https://threejs.org/)
[![Vite](https://img.shields.io/badge/Vite-6.2-purple)](https://vitejs.dev/)

</div>

## Overview

NeuroView AI is a sophisticated web-based medical imaging platform designed for visualizing, analyzing, and interpreting NIfTI (Neuroimaging Informatics Technology Initiative) files. Built with modern web technologies, it provides advanced 3D volume rendering, multi-volume overlay capabilities, and AI-powered analysis using Google's Gemini Vision API.

### Key Features

#### 🎨 **Advanced Visualization**
- **Multi-Volume Overlay**: Load and display multiple NIfTI volumes simultaneously with various blending modes (Blend, Difference, Multiply, Additive, Maximum, Minimum)
- **3D Volume Rendering**: High-quality ray-marched volume rendering with multiple styles:
  - **MIP (Maximum Intensity Projection)**: X-ray-like visualization
  - **ISO Surface**: Surface rendering with threshold-based extraction
  - **Volumetric**: Full volume cloud rendering with transparency
- **Multi-Planar Views**: Simultaneous display of Axial, Sagittal, and Coronal slices in quad-view mode
- **Time-Series Support**: Full 4D volume support with playback controls, timeline scrubbing, and animation
- **Advanced Shaders**: 
  - Phong lighting model with configurable light sources
  - Ambient occlusion for enhanced depth perception
  - Adaptive ray marching for optimal performance
  - Render quality presets (Fast/Medium/High/Ultra)

#### 🧠 **AI-Powered Analysis**
- **Enhanced Medical Analysis**: Detailed anatomical structure identification using Gemini Vision API
- **Automated Segmentation**: AI-powered region-of-interest (ROI) detection and mask generation
- **Anomaly Detection**: Intelligent flagging of potential abnormalities with confidence scoring
- **Quantitative Measurements**: Automated distance, area, volume, and intensity measurements
- **Batch Analysis**: Process multiple slices simultaneously with progress tracking
- **AI-Suggested Settings**: Automatic window/level optimization recommendations

#### 📊 **Analysis Tools**
- **Histogram Visualization**: Real-time histogram display with window/level controls
- **Window/Level Presets**: Pre-configured settings for Brain, Bone, Lung, Soft Tissue, and Abdomen
- **Measurement Tools**: Ruler tool for distance measurements with real-world units (mm)
- **Annotation System**: Mark and label regions of interest
- **ROI Statistics**: Real-time voxel intensity and density classification on hover

#### 🎛️ **Volume Management**
- **Multi-Volume Support**: Load, manage, and overlay multiple volumes
- **Per-Volume Controls**: Individual opacity, visibility, color map, and window/level settings
- **Volume Metadata**: Custom naming, color coding, and organization
- **Time-Series Playback**: Frame-by-frame navigation with variable speed control

#### ⚡ **Performance Optimizations**
- **Adaptive Rendering**: Dynamic quality adjustment based on performance
- **Efficient Memory Management**: Optimized texture handling and data structures
- **Progressive Loading**: Smooth loading experience for large files

## Installation

### Prerequisites

- **Node.js** 18+ and npm
- **Google Gemini API Key** (for AI features)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd neuroview-ai
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   
   Create a `.env.local` file in the root directory:
   ```env
   API_KEY=your_gemini_api_key_here
   ```
   
   > **Note**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:5173` (or the port shown in the terminal)

## Usage Guide

### Loading Volumes

1. **Single Volume**: Click the file upload area and select a `.nii` or `.nii.gz` file
2. **Multiple Volumes**: Load additional volumes to enable overlay mode
3. **4D Volumes**: Time-series volumes are automatically detected and playback controls appear

### Viewing Modes

- **2D Slice**: Single slice view with orientation selector
- **Quad View**: Simultaneous display of Axial, Sagittal, Coronal, and 3D volume
- **3D Volume**: Full-screen volume rendering

### Tools

- **Pointer**: Default tool for navigation and ROI inspection
- **Ruler**: Click and drag to measure distances (displayed in mm)
- **Marker**: Click to add text annotations

### Volume Controls

Access the volume management panel (right sidebar) to:
- Toggle volume visibility
- Adjust opacity (0-100%)
- Change overlay blending mode
- Set individual window/level values
- Remove volumes

### AI Analysis

1. **Slice Analysis**: Click the brain icon in any slice viewer to analyze the current slice
2. **Batch Analysis**: Use the enhanced analysis features for comprehensive evaluation
3. **Segmentation**: AI-powered structure identification and ROI detection
4. **Anomaly Detection**: Automatic flagging of potential findings

### Window/Level Adjustment

- Use the **Histogram Panel** to:
  - View intensity distribution
  - Apply presets (Brain, Bone, Lung, etc.)
  - Manually adjust window and level
  - See real-time updates

### 3D Rendering Controls

- **Render Style**: Switch between MIP, ISO Surface, and Volumetric
- **Color Maps**: Choose from Grayscale, Hot Iron, Cool Blue, Rainbow, Anatomy, Density Heatmap
- **Tissue Presets**: Quick access to Skin, Bone, Soft Tissue, and Vessel visualization
- **Lightsaber Cut**: Interactive plane cutting for internal structure exploration
- **Render Quality**: Adjust quality vs. performance (Fast/Medium/High/Ultra)

### Time-Series Playback

For 4D volumes:
- Use play/pause button to start/stop animation
- Drag timeline slider to navigate frames
- Adjust playback speed (0.5x to 5x)
- Enable/disable looping

## Project Structure

```
neuroview-ai/
├── components/
│   ├── Viewer.tsx              # 2D slice viewer with multi-volume overlay
│   ├── VolumeViewer.tsx         # 3D volume renderer with advanced shaders
│   ├── FileUpload.tsx           # File upload component
│   └── HistogramPanel.tsx      # Histogram and window/level controls
├── services/
│   ├── geminiService.ts         # Enhanced AI analysis service
│   ├── segmentationService.ts   # AI-powered segmentation
│   └── anomalyDetectionService.ts # Anomaly detection service
├── utils/
│   └── niftiLoader.ts           # NIfTI file parser with 4D support
├── types.ts                     # TypeScript type definitions
├── App.tsx                      # Main application component
└── package.json                 # Dependencies and scripts
```

## Architecture

### Core Technologies

- **React 19.2**: Modern UI framework with hooks
- **TypeScript 5.8**: Type-safe development
- **Three.js 0.181**: 3D graphics and WebGL rendering
- **Vite 6.2**: Fast build tool and dev server
- **Google Gemini API**: AI-powered medical image analysis

### Key Components

#### Viewer Component
- Handles 2D slice rendering with multi-volume overlay
- Implements various blending algorithms
- Supports measurement and annotation tools
- Real-time ROI statistics extraction

#### VolumeViewer Component
- Advanced ray-marching shader implementation
- Phong lighting with ambient occlusion
- Adaptive step sizing for performance
- Multiple rendering styles and color maps

#### Services Layer
- **geminiService**: Enhanced analysis with structured output
- **segmentationService**: ROI detection and mask generation
- **anomalyDetectionService**: Automated anomaly flagging

## API Reference

### Gemini Service

```typescript
// Enhanced analysis with structured results
analyzeMedicalSliceEnhanced(
  base64Image: string,
  viewType?: string,
  sliceIndex?: number
): Promise<EnhancedAnalysisResult>

// Batch analysis
analyzeBatchSlices(
  base64Images: string[],
  viewType?: string,
  onProgress?: (current: number, total: number) => void
): Promise<EnhancedAnalysisResult[]>

// Window/level suggestions
suggestWindowLevel(
  base64Image: string,
  currentWindow: number,
  currentLevel: number
): Promise<{ window: number; level: number; reason: string }>
```

### Segmentation Service

```typescript
// Segment structures
segmentStructures(
  base64Image: string,
  viewType: string,
  sliceIndex: number
): Promise<SegmentationRegion[]>

// Create segmentation mask
createSegmentationMask(
  volume: VolumeData,
  regions: SegmentationRegion[],
  name: string
): SegmentationMask
```

### Anomaly Detection Service

```typescript
// Detect anomalies
detectAnomalies(
  base64Image: string,
  viewType: ViewType,
  sliceIndex: number
): Promise<AnomalyDetection>

// Batch detection
detectAnomaliesBatch(
  base64Images: string[],
  viewType: ViewType,
  onProgress?: (current: number, total: number) => void
): Promise<AnomalyDetection[]>
```

## Development

### Available Scripts

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory, ready for deployment.

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

**Note**: WebGL 2.0 support is required for 3D volume rendering.

## Limitations & Disclaimers

⚠️ **Medical Disclaimer**: This software is for visualization and educational purposes only. It is NOT a medical device and should NOT be used for diagnostic purposes. All AI-generated analyses are suggestions and must be reviewed by qualified medical professionals.

### Current Limitations

- Maximum volume size depends on available GPU memory
- Very large 4D volumes may experience performance issues
- AI analysis requires internet connection and API key
- Segmentation accuracy depends on image quality and AI model capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution

- Additional file format support (DICOM, Analyze, etc.)
- More advanced segmentation algorithms
- Export functionality (screenshots, measurements, reports)
- Keyboard shortcuts
- Plugin system for custom analysis tools

## License

[Specify your license here]

## Acknowledgments

- **NIfTI Format**: Neuroimaging Informatics Technology Initiative
- **Three.js**: 3D graphics library
- **Google Gemini**: AI analysis capabilities
- **nifti-reader-js**: NIfTI file parsing library

## Support

For issues, questions, or feature requests, please open an issue on the repository.

---

<div align="center">

**Built with ❤️ for the medical imaging community**

</div>
