# Brain Region Segmentation Implementation Guide

This guide details how to implement three advanced segmentation methods: **Non-linear Registration**, **Deep Learning**, and **Multi-Atlas Fusion**.

---

## 1. Non-Linear Registration (Deformable Registration)

### Overview
Warp a standard brain atlas to match individual brain shape using dense deformation fields. This handles shape variations between brains.

### How It Works

#### Step 1: Choose Registration Tool
**Option A: ANTs (Advanced Normalization Tools)** - Most accurate, industry standard
**Option B: FSL FNIRT** - Good alternative, easier setup
**Option C: Elastix** - Open-source, flexible

#### Step 2: Implementation Flow

```
User's Brain (NIfTI) 
    ↓
[Preprocessing: Skull stripping, intensity normalization]
    ↓
[Affine Registration: Initial alignment]
    ↓
[Non-linear Registration: Dense deformation field]
    ↓
[Apply Deformation to Atlas Labels]
    ↓
[Resample to User's Space]
    ↓
Segmented Brain Regions (Label Map)
```

#### Step 3: Backend API Endpoint

**File: `ai-training/api_server.py`**

```python
import subprocess
import tempfile
import os
import nibabel as nib
import numpy as np
from flask import request, jsonify, send_file

@app.route('/segment/nonlinear', methods=['POST'])
def segment_nonlinear():
    """
    Segment brain using non-linear registration.
    
    Request:
    - file: NIfTI file (multipart/form-data)
    - atlas: Atlas name ('AAL', 'HarvardOxford', 'DesikanKilliany')
    - method: 'ANTs', 'FSL', or 'Elastix'
    
    Response:
    - segmentation: Label map NIfTI file
    - regions: List of region IDs and names
    - processing_time: Seconds
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        atlas_name = request.form.get('atlas', 'HarvardOxford')
        method = request.form.get('method', 'ANTs')
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_input:
            file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Load atlas (pre-downloaded)
        atlas_path = f'atlases/{atlas_name}_cortical.nii.gz'
        atlas_labels_path = f'atlases/{atlas_name}_labels.json'
        
        if not os.path.exists(atlas_path):
            return jsonify({'error': f'Atlas {atlas_name} not found'}), 404
        
        # Run registration
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_output:
            output_path = tmp_output.name
        
        if method == 'ANTs':
            result = run_ants_registration(input_path, atlas_path, output_path)
        elif method == 'FSL':
            result = run_fsl_registration(input_path, atlas_path, output_path)
        else:
            result = run_elastix_registration(input_path, atlas_path, output_path)
        
        # Load segmentation result
        seg_img = nib.load(output_path)
        seg_data = seg_img.get_fdata()
        
        # Extract unique regions
        unique_labels = np.unique(seg_data[seg_data > 0])
        regions = []
        with open(atlas_labels_path, 'r') as f:
            label_map = json.load(f)
            for label_id in unique_labels:
                regions.append({
                    'id': int(label_id),
                    'name': label_map.get(str(int(label_id)), f'Region_{int(label_id)}')
                })
        
        # Return segmentation file
        return send_file(
            output_path,
            mimetype='application/gzip',
            as_attachment=True,
            download_name='segmentation.nii.gz'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_ants_registration(moving_path, fixed_path, output_path):
    """Run ANTs registration pipeline."""
    import time
    start_time = time.time()
    
    # ANTs commands
    # Step 1: Affine registration
    affine_matrix = f'{output_path}_affine.mat'
    subprocess.run([
        'antsRegistration',
        '--dimensionality', '3',
        '--float', '0',
        '--output', f'{output_path}_',
        '--interpolation', 'Linear',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--use-histogram-matching', '0',
        '--initial-moving-transform', f'[{fixed_path},{moving_path},1]',
        '--transform', 'Affine[0.1]',
        '--metric', 'MI[{fixed_path},{moving_path},1,32,Regular,0.25]',
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox'
    ], check=True)
    
    # Step 2: Non-linear SyN registration
    subprocess.run([
        'antsRegistration',
        '--dimensionality', '3',
        '--float', '0',
        '--output', f'{output_path}_syn_',
        '--interpolation', 'Linear',
        '--winsorize-image-intensities', '[0.005,0.995]',
        '--use-histogram-matching', '0',
        '--initial-moving-transform', f'{output_path}_0GenericAffine.mat',
        '--transform', 'SyN[0.1,3,0]',
        '--metric', 'CC[{fixed_path},{moving_path},1,4]',
        '--convergence', '[100x50x30x20,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox'
    ], check=True)
    
    # Step 3: Apply transformation to atlas labels
    atlas_labels_path = fixed_path.replace('.nii.gz', '_labels.nii.gz')
    subprocess.run([
        'antsApplyTransforms',
        '--dimensionality', '3',
        '--input', atlas_labels_path,
        '--reference-image', moving_path,
        '--output', output_path,
        '--interpolation', 'NearestNeighbor',  # Preserve label values
        '--transform', f'{output_path}_syn_1Warp.nii.gz',
        '--transform', f'{output_path}_syn_0GenericAffine.mat'
    ], check=True)
    
    processing_time = time.time() - start_time
    return {'success': True, 'processing_time': processing_time}
```

### Requirements

#### Software Dependencies
```bash
# Install ANTs (Linux/Mac)
wget https://github.com/ANTsX/ANTs/releases/download/v2.5.0/ants-Linux-centos7_x86_64-v2.5.0-c726b52.tar.gz
tar -xzf ants-*.tar.gz
export ANTSPATH=/path/to/ants/bin
export PATH=$ANTSPATH:$PATH

# Or use Docker
docker pull antsx/ants

# Python dependencies
pip install nibabel numpy scipy
```

#### Atlas Files Required
```
atlases/
├── HarvardOxford_cortical.nii.gz      # Template brain
├── HarvardOxford_labels.json          # Label ID → Name mapping
├── AAL_cortical.nii.gz
├── AAL_labels.json
└── DesikanKilliany_cortical.nii.gz
└── DesikanKilliany_labels.json
```

**Download Atlases:**
- Harvard-Oxford: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
- AAL: https://www.gin.cnrs.fr/en/tools/aal/
- Desikan-Killiany: https://surfer.nmr.mgh.harvard.edu/

#### Infrastructure
- **CPU**: Multi-core recommended (4+ cores)
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB+ for atlases
- **Processing Time**: 5-30 minutes per brain (depends on size)

#### API Integration

**Frontend Service: `services/segmentationService.ts`**

```typescript
export interface NonLinearSegmentationResult {
  segmentationUrl: string; // URL to download segmentation file
  regions: Array<{
    id: number;
    name: string;
    voxelCount?: number;
  }>;
  processingTime: number;
  method: 'ANTs' | 'FSL' | 'Elastix';
}

export const segmentWithNonLinear = async (
  file: File,
  atlas: 'AAL' | 'HarvardOxford' | 'DesikanKilliany' = 'HarvardOxford',
  method: 'ANTs' | 'FSL' | 'Elastix' = 'ANTs'
): Promise<NonLinearSegmentationResult> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('atlas', atlas);
  formData.append('method', method);

  const response = await fetch(`${API_BASE_URL}/segment/nonlinear`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Segmentation failed');
  }

  // Download segmentation file
  const blob = await response.blob();
  const segmentationUrl = URL.createObjectURL(blob);

  // Parse metadata from headers or separate endpoint
  const metadata = JSON.parse(response.headers.get('X-Segmentation-Metadata') || '{}');

  return {
    segmentationUrl,
    regions: metadata.regions || [],
    processingTime: metadata.processing_time || 0,
    method,
  };
};
```

---

## 2. Deep Learning Segmentation

### Overview
Train or use a pre-trained neural network to automatically segment brain regions. Fast inference (~seconds), high accuracy.

### How It Works

#### Step 1: Choose Model Architecture

**Option A: nnU-Net** (Recommended - State-of-the-art)
- Self-configuring, works out-of-the-box
- Best accuracy for medical imaging
- Pre-trained models available

**Option B: FastSurfer**
- Specifically for brain segmentation
- Fast inference (~1 minute)
- FreeSurfer-compatible output

**Option C: Custom U-Net**
- Train your own model
- Full control over architecture
- Requires labeled dataset

#### Step 2: Implementation Flow

```
User's Brain (NIfTI)
    ↓
[Preprocessing: Normalization, resampling to model input size]
    ↓
[Model Inference: Forward pass through neural network]
    ↓
[Post-processing: Label smoothing, connected components]
    ↓
Segmented Brain Regions (Label Map)
```

#### Step 3: Backend Implementation

**File: `ai-training/segmentation_model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from typing import Tuple, Optional

class BrainSegmentationUNet(nn.Module):
    """
    3D U-Net for brain region segmentation.
    Outputs: 50+ brain regions (cortical + subcortical)
    """
    def __init__(self, in_channels=1, num_classes=50, base_channels=32):
        super().__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self._make_conv_block(in_channels, base_channels)
        self.enc2 = self._make_conv_block(base_channels, base_channels * 2)
        self.enc3 = self._make_conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder (Expanding Path)
        self.dec4 = self._make_conv_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.dec3 = self._make_conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._make_conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._make_conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Output
        self.final = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        
        # Pooling/Upsampling
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        out = self.final(d1)
        return out

def preprocess_volume(volume_path: str, target_shape: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Preprocess NIfTI volume for model input."""
    img = nib.load(volume_path)
    data = img.get_fdata().astype(np.float32)
    
    # Normalize intensity (0-1)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # Resample to target shape (simple linear interpolation)
    from scipy.ndimage import zoom
    zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)]
    data = zoom(data, zoom_factors, order=1)
    
    # Add channel dimension
    data = data[np.newaxis, ...]  # (1, D, H, W)
    
    return data

def postprocess_segmentation(prediction: np.ndarray, original_shape: Tuple[int, int, int]) -> np.ndarray:
    """Post-process model output."""
    # Get class predictions (argmax)
    seg = np.argmax(prediction, axis=0)  # (D, H, W)
    
    # Resample back to original shape
    from scipy.ndimage import zoom
    zoom_factors = [original_shape[i] / seg.shape[i] for i in range(3)]
    seg = zoom(seg, zoom_factors, order=0)  # Nearest neighbor for labels
    
    # Smooth labels (optional)
    from scipy.ndimage import median_filter
    seg = median_filter(seg, size=3)
    
    return seg.astype(np.uint8)
```

**File: `ai-training/api_server.py` (Add endpoint)**

```python
import torch
from segmentation_model import BrainSegmentationUNet, preprocess_volume, postprocess_segmentation

# Global model instance
segmentation_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_segmentation_model():
    """Load pre-trained segmentation model."""
    global segmentation_model
    
    model = BrainSegmentationUNet(in_channels=1, num_classes=50, base_channels=32)
    checkpoint_path = 'checkpoints/brain_segmentation_model.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        segmentation_model = model
        print(f"Segmentation model loaded on {device}")
    else:
        print("Warning: Segmentation model not found. Using pre-trained weights or download model.")
        # Option: Download from model zoo
        # download_pretrained_model(checkpoint_path)

@app.route('/segment/deeplearning', methods=['POST'])
def segment_deeplearning():
    """
    Segment brain using deep learning model.
    
    Request:
    - file: NIfTI file
    
    Response:
    - segmentation: Label map NIfTI file
    - regions: List of detected regions
    - confidence: Average confidence score
    """
    try:
        if segmentation_model is None:
            load_segmentation_model()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_input:
            file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Load original volume for shape
        img = nib.load(input_path)
        original_shape = img.shape[:3]
        
        # Preprocess
        preprocessed = preprocess_volume(input_path, target_shape=(128, 128, 128))
        input_tensor = torch.from_numpy(preprocessed).unsqueeze(0).to(device)  # (1, 1, D, H, W)
        
        # Inference
        with torch.no_grad():
            output = segmentation_model(input_tensor)  # (1, num_classes, D, H, W)
            probabilities = F.softmax(output, dim=1)
            prediction = output.cpu().numpy()[0]  # (num_classes, D, H, W)
        
        # Post-process
        segmentation = postprocess_segmentation(prediction, original_shape)
        
        # Calculate confidence
        prob_array = probabilities.cpu().numpy()[0]
        max_probs = np.max(prob_array, axis=0)
        avg_confidence = float(np.mean(max_probs))
        
        # Extract regions
        unique_labels = np.unique(segmentation[segmentation > 0])
        regions = []
        with open('atlases/brain_regions.json', 'r') as f:
            label_map = json.load(f)
            for label_id in unique_labels:
                regions.append({
                    'id': int(label_id),
                    'name': label_map.get(str(int(label_id)), f'Region_{int(label_id)}'),
                    'voxelCount': int(np.sum(segmentation == label_id))
                })
        
        # Save segmentation
        seg_img = nib.Nifti1Image(segmentation, img.affine, img.header)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz', delete=False) as tmp_output:
            output_path = tmp_output.name
            nib.save(seg_img, output_path)
        
        return send_file(
            output_path,
            mimetype='application/gzip',
            as_attachment=True,
            download_name='segmentation_dl.nii.gz'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Requirements

#### Pre-trained Models

**Option 1: Use nnU-Net Pre-trained Models**
```bash
# Install nnU-Net
pip install nnunet

# Download pre-trained brain segmentation model
nnUNet_download_preprocessed_data -d 001 -p nnUNet_preprocessed
```

**Option 2: Use FastSurfer**
```bash
# Install FastSurfer
pip install fastsurfer

# Download model
fastsurfer --download_model
```

**Option 3: Train Your Own Model**
- Requires labeled dataset (e.g., OASIS, ADNI)
- Training time: Days to weeks on GPU
- See `ai-training/main_train_segmentation.py` (create this)

#### Infrastructure
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: Can run on CPU but 10-100x slower
- **RAM**: 16GB+ recommended
- **Storage**: 2-5GB for model files
- **Processing Time**: 
  - GPU: 5-30 seconds
  - CPU: 2-10 minutes

#### Dataset Requirements (if training)
- **Labeled scans**: 100+ brain scans with region annotations
- **Format**: NIfTI files + label maps
- **Sources**: OASIS, ADNI, IXI, FreeSurfer datasets

#### API Integration

**Frontend Service: `services/segmentationService.ts`**

```typescript
export interface DeepLearningSegmentationResult {
  segmentationUrl: string;
  regions: Array<{
    id: number;
    name: string;
    voxelCount: number;
  }>;
  confidence: number;
  processingTime: number;
}

export const segmentWithDeepLearning = async (
  file: File
): Promise<DeepLearningSegmentationResult> => {
  const formData = new FormData();
  formData.append('file', file);

  const startTime = Date.now();
  
  const response = await fetch(`${API_BASE_URL}/segment/deeplearning`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Deep learning segmentation failed');
  }

  const blob = await response.blob();
  const segmentationUrl = URL.createObjectURL(blob);
  
  const metadata = JSON.parse(response.headers.get('X-Segmentation-Metadata') || '{}');

  return {
    segmentationUrl,
    regions: metadata.regions || [],
    confidence: metadata.confidence || 0,
    processingTime: (Date.now() - startTime) / 1000,
  };
};
```

---

## 3. Multi-Atlas Fusion

### Overview
Use multiple atlases, register each to the user's brain, then combine results using voting or weighted averaging.

### How It Works

#### Step 1: Atlas Selection
Choose 3-5 diverse atlases:
- **AAL (Automated Anatomical Labeling)**: 116 regions
- **Harvard-Oxford**: Cortical + subcortical
- **Desikan-Killiany**: FreeSurfer parcellation
- **JHU White Matter**: White matter tracts
- **MNI152**: Standard space template

#### Step 2: Implementation Flow

```
User's Brain (NIfTI)
    ↓
[For each atlas:]
    ├─→ [Register to user's brain]
    ├─→ [Apply transformation to labels]
    └─→ [Get segmentation]
    ↓
[Combine segmentations:]
    ├─→ Majority Voting (most common label wins)
    ├─→ Weighted Average (by registration confidence)
    └─→ Probabilistic Fusion (probability maps)
    ↓
Final Segmentation (Consensus)
```

#### Step 3: Backend Implementation

**File: `ai-training/multi_atlas_fusion.py`**

```python
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess

class MultiAtlasFusion:
    def __init__(self, atlas_configs: List[Dict]):
        """
        Initialize with multiple atlas configurations.
        
        Args:
            atlas_configs: List of dicts with keys:
                - name: Atlas name
                - template_path: Path to template brain
                - labels_path: Path to label map
                - labels_json: Path to label ID → name mapping
                - weight: Confidence weight (0-1)
        """
        self.atlas_configs = atlas_configs
    
    def fuse_segmentations(
        self,
        user_brain_path: str,
        method: str = 'majority_vote'
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fuse multiple atlas segmentations.
        
        Args:
            user_brain_path: Path to user's brain NIfTI
            method: 'majority_vote', 'weighted_average', or 'probabilistic'
        
        Returns:
            Tuple of (segmentation array, metadata)
        """
        # Register each atlas in parallel
        segmentations = []
        confidences = []
        
        with ThreadPoolExecutor(max_workers=len(self.atlas_configs)) as executor:
            futures = []
            for atlas_config in self.atlas_configs:
                future = executor.submit(
                    self._register_and_segment,
                    user_brain_path,
                    atlas_config
                )
                futures.append((future, atlas_config))
            
            for future, atlas_config in futures:
                seg, conf = future.result()
                segmentations.append(seg)
                confidences.append(conf * atlas_config.get('weight', 1.0))
        
        # Load reference image for shape/affine
        ref_img = nib.load(user_brain_path)
        shape = ref_img.shape[:3]
        
        # Fuse segmentations
        if method == 'majority_vote':
            fused = self._majority_vote(segmentations, confidences, shape)
        elif method == 'weighted_average':
            fused = self._weighted_average(segmentations, confidences, shape)
        else:
            fused = self._probabilistic_fusion(segmentations, confidences, shape)
        
        # Extract metadata
        unique_labels = np.unique(fused[fused > 0])
        metadata = {
            'num_atlases': len(self.atlas_configs),
            'method': method,
            'regions': self._extract_regions(fused, unique_labels),
            'agreement': self._calculate_agreement(segmentations, fused)
        }
        
        return fused, metadata
    
    def _register_and_segment(
        self,
        user_brain_path: str,
        atlas_config: Dict
    ) -> Tuple[np.ndarray, float]:
        """Register one atlas and return segmentation."""
        # Run registration (simplified - use ANTs or FSL)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp:
            output_path = tmp.name
        
        # Run ANTs registration
        result = run_ants_registration(
            user_brain_path,
            atlas_config['template_path'],
            output_path
        )
        
        # Load warped labels
        labels_img = nib.load(atlas_config['labels_path'])
        warped_labels = apply_transformation(
            labels_img,
            result['warp_field'],
            result['affine_matrix'],
            reference=user_brain_path
        )
        
        seg_data = warped_labels.get_fdata().astype(np.uint16)
        confidence = result.get('confidence', 0.8)  # Registration quality metric
        
        return seg_data, confidence
    
    def _majority_vote(
        self,
        segmentations: List[np.ndarray],
        confidences: List[float],
        shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Majority voting: most common label wins."""
        # Stack all segmentations
        stacked = np.stack(segmentations, axis=0)  # (N_atlases, D, H, W)
        
        # For each voxel, find most common label
        fused = np.zeros(shape, dtype=np.uint16)
        
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    voxel_labels = stacked[:, z, y, x]
                    voxel_labels = voxel_labels[voxel_labels > 0]  # Ignore background
                    
                    if len(voxel_labels) > 0:
                        # Weighted voting
                        unique_labels, counts = np.unique(voxel_labels, return_counts=True)
                        # Apply confidence weights
                        weighted_counts = []
                        for label in unique_labels:
                            weight_sum = sum(
                                confidences[i] 
                                for i, lab in enumerate(voxel_labels) 
                                if lab == label
                            )
                            weighted_counts.append(weight_sum)
                        
                        # Most common label
                        best_idx = np.argmax(weighted_counts)
                        fused[z, y, x] = unique_labels[best_idx]
        
        return fused
    
    def _weighted_average(
        self,
        segmentations: List[np.ndarray],
        confidences: List[float],
        shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Weighted average: blend labels by confidence."""
        # Normalize confidences
        total_conf = sum(confidences)
        weights = [c / total_conf for c in confidences]
        
        # Create probability maps for each label
        all_labels = set()
        for seg in segmentations:
            all_labels.update(np.unique(seg[seg > 0]))
        
        # For each label, compute weighted probability
        label_probs = {}
        for label_id in all_labels:
            prob_map = np.zeros(shape, dtype=np.float32)
            for seg, weight in zip(segmentations, weights):
                prob_map += (seg == label_id).astype(np.float32) * weight
            label_probs[label_id] = prob_map
        
        # Assign label with highest probability
        fused = np.zeros(shape, dtype=np.uint16)
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    best_label = 0
                    best_prob = 0
                    for label_id, prob_map in label_probs.items():
                        if prob_map[z, y, x] > best_prob:
                            best_prob = prob_map[z, y, x]
                            best_label = label_id
                    fused[z, y, x] = best_label
        
        return fused
    
    def _probabilistic_fusion(
        self,
        segmentations: List[np.ndarray],
        confidences: List[float],
        shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Probabilistic fusion: output probability maps."""
        # Similar to weighted_average but keep probabilities
        # For now, return hard segmentation (can extend to return prob maps)
        return self._weighted_average(segmentations, confidences, shape)
    
    def _extract_regions(
        self,
        segmentation: np.ndarray,
        unique_labels: np.ndarray
    ) -> List[Dict]:
        """Extract region information."""
        regions = []
        with open('atlases/combined_labels.json', 'r') as f:
            label_map = json.load(f)
            for label_id in unique_labels:
                regions.append({
                    'id': int(label_id),
                    'name': label_map.get(str(int(label_id)), f'Region_{int(label_id)}'),
                    'voxelCount': int(np.sum(segmentation == label_id))
                })
        return regions
    
    def _calculate_agreement(
        self,
        segmentations: List[np.ndarray],
        fused: np.ndarray
    ) -> float:
        """Calculate inter-atlas agreement."""
        agreements = []
        for seg in segmentations:
            # Percentage of voxels that match fused segmentation
            match = np.sum((seg > 0) & (seg == fused)) / np.sum(fused > 0)
            agreements.append(match)
        return float(np.mean(agreements))
```

**File: `ai-training/api_server.py` (Add endpoint)**

```python
from multi_atlas_fusion import MultiAtlasFusion

# Initialize multi-atlas fusion
atlas_configs = [
    {
        'name': 'HarvardOxford',
        'template_path': 'atlases/HarvardOxford_cortical.nii.gz',
        'labels_path': 'atlases/HarvardOxford_labels.nii.gz',
        'labels_json': 'atlases/HarvardOxford_labels.json',
        'weight': 1.0
    },
    {
        'name': 'AAL',
        'template_path': 'atlases/AAL_cortical.nii.gz',
        'labels_path': 'atlases/AAL_labels.nii.gz',
        'labels_json': 'atlases/AAL_labels.json',
        'weight': 0.9
    },
    {
        'name': 'DesikanKilliany',
        'template_path': 'atlases/DesikanKilliany_cortical.nii.gz',
        'labels_path': 'atlases/DesikanKilliany_labels.nii.gz',
        'labels_json': 'atlases/DesikanKilliany_labels.json',
        'weight': 0.85
    }
]

fusion_engine = MultiAtlasFusion(atlas_configs)

@app.route('/segment/multi-atlas', methods=['POST'])
def segment_multi_atlas():
    """
    Segment brain using multi-atlas fusion.
    
    Request:
    - file: NIfTI file
    - method: 'majority_vote', 'weighted_average', or 'probabilistic'
    - atlases: Comma-separated list of atlas names (optional)
    
    Response:
    - segmentation: Label map NIfTI file
    - regions: List of regions
    - agreement: Inter-atlas agreement score
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        method = request.form.get('method', 'majority_vote')
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_input:
            file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Run fusion
        segmentation, metadata = fusion_engine.fuse_segmentations(input_path, method)
        
        # Save segmentation
        ref_img = nib.load(input_path)
        seg_img = nib.Nifti1Image(segmentation, ref_img.affine, ref_img.header)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_output:
            output_path = tmp_output.name
            nib.save(seg_img, output_path)
        
        return send_file(
            output_path,
            mimetype='application/gzip',
            as_attachment=True,
            download_name='segmentation_multi_atlas.nii.gz'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Requirements

#### Multiple Atlases
- **3-5 different atlases** (see list above)
- Each atlas needs:
  - Template brain (NIfTI)
  - Label map (NIfTI with region IDs)
  - Label ID → Name JSON mapping
- **Total storage**: 10-20GB for all atlases

#### Infrastructure
- **CPU**: Multi-core essential (8+ cores recommended)
- **RAM**: 32GB+ (runs multiple registrations in parallel)
- **Storage**: 20GB+ for atlases
- **Processing Time**: 
  - Sequential: 30-120 minutes (3-5 × single registration time)
  - Parallel: 10-30 minutes (limited by slowest registration)

#### API Integration

**Frontend Service: `services/segmentationService.ts`**

```typescript
export interface MultiAtlasSegmentationResult {
  segmentationUrl: string;
  regions: Array<{
    id: number;
    name: string;
    voxelCount: number;
  }>;
  agreement: number; // 0-1, inter-atlas agreement score
  numAtlases: number;
  method: 'majority_vote' | 'weighted_average' | 'probabilistic';
  processingTime: number;
}

export const segmentWithMultiAtlas = async (
  file: File,
  method: 'majority_vote' | 'weighted_average' | 'probabilistic' = 'majority_vote'
): Promise<MultiAtlasSegmentationResult> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('method', method);

  const startTime = Date.now();
  
  const response = await fetch(`${API_BASE_URL}/segment/multi-atlas`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Multi-atlas segmentation failed');
  }

  const blob = await response.blob();
  const segmentationUrl = URL.createObjectURL(blob);
  
  const metadata = JSON.parse(response.headers.get('X-Segmentation-Metadata') || '{}');

  return {
    segmentationUrl,
    regions: metadata.regions || [],
    agreement: metadata.agreement || 0,
    numAtlases: metadata.num_atlases || 0,
    method,
    processingTime: (Date.now() - startTime) / 1000,
  };
};
```

---

## Integration with Frontend

### Update `App.tsx` to Use Segmentation

```typescript
import { 
  segmentWithNonLinear, 
  segmentWithDeepLearning, 
  segmentWithMultiAtlas 
} from './services/segmentationService';

// Add state
const [segmentationMethod, setSegmentationMethod] = useState<
  'nonlinear' | 'deeplearning' | 'multi-atlas' | null
>(null);
const [segmentationResult, setSegmentationResult] = useState<any>(null);
const [isSegmenting, setIsSegmenting] = useState(false);

// Add handler
const handleSegmentBrain = async (method: 'nonlinear' | 'deeplearning' | 'multi-atlas') => {
  if (!primaryVolume) return;
  
  setIsSegmenting(true);
  try {
    // Convert volume to file (or send volume data directly)
    const file = await volumeToFile(primaryVolume);
    
    let result;
    if (method === 'nonlinear') {
      result = await segmentWithNonLinear(file, 'HarvardOxford', 'ANTs');
    } else if (method === 'deeplearning') {
      result = await segmentWithDeepLearning(file);
    } else {
      result = await segmentWithMultiAtlas(file, 'majority_vote');
    }
    
    setSegmentationResult(result);
    setSegmentationMethod(method);
    
    // Load segmentation into viewer
    await loadSegmentationIntoViewer(result.segmentationUrl);
  } catch (error) {
    console.error('Segmentation failed:', error);
  } finally {
    setIsSegmenting(false);
  }
};
```

### Update `VolumeViewer.tsx` to Render Segmentation

```typescript
// Add segmentation texture uniform
uniform sampler3D uSegmentation; // Label map
uniform bool uUseSegmentation;
uniform vec3 uRegionColors[50]; // Array of region colors

// In fragment shader:
if (uUseSegmentation) {
  float labelId = texture(uSegmentation, uv).r * 255.0;
  if (labelId > 0.0) {
    vec3 regionColor = uRegionColors[int(labelId)];
    // Apply region color
    baseColor = mix(baseColor, regionColor, 0.8);
  }
}
```

---

## Comparison Summary

| Method | Accuracy | Speed | Complexity | Best For |
|--------|----------|-------|------------|----------|
| **Non-linear Registration** | ⭐⭐⭐⭐⭐ | ⚡⚡ (5-30 min) | ⭐⭐⭐⭐ | Production, high accuracy needed |
| **Deep Learning** | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ (5-30 sec) | ⭐⭐⭐ | Fast, scalable, modern approach |
| **Multi-Atlas** | ⭐⭐⭐⭐⭐ | ⚡ (10-30 min) | ⭐⭐⭐⭐⭐ | Research, maximum robustness |

---

## Recommended Implementation Order

1. **Start with Deep Learning** (fastest to implement if using pre-trained models)
2. **Add Non-linear Registration** (for highest accuracy)
3. **Implement Multi-Atlas** (for research/validation)

---

## Next Steps

1. Set up backend API endpoints
2. Download required atlases/models
3. Test with sample brain scans
4. Integrate with frontend visualization
5. Add caching for faster re-segmentation
