"""
Example API endpoints for brain segmentation.
Add these to your existing api_server.py file.
"""

from flask import Flask, request, jsonify, send_file
import tempfile
import os
import json
import subprocess
import time
import nibabel as nib
import numpy as np

app = Flask(__name__)

# ============================================================================
# 1. NON-LINEAR REGISTRATION ENDPOINT
# ============================================================================

@app.route('/segment/nonlinear', methods=['POST'])
def segment_nonlinear():
    """
    Segment brain using non-linear registration (ANTs/FSL).
    
    Request (multipart/form-data):
    - file: NIfTI file (.nii or .nii.gz)
    - atlas: 'AAL', 'HarvardOxford', or 'DesikanKilliany'
    - method: 'ANTs' or 'FSL'
    
    Response:
    - segmentation file (NIfTI)
    - metadata in headers: X-Segmentation-Metadata
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        atlas_name = request.form.get('atlas', 'HarvardOxford')
        method = request.form.get('method', 'ANTs')
        
        # Validate atlas
        valid_atlases = ['AAL', 'HarvardOxford', 'DesikanKilliany']
        if atlas_name not in valid_atlases:
            return jsonify({'error': f'Invalid atlas. Choose from: {valid_atlases}'}), 400
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_input:
            file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Get atlas paths
        atlas_dir = 'atlases'
        atlas_template = os.path.join(atlas_dir, f'{atlas_name}_cortical.nii.gz')
        atlas_labels = os.path.join(atlas_dir, f'{atlas_name}_labels.nii.gz')
        atlas_json = os.path.join(atlas_dir, f'{atlas_name}_labels.json')
        
        if not os.path.exists(atlas_template):
            return jsonify({'error': f'Atlas {atlas_name} not found. Please download atlases first.'}), 404
        
        # Run registration
        start_time = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_output:
            output_path = tmp_output.name
        
        if method == 'ANTs':
            result = run_ants_registration(input_path, atlas_template, atlas_labels, output_path)
        else:
            result = run_fsl_registration(input_path, atlas_template, atlas_labels, output_path)
        
        processing_time = time.time() - start_time
        
        # Load segmentation and extract regions
        seg_img = nib.load(output_path)
        seg_data = seg_img.get_fdata()
        unique_labels = np.unique(seg_data[seg_data > 0])
        
        regions = []
        if os.path.exists(atlas_json):
            with open(atlas_json, 'r') as f:
                label_map = json.load(f)
                for label_id in unique_labels:
                    label_id_int = int(label_id)
                    regions.append({
                        'id': label_id_int,
                        'name': label_map.get(str(label_id_int), f'Region_{label_id_int}'),
                        'voxelCount': int(np.sum(seg_data == label_id))
                    })
        
        # Prepare metadata
        metadata = {
            'regions': regions,
            'processing_time': processing_time,
            'method': method,
            'atlas': atlas_name,
            'num_regions': len(regions)
        }
        
        # Return file with metadata in headers
        response = send_file(
            output_path,
            mimetype='application/gzip',
            as_attachment=True,
            download_name=f'segmentation_{atlas_name}.nii.gz'
        )
        response.headers['X-Segmentation-Metadata'] = json.dumps(metadata)
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup temp files
        if 'input_path' in locals() and os.path.exists(input_path):
            os.unlink(input_path)


def run_ants_registration(moving_path, fixed_template, fixed_labels, output_path):
    """Run ANTs registration pipeline."""
    try:
        # Step 1: Affine registration
        affine_output = f'{output_path}_affine'
        subprocess.run([
            'antsRegistration',
            '--dimensionality', '3',
            '--float', '0',
            '--output', f'{affine_output}_',
            '--interpolation', 'Linear',
            '--winsorize-image-intensities', '[0.005,0.995]',
            '--use-histogram-matching', '0',
            '--initial-moving-transform', f'[{fixed_template},{moving_path},1]',
            '--transform', 'Affine[0.1]',
            '--metric', 'MI[{fixed_template},{moving_path},1,32,Regular,0.25]',
            '--convergence', '[1000x500x250x100,1e-6,10]',
            '--shrink-factors', '8x4x2x1',
            '--smoothing-sigmas', '3x2x1x0vox'
        ], check=True, capture_output=True)
        
        # Step 2: Non-linear SyN registration
        syn_output = f'{output_path}_syn'
        subprocess.run([
            'antsRegistration',
            '--dimensionality', '3',
            '--float', '0',
            '--output', f'{syn_output}_',
            '--interpolation', 'Linear',
            '--winsorize-image-intensities', '[0.005,0.995]',
            '--use-histogram-matching', '0',
            '--initial-moving-transform', f'{affine_output}_0GenericAffine.mat',
            '--transform', 'SyN[0.1,3,0]',
            '--metric', 'CC[{fixed_template},{moving_path},1,4]',
            '--convergence', '[100x50x30x20,1e-6,10]',
            '--shrink-factors', '8x4x2x1',
            '--smoothing-sigmas', '3x2x1x0vox'
        ], check=True, capture_output=True)
        
        # Step 3: Apply transformation to labels
        subprocess.run([
            'antsApplyTransforms',
            '--dimensionality', '3',
            '--input', fixed_labels,
            '--reference-image', moving_path,
            '--output', output_path,
            '--interpolation', 'NearestNeighbor',  # Preserve label values
            '--transform', f'{syn_output}_1Warp.nii.gz',
            '--transform', f'{syn_output}_0GenericAffine.mat'
        ], check=True, capture_output=True)
        
        return {'success': True}
    except subprocess.CalledProcessError as e:
        raise Exception(f'ANTs registration failed: {e.stderr.decode()}')


def run_fsl_registration(moving_path, fixed_template, fixed_labels, output_path):
    """Run FSL FNIRT registration."""
    try:
        # Affine registration
        affine_mat = f'{output_path}_affine.mat'
        subprocess.run([
            'flirt',
            '-in', moving_path,
            '-ref', fixed_template,
            '-omat', affine_mat,
            '-dof', '12'
        ], check=True)
        
        # Non-linear registration
        warp_field = f'{output_path}_warp.nii.gz'
        subprocess.run([
            'fnirt',
            '--in', moving_path,
            '--ref', fixed_template,
            '--aff', affine_mat,
            '--iout', output_path,
            '--cout', warp_field
        ], check=True)
        
        # Apply to labels
        subprocess.run([
            'applywarp',
            '--in', fixed_labels,
            '--ref', moving_path,
            '--out', output_path,
            '--warp', warp_field,
            '--interp', 'nn'  # Nearest neighbor for labels
        ], check=True)
        
        return {'success': True}
    except subprocess.CalledProcessError as e:
        raise Exception(f'FSL registration failed: {e.stderr.decode()}')


# ============================================================================
# 2. DEEP LEARNING SEGMENTATION ENDPOINT
# ============================================================================

# Global model instance (load once at startup)
segmentation_model = None
device = None

def load_segmentation_model():
    """Load pre-trained segmentation model."""
    global segmentation_model, device
    
    try:
        import torch
        from segmentation_model import BrainSegmentationUNet
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading segmentation model on {device}")
        
        model = BrainSegmentationUNet(in_channels=1, num_classes=50, base_channels=32)
        checkpoint_path = 'checkpoints/brain_segmentation_model.pth'
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            segmentation_model = model
            print(f"Segmentation model loaded successfully")
        else:
            print("Warning: Model checkpoint not found. Please train or download model.")
            return False
        
        return True
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        return False


@app.route('/segment/deeplearning', methods=['POST'])
def segment_deeplearning():
    """
    Segment brain using deep learning model.
    
    Request (multipart/form-data):
    - file: NIfTI file
    
    Response:
    - segmentation file (NIfTI)
    - metadata in headers: X-Segmentation-Metadata
    """
    try:
        if segmentation_model is None:
            if not load_segmentation_model():
                return jsonify({'error': 'Segmentation model not available'}), 503
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_input:
            file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Load original volume
        img = nib.load(input_path)
        original_shape = img.shape[:3]
        
        # Preprocess
        from segmentation_model import preprocess_volume, postprocess_segmentation
        import torch
        import torch.nn.functional as F
        
        preprocessed = preprocess_volume(input_path, target_shape=(128, 128, 128))
        input_tensor = torch.from_numpy(preprocessed).unsqueeze(0).to(device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            output = segmentation_model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = output.cpu().numpy()[0]
        
        # Post-process
        segmentation = postprocess_segmentation(prediction, original_shape)
        processing_time = time.time() - start_time
        
        # Calculate confidence
        prob_array = probabilities.cpu().numpy()[0]
        max_probs = np.max(prob_array, axis=0)
        avg_confidence = float(np.mean(max_probs))
        
        # Extract regions
        unique_labels = np.unique(segmentation[segmentation > 0])
        regions = []
        label_map_path = 'atlases/brain_regions.json'
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
                for label_id in unique_labels:
                    label_id_int = int(label_id)
                    regions.append({
                        'id': label_id_int,
                        'name': label_map.get(str(label_id_int), f'Region_{label_id_int}'),
                        'voxelCount': int(np.sum(segmentation == label_id))
                    })
        
        # Save segmentation
        seg_img = nib.Nifti1Image(segmentation, img.affine, img.header)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_output:
            output_path = tmp_output.name
            nib.save(seg_img, output_path)
        
        metadata = {
            'regions': regions,
            'processing_time': processing_time,
            'method': 'deep_learning',
            'confidence': avg_confidence,
            'num_regions': len(regions)
        }
        
        response = send_file(
            output_path,
            mimetype='application/gzip',
            as_attachment=True,
            download_name='segmentation_dl.nii.gz'
        )
        response.headers['X-Segmentation-Metadata'] = json.dumps(metadata)
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# 3. MULTI-ATLAS FUSION ENDPOINT
# ============================================================================

from multi_atlas_fusion import MultiAtlasFusion

# Initialize fusion engine (do this at startup)
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
    
    Request (multipart/form-data):
    - file: NIfTI file
    - method: 'majority_vote', 'weighted_average', or 'probabilistic'
    
    Response:
    - segmentation file (NIfTI)
    - metadata in headers: X-Segmentation-Metadata
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        method = request.form.get('method', 'majority_vote')
        
        valid_methods = ['majority_vote', 'weighted_average', 'probabilistic']
        if method not in valid_methods:
            return jsonify({'error': f'Invalid method. Choose from: {valid_methods}'}), 400
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_input:
            file.save(tmp_input.name)
            input_path = tmp_input.name
        
        # Run fusion
        start_time = time.time()
        segmentation, metadata = fusion_engine.fuse_segmentations(input_path, method)
        processing_time = time.time() - start_time
        
        # Save segmentation
        ref_img = nib.load(input_path)
        seg_img = nib.Nifti1Image(segmentation, ref_img.affine, ref_img.header)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_output:
            output_path = tmp_output.name
            nib.save(seg_img, output_path)
        
        # Add processing time to metadata
        metadata['processing_time'] = processing_time
        metadata['method'] = 'multi_atlas_fusion'
        
        response = send_file(
            output_path,
            mimetype='application/gzip',
            as_attachment=True,
            download_name='segmentation_multi_atlas.nii.gz'
        )
        response.headers['X-Segmentation-Metadata'] = json.dumps(metadata)
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    # Load models at startup
    print("Initializing segmentation services...")
    
    # Try to load deep learning model (optional)
    try:
        load_segmentation_model()
    except Exception as e:
        print(f"Could not load DL model: {e}")
    
    print("Segmentation API ready!")
    app.run(debug=True, port=5000)
