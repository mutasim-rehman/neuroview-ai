# How the API Detects Diseases: Autoencoder vs Classification

## Two Different Approaches

### 1. **Autoencoder (Anomaly Detection)** - What the API Uses
**Script:** `main_train_healthy.py`  
**Model:** `BrainScanModel` (encoder-decoder architecture)

**How it works:**
- ✅ Trained **only on healthy brain scans**
- ✅ Learns to reconstruct healthy brain patterns
- ✅ When given a new scan:
  - **Low reconstruction error** → Scan looks healthy (model can reconstruct it well)
  - **High reconstruction error** → Scan has defects/anomalies (model can't reconstruct it because it learned only healthy patterns)

**Detection Logic (from `api_server.py`):**
```python
# Calculate reconstruction error
anomaly_score = error_metrics['mse']

# If MSE > 0.01, likely anomaly
is_healthy = anomaly_score < 0.01
```

**Advantages:**
- Only needs healthy data (easier to collect)
- Can detect ANY anomaly, even unseen diseases
- Works as "one-class classification"

**Output:**
- `prediction: "healthy"` or `"defect"`
- `anomaly_score`: How different from healthy patterns
- `confidence`: Based on reconstruction error

---

### 2. **Classification Model** - NOT Used by API
**Script:** `main_train.py` or `main_train_diseases.py`  
**Model:** `BrainTumorClassificationModel`

**How it works:**
- ❌ Requires labeled data (healthy + specific disease types)
- ❌ Trained to classify: "Healthy", "Glioma", "Meningioma", "Pituitary", etc.
- ❌ Can only detect diseases it was trained on

**Output:**
- Class probabilities for each disease type
- Can't detect new/unknown diseases

---

## Why the API Uses Autoencoder

The API endpoint `/predict` uses **anomaly detection** because:

1. **More flexible**: Detects any abnormality, not just specific diseases
2. **Easier data collection**: Only needs healthy scans for training
3. **Better for screening**: Binary decision (healthy vs defect) is often sufficient
4. **Generalizable**: Works even if new disease types appear

## Current API Behavior

Looking at `api_server.py` line 165:
```python
# Simple threshold: if MSE > 0.01, likely anomaly
is_healthy = anomaly_score < 0.01
```

The API:
- ✅ Returns `"healthy"` if reconstruction error < 0.01
- ✅ Returns `"defect"` if reconstruction error >= 0.01
- ✅ Provides `anomaly_score` and `confidence` for more detailed analysis

## Summary

| Aspect | Autoencoder (API) | Classification |
|--------|------------------|----------------|
| Training Data | Healthy only | Healthy + Disease types |
| Output | Healthy/Defect | Specific disease class |
| Can detect new diseases? | ✅ Yes | ❌ No |
| Use case | Screening, anomaly detection | Specific diagnosis |
| Script | `main_train_healthy.py` | `main_train.py` |

**For the API, you need `main_train_healthy.py`** - it trains the autoencoder that detects diseases by reconstruction error!

