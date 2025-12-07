"""
Receipt Processing Pipeline - Hugging Face Spaces App
Ensemble classification, OCR, field extraction, anomaly detection, and agentic routing.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import easyocr
import json
import re
from PIL import Image, ImageDraw
from datetime import datetime
from torchvision import transforms, models
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Work around Gradio json_schema traversal crash:
# - guard bool schema entries
# ---------------------------------------------------------------------------
import gradio_client.utils as grc_utils
_orig_get_type = grc_utils.get_type
_orig_json_schema_to_python_type = grc_utils.json_schema_to_python_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "any"
    return _orig_get_type(schema)


def _safe_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "any"
    try:
        return _orig_json_schema_to_python_type(schema, defs)
    except Exception:
        return "any"


grc_utils.get_type = _safe_get_type
grc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type

# ---------------------------------------------------------------------------
# JSON sanitation helper (convert numpy types & PIL-friendly outputs)
# ---------------------------------------------------------------------------
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Image.Image):
        return None
    return obj

# ---------------------------------------------------------------------------
# Feedback persistence helper (CSV; optionally include section label)
# ---------------------------------------------------------------------------
def save_feedback(assessment, notes, results_json_str, section="overall"):
    try:
        parsed = json.loads(results_json_str) if results_json_str else {}
    except Exception:
        parsed = {"raw": results_json_str}
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "section": section or "",
        "assessment": assessment or "",
        "notes": notes or "",
        "results": parsed,
    }
    import csv
    fieldnames = ["timestamp", "section", "assessment", "notes", "results"]
    file_exists = os.path.exists("feedback_logs.csv")
    with open("feedback_logs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": entry["timestamp"],
            "section": entry.get("section", ""),
            "assessment": entry["assessment"],
            "notes": entry["notes"],
            "results": json.dumps(entry["results"]),
        })
    return "✅ Feedback saved. (Stored in feedback_logs.csv)"

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models'

print(f"Device: {DEVICE}")
print(f"Models directory: {MODELS_DIR}")

# ============================================================================
# Model Classes
# ============================================================================

class DocumentClassifier:
    """ViT-based document classifier (receipt vs other)."""
    
    def __init__(self, num_labels=2, model_path=None):
        self.num_labels = num_labels
        self.model = None
        self.processor = None
        self.model_path = model_path or os.path.join(MODELS_DIR, 'rvl_classifier.pt')
        self.pretrained = 'WinKawaks/vit-tiny-patch16-224'
    
    def load_model(self):
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.pretrained)
        except:
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        self.model = ViTForImageClassification.from_pretrained(
            self.pretrained,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        self.model = self.model.to(DEVICE)
        self.model.eval()
        return self.model
    
    def load_weights(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"  Loaded ViT weights from {path}")
    
    def predict(self, image):
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred].item()
        
        is_receipt = pred == 1
        label = "receipt" if is_receipt else "other"
        
        return {
            'is_receipt': is_receipt,
            'confidence': conf,
            'label': label,
            'probabilities': probs[0].cpu().numpy().tolist()
        }


class ResNetDocumentClassifier:
    """ResNet18-based document classifier."""
    
    def __init__(self, num_labels=2, model_path=None):
        self.num_labels = num_labels
        self.model = None
        self.model_path = model_path or os.path.join(MODELS_DIR, 'resnet18_rvlcdip.pt')
        self.use_class_mapping = False
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        self.model = models.resnet18(weights=None)
        self.model = self.model.to(DEVICE)
        self.model.eval()
        return self.model
    
    def load_weights(self, path):
        if not os.path.exists(path):
            return
        
        checkpoint = torch.load(path, map_location=DEVICE)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            id2label = checkpoint.get('id2label', None)
        else:
            state_dict = checkpoint
            id2label = None
        
        # Determine number of classes from checkpoint
        fc_weight_key = 'fc.weight'
        if fc_weight_key in state_dict:
            num_classes = state_dict[fc_weight_key].shape[0]
        else:
            num_classes = self.num_labels
        
        # Rebuild final layer if needed
        if num_classes != self.model.fc.out_features:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model = self.model.to(DEVICE)
        
        self.model.load_state_dict(state_dict, strict=False)
        
        # Handle 16-class RVL-CDIP models
        if num_classes == 16:
            self.use_class_mapping = True
            self.receipt_class_idx = 11  # Receipt class in RVL-CDIP
        
        print(f"  Loaded ResNet weights from {path} ({num_classes} classes)")
    
    def predict(self, image):
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=-1)
        
        if self.use_class_mapping:
            receipt_prob = probs[0, self.receipt_class_idx].item()
            other_prob = 1.0 - receipt_prob
            is_receipt = receipt_prob > 0.5
            conf = receipt_prob if is_receipt else other_prob
            final_probs = [other_prob, receipt_prob]
        else:
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred].item()
            is_receipt = pred == 1
            final_probs = probs[0].cpu().numpy().tolist()
        
        return {
            'is_receipt': is_receipt,
            'confidence': conf,
            'label': "receipt" if is_receipt else "other",
            'probabilities': final_probs
        }


class EnsembleDocumentClassifier:
    """Ensemble of ViT and ResNet classifiers."""
    
    def __init__(self, model_configs=None, weights=None):
        self.model_configs = model_configs or [
            {'name': 'vit_base', 'path': os.path.join(MODELS_DIR, 'rvl_classifier.pt')},
            {'name': 'resnet18', 'path': os.path.join(MODELS_DIR, 'resnet18_rvlcdip.pt')},
        ]
        
        # Filter to existing models
        self.model_configs = [cfg for cfg in self.model_configs if os.path.exists(cfg['path'])]
        
        if not self.model_configs:
            print("Warning: No model files found, will use default ViT")
            self.model_configs = [{'name': 'vit_default', 'path': None}]
        
        self.weights = weights or [1.0 / len(self.model_configs)] * len(self.model_configs)
        self.classifiers = []
        self.processor = None
    
    def load_models(self):
        print(f"Loading ensemble with {len(self.model_configs)} models...")
        
        for cfg in self.model_configs:
            is_resnet = 'resnet' in cfg['name'].lower() or 'resnet' in cfg.get('path', '').lower()
            
            if is_resnet:
                classifier = ResNetDocumentClassifier(num_labels=2, model_path=cfg['path'])
            else:
                classifier = DocumentClassifier(num_labels=2, model_path=cfg['path'])
            
            classifier.load_model()
            
            if cfg['path'] and os.path.exists(cfg['path']):
                try:
                    classifier.load_weights(cfg['path'])
                except Exception as e:
                    print(f"  Warning: Could not load {cfg['name']}: {e}")
            
            self.classifiers.append(classifier)
            
            if self.processor is None:
                if hasattr(classifier, 'processor'):
                    self.processor = classifier.processor
                elif hasattr(classifier, 'transform'):
                    self.processor = classifier.transform
        
        print(f"Ensemble ready with {len(self.classifiers)} models")
        return self
    
    def predict(self, image, return_individual=False):
        if not self.classifiers:
            self.load_models()
        
        all_probs = []
        individual_results = []
        
        for i, classifier in enumerate(self.classifiers):
            result = classifier.predict(image)
            probs = result.get('probabilities', [0.5, 0.5])
            if len(probs) < 2:
                probs = [1 - result['confidence'], result['confidence']]
            all_probs.append(probs)
            individual_results.append({
                'name': self.model_configs[i]['name'],
                'prediction': result['label'],
                'confidence': result['confidence'],
                'probabilities': probs
            })
        
        # Weighted average
        ensemble_probs = np.zeros(2)
        for i, probs in enumerate(all_probs):
            ensemble_probs += np.array(probs[:2]) * self.weights[i]
        
        pred = np.argmax(ensemble_probs)
        is_receipt = pred == 1
        conf = ensemble_probs[pred]
        
        result = {
            'is_receipt': is_receipt,
            'confidence': float(conf),
            'label': "receipt" if is_receipt else "other",
            'probabilities': ensemble_probs.tolist()
        }
        
        if return_individual:
            result['individual_results'] = individual_results
        
        return result


# ============================================================================
# OCR
# ============================================================================

class ReceiptOCR:
    """EasyOCR wrapper with retry logic."""
    
    def __init__(self):
        self.reader = None
    
    def load(self):
        if self.reader is None:
            print("Loading EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("EasyOCR ready")
        return self
    
    def extract_with_positions(self, image, min_confidence=0.3):
        if self.reader is None:
            self.load()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        results = self.reader.readtext(image)
        
        extracted = []
        for bbox, text, conf in results:
            if conf >= min_confidence:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                extracted.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                })
        
        return extracted
    
    def postprocess_receipt(self, ocr_results):
        """Extract structured fields from OCR results."""
        full_text = ' '.join([r['text'] for r in ocr_results])
        
        fields = {
            'vendor': self._extract_vendor(ocr_results),
            'date': self._extract_date(full_text),
            'total': self._extract_total(full_text),
            'time': self._extract_time(full_text)
        }
        
        return fields
    
    def _extract_vendor(self, ocr_results):
        if ocr_results:
            # Usually first line is vendor
            return ocr_results[0]['text']
        return None
    
    def _extract_date(self, text):
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None
    
    def _extract_total(self, text):
        patterns = [
            r'TOTAL[:\s]*\$?(\d+\.?\d*)',
            r'AMOUNT[:\s]*\$?(\d+\.?\d*)',
            r'DUE[:\s]*\$?(\d+\.?\d*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Find largest dollar amount
        amounts = re.findall(r'\$(\d+\.\d{2})', text)
        if amounts:
            return max(amounts, key=float)
        return None
    
    def _extract_time(self, text):
        pattern = r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group() if match else None


# ============================================================================
# Anomaly Detection
# ============================================================================

class AnomalyDetector:
    """Isolation Forest-based anomaly detection."""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
    
    def extract_features(self, fields):
        """Extract features from receipt fields."""
        total = 0
        try:
            total = float(fields.get('total', 0) or 0)
        except:
            pass
        
        vendor = fields.get('vendor', '') or ''
        date = fields.get('date', '') or ''
        
        features = [
            total,
            np.log1p(total),
            len(vendor),
            1 if date else 0,
            1,  # num_items placeholder
            12,  # hour placeholder
            total,  # amount_per_item placeholder
            0  # is_weekend placeholder
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, fields):
        features = self.extract_features(fields)
        
        # Simple rule-based detection if model not fitted
        reasons = []
        total = float(fields.get('total', 0) or 0)
        
        if total > 1000:
            reasons.append(f"High amount: ${total:.2f}")
        if not fields.get('vendor'):
            reasons.append("Missing vendor")
        if not fields.get('date'):
            reasons.append("Missing date")
        
        is_anomaly = len(reasons) > 0
        
        return {
            'is_anomaly': is_anomaly,
            'score': -0.5 if is_anomaly else 0.5,
            'prediction': 'ANOMALY' if is_anomaly else 'NORMAL',
            'reasons': reasons
        }


# ============================================================================
# Initialize Models
# ============================================================================

print("\n" + "="*50)
print("Initializing models...")
print("="*50)

# Check for model files
model_files = []
if os.path.exists(MODELS_DIR):
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    print(f"Found model files: {model_files}")
else:
    print(f"Models directory not found: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize components
try:
    ensemble_classifier = EnsembleDocumentClassifier()
    ensemble_classifier.load_models()
except Exception as e:
    print(f"Warning: Could not load ensemble classifier: {e}")
    ensemble_classifier = None

try:
    receipt_ocr = ReceiptOCR()
    receipt_ocr.load()
except Exception as e:
    print(f"Warning: Could not load OCR: {e}")
    receipt_ocr = None

anomaly_detector = AnomalyDetector()

print("\n" + "="*50)
print("Initialization complete!")
print("="*50 + "\n")


# ============================================================================
# Helper Functions
# ============================================================================

def draw_ocr_boxes(image, ocr_results):
    """Draw bounding boxes on image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for r in ocr_results:
        conf = r.get('confidence', 0.5)
        bbox = r.get('bbox', [])
        
        if conf > 0.8:
            color = '#28a745'  # Green
        elif conf > 0.5:
            color = '#ffc107'  # Yellow
        else:
            color = '#dc3545'  # Red
        
        if len(bbox) >= 4:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=2)
    
    return img_copy


def process_receipt(image):
    """Main processing function for Gradio."""
    if image is None:
        return (
            "<div style='padding: 20px; text-align: center;'>Upload an image to begin</div>",
            None, "", ""
        )
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    
    results = {}
    
    # 1. Classification
    classifier_html = ""
    try:
        if ensemble_classifier:
            class_result = ensemble_classifier.predict(image, return_individual=True)
        else:
            class_result = {'is_receipt': True, 'confidence': 0.5, 'label': 'unknown'}
        
        conf = class_result['confidence']
        label = class_result['label'].upper()
        color = '#28a745' if class_result.get('is_receipt') else '#dc3545'
        bar_color = '#28a745' if conf > 0.8 else '#ffc107' if conf > 0.6 else '#dc3545'
        
        classifier_html = f"""
        <div style="padding: 16px; background: #111827; color: #e5e7eb; border-radius: 12px; margin: 8px 0; border: 1px solid #1f2937;">
            <h4 style="margin: 0 0 8px 0; color: #e5e7eb;">Classification</h4>
            <div style="font-size: 20px; font-weight: bold; color: {color};">{label}</div>
            <div style="margin-top: 8px; color: #e5e7eb;">
                <span>Confidence: </span>
                <div style="display: inline-block; width: 100px; height: 8px; background: #1f2937; border-radius: 4px;">
                    <div style="width: {conf*100}%; height: 100%; background: {bar_color}; border-radius: 4px;"></div>
                </div>
                <span style="margin-left: 8px;">{conf:.1%}</span>
            </div>
        </div>
        """
        results['classification'] = class_result
    except Exception as e:
        classifier_html = f"<div style='color: red;'>Classification error: {e}</div>"
    
    # 2. OCR
    ocr_text = ""
    ocr_image = None
    ocr_results = []
    try:
        if receipt_ocr:
            ocr_results = receipt_ocr.extract_with_positions(image)
            ocr_image = draw_ocr_boxes(image, ocr_results)
            
            lines = [f"{i+1}. [{r['confidence']:.0%}] {r['text']}" for i, r in enumerate(ocr_results)]
            ocr_text = f"Detected {len(ocr_results)} text regions:\n\n" + "\n".join(lines)
        results['ocr'] = ocr_results
    except Exception as e:
        ocr_text = f"OCR error: {e}"
    
    # 3. Field Extraction
    fields = {}
    fields_html = ""
    try:
        if receipt_ocr and ocr_results:
            fields = receipt_ocr.postprocess_receipt(ocr_results)
        
        fields_html = "<div style='padding: 16px; background: #111827; color: #e5e7eb; border-radius: 12px; border: 1px solid #1f2937;'><h4 style=\"color: #e5e7eb;\">Extracted Fields</h4>"
        for name, value in [('Vendor', fields.get('vendor')), ('Date', fields.get('date')), 
                           ('Total', f"${fields.get('total')}" if fields.get('total') else None),
                           ('Time', fields.get('time'))]:
            display = value or '<span style="color: #9ca3af;">Not found</span>'
            fields_html += f"<div style='padding: 8px; background: #0f172a; color: #e5e7eb; border: 1px solid #1f2937; border-radius: 6px; margin: 4px 0;'><b>{name}:</b> {display}</div>"
        fields_html += "</div>"
        results['fields'] = fields
    except Exception as e:
        fields_html = f"<div style='color: red;'>Extraction error: {e}</div>"
    
    # 4. Anomaly Detection
    anomaly_html = ""
    try:
        anomaly_result = anomaly_detector.predict(fields)
        status_color = '#dc3545' if anomaly_result['is_anomaly'] else '#28a745'
        status_text = anomaly_result['prediction']
        
        anomaly_html = f"""
        <div style="padding: 16px; background: #111827; color: #e5e7eb; border-radius: 12px; margin: 8px 0; border: 1px solid #1f2937;">
            <h4 style="margin: 0 0 8px 0; color: #e5e7eb;">Anomaly Detection</h4>
            <div style="font-size: 18px; font-weight: bold; color: {status_color};">{status_text}</div>
        """
        if anomaly_result['reasons']:
            anomaly_html += "<ul style='margin: 8px 0; padding-left: 20px;'>"
            for reason in anomaly_result['reasons']:
                anomaly_html += f"<li>{reason}</li>"
            anomaly_html += "</ul>"
        anomaly_html += "</div>"
        results['anomaly'] = anomaly_result
    except Exception as e:
        anomaly_html = f"<div style='color: red;'>Anomaly detection error: {e}</div>"
    
    # 5. Final Decision
    is_receipt = results.get('classification', {}).get('is_receipt', True)
    is_anomaly = results.get('anomaly', {}).get('is_anomaly', False)
    conf = results.get('classification', {}).get('confidence', 0.5)
    
    if not is_receipt:
        decision = "REJECT"
        decision_color = "#dc3545"
        reason = "Not a receipt"
    elif is_anomaly:
        decision = "REVIEW"
        decision_color = "#ffc107"
        reason = "Anomaly detected"
    elif conf < 0.7:
        decision = "REVIEW"
        decision_color = "#ffc107"
        reason = "Low confidence"
    else:
        decision = "APPROVE"
        decision_color = "#28a745"
        reason = "All checks passed"
    
    summary_html = f"""
    <div style="padding: 24px; background: linear-gradient(135deg, {decision_color}22, {decision_color}11); 
                border-left: 4px solid {decision_color}; border-radius: 12px; text-align: center;">
        <div style="font-size: 32px; font-weight: bold; color: {decision_color};">{decision}</div>
        <div style="color: #6c757d; margin-top: 8px;">{reason}</div>
    </div>
    {classifier_html}
    {anomaly_html}
    {fields_html}
    """
    
    return summary_html, ocr_image, ocr_text, json.dumps(to_jsonable(results), indent=2)


# ============================================================================
# Gradio Interface
# ============================================================================

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; background: #0b0c0e; color: #e5e7eb; }
.main-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%); 
               border-radius: 12px; color: #e5e7eb; margin-bottom: 20px; border: 1px solid #1f2937; }
.gr-button { border-radius: 10px; background: #111827; color: #e5e7eb; border: 1px solid #1f2937; }
.gr-button-primary { background: #111827; border: 1px solid #22c55e; color: #e5e7eb; }
.gr-box { border: 1px solid #1f2937; background: #111827; color: #e5e7eb; }
.gradio-accordion { border: 1px solid #1f2937 !important; background: #0f172a !important; color: #e5e7eb !important; }
.gr-markdown { color: #e5e7eb; }
.gr-textbox textarea { background: #0f172a !important; color: #e5e7eb !important; border: 1px solid #1f2937 !important; }
.gr-radio { color: #e5e7eb !important; }
"""

with gr.Blocks(title="Receipt Processing Agent", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("""
    <div class="main-header">
        <h1>Receipt Processing Agent</h1>
        <p>Ensemble classification, OCR, field extraction, and anomaly detection</p>
    </div>
    """)
    
    gr.Markdown("""
    ### How It Works
    Upload a receipt image to automatically:
    - **Classify** document type with ViT + ResNet ensemble
    - **Extract text** with EasyOCR (with bounding boxes)
    - **Extract fields** (vendor, date, total) using regex patterns
    - **Detect anomalies** with rule-based checks
    - **Route** to APPROVE / REVIEW / REJECT
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Receipt")
            input_image = gr.Image(type="pil", label="Receipt Image", height=350)
            process_btn = gr.Button("Process Receipt", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            agent_summary = gr.HTML(
                label="Results",
                value="<div style='padding: 40px; text-align: center; color: #6c757d;'>Upload an image to begin</div>"
            )
    
    with gr.Accordion("OCR Results", open=False):
        with gr.Row():
            ocr_image_output = gr.Image(label="Detected Text Regions", height=300)
            ocr_text_output = gr.Textbox(label="Extracted Text", lines=12)
    
    with gr.Accordion("Raw Results (JSON)", open=False):
        results_json = gr.Textbox(label="Full Results", lines=15)

    with gr.Accordion("Classification Feedback", open=False):
        cls_assess = gr.Radio(choices=["Correct", "Incorrect"], label="Classification correct?", value=None)
        cls_notes = gr.Textbox(label="Notes (optional)", placeholder="What should be improved or fixed?", lines=2)
        cls_status = gr.Markdown(value="")
        cls_submit = gr.Button("Submit Classification Feedback", variant="primary")
        cls_submit.click(
            fn=save_feedback,
            inputs=[cls_assess, cls_notes, results_json, gr.State("classification")],
            outputs=cls_status
        )

    with gr.Accordion("OCR Feedback", open=False):
        ocr_assess = gr.Radio(choices=["Correct", "Incorrect"], label="OCR correct?", value=None)
        ocr_notes = gr.Textbox(label="Notes (optional)", placeholder="What should be improved or fixed?", lines=2)
        ocr_status = gr.Markdown(value="")
        ocr_submit = gr.Button("Submit OCR Feedback", variant="primary")
        ocr_submit.click(
            fn=save_feedback,
            inputs=[ocr_assess, ocr_notes, results_json, gr.State("ocr")],
            outputs=ocr_status
        )

    with gr.Accordion("Field Extraction Feedback", open=False):
        fld_assess = gr.Radio(choices=["Correct", "Incorrect"], label="Fields correct?", value=None)
        fld_notes = gr.Textbox(label="Notes (optional)", placeholder="What should be improved or fixed?", lines=2)
        fld_status = gr.Markdown(value="")
        fld_submit = gr.Button("Submit Fields Feedback", variant="primary")
        fld_submit.click(
            fn=save_feedback,
            inputs=[fld_assess, fld_notes, results_json, gr.State("fields")],
            outputs=fld_status
        )

    with gr.Accordion("Anomaly Feedback", open=False):
        an_assess = gr.Radio(choices=["Correct", "Incorrect"], label="Anomaly correct?", value=None)
        an_notes = gr.Textbox(label="Notes (optional)", placeholder="What should be improved or fixed?", lines=2)
        an_status = gr.Markdown(value="")
        an_submit = gr.Button("Submit Anomaly Feedback", variant="primary")
        an_submit.click(
            fn=save_feedback,
            inputs=[an_assess, an_notes, results_json, gr.State("anomaly")],
            outputs=an_status
        )

    with gr.Accordion("Feedback", open=True):
        gr.Markdown("Review the agent output below and submit a quick assessment.")
        with gr.Row():
            feedback_assessment = gr.Radio(
                choices=["Correct", "Incorrect"],
                label="Is the response correct?",
                value=None
            )
            feedback_notes = gr.Textbox(
                label="Notes (optional)",
                placeholder="What should be improved or fixed?",
                lines=3
            )
        feedback_status = gr.Markdown(value="")
        submit_feedback = gr.Button("Submit Feedback", variant="primary")

    process_btn.click(
        fn=process_receipt,
        inputs=[input_image],
        outputs=[agent_summary, ocr_image_output, ocr_text_output, results_json]
    )
    
    submit_feedback.click(
        fn=save_feedback,
        inputs=[feedback_assessment, feedback_notes, results_json, gr.State("overall")],
        outputs=feedback_status
    )
    
    gr.Markdown("""
    ---
    ### About This Demo
    
    This is a simplified version of the full pipeline for demonstration purposes.
    The complete system includes:
    - LayoutLMv3 for advanced field extraction
    - 4-model anomaly detection ensemble
    - LangGraph agentic workflow
    - Human feedback loop with model fine-tuning
    
    **Repository**: [GitHub](https://github.com/RogueTex/StreamingDataforModelTraining)
    """)


# Launch (Spaces needs share=True when localhost is blocked)
if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )

