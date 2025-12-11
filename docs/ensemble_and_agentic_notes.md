# Ensemble Methods & Agentic Pipeline - Technical Notes

> Complete technical reference for presentation and speaker notes

---

## Part 1: Ensemble Combining Methods

We use **4 different ensembles** in the pipeline, each with different combining strategies.

---

### 1. Classification Ensemble (ViT + ResNet + Stacking)

**Architecture**: Two-level stacking ensemble

```
Level 0 (Base Models):
├── ViT-Tiny Classifier
├── ResNet18 Classifier  
└── ViT-10k (fine-tuned)
         ↓
Level 1 (Meta-Learners):
├── XGBoost
├── Logistic Regression
└── Random Forest
         ↓
      Final Decision
```

**How it combines:**

1. **Feature Extraction** from each base model:
   - Probability score for "receipt" class
   - Prediction confidence
   - Agreement score (how much models agree)

2. **Meta-learner Training**:
   - Takes probability outputs from Level 0
   - Learns optimal combination weights
   - Uses cross-validation to select best meta-learner

3. **Final Prediction**:
   ```python
   # Extract features from base models
   features = [vit_prob, resnet_prob, vit10k_prob, vit_conf, resnet_conf, ...]
   
   # Scale features
   features_scaled = scaler.transform(features)
   
   # Meta-learner predicts
   final_prob = xgboost_meta.predict_proba(features_scaled)
   ```

**Key Formula**:
```
P(receipt) = f_meta(P_vit, P_resnet, P_vit10k, conf_vit, conf_resnet, ...)
```
Where `f_meta` is a learned function (XGBoost, LogReg, or RandomForest).

---

### 2. Field Extraction Ensemble (LayoutLM + Regex + Position + NER)

**Architecture**: Confidence-weighted voting

```
OCR Results
    ↓
┌───────────────────────────────────────────────┐
│  LayoutLMv3 (35%)  │  Regex (25%)             │
│  Position (20%)    │  NER (20%)               │
└───────────────────────────────────────────────┘
    ↓
Confidence-Weighted Voting
    ↓
{vendor, date, total}
```

**How it combines:**

1. **Each method extracts independently**:
   ```python
   layoutlm_result = layoutlm.predict(image, ocr_results)  # {vendor: "STARBUCKS", conf: 0.92}
   regex_result = regex.extract(ocr_text)                   # {date: "12/07/2025", conf: 0.85}
   position_result = position.extract(ocr_results)          # {total: 45.99, conf: 0.78}
   ner_result = spacy_ner.extract(ocr_text)                 # {vendor: "Starbucks", conf: 0.65}
   ```

2. **Weighted voting formula**:
   ```
   score(field, value) = Σ (weight_i × conf_i × match_i)
   
   Where:
   - weight_i = method weight (0.35, 0.25, 0.20, 0.20)
   - conf_i = method's confidence for this extraction
   - match_i = 1 if method extracted this value, 0 otherwise
   ```

3. **Agreement bonus**:
   ```python
   if multiple_methods_agree(value):
       score *= 1.2  # 20% bonus for agreement
   ```

4. **Final selection**:
   ```python
   final_vendor = max(candidates, key=lambda x: x['score'])
   ```

---

### 3. Anomaly Detection Ensemble (Isolation Forest + XGBoost + HistGradient + SVM)

**Architecture**: Weighted voting with majority rule

```
Receipt Features (8 values)
         ↓
┌────────────────────────────────────────────────┐
│ Isolation Forest (35%)  │  XGBoost (30%)       │
│ HistGradientBoosting (20%) │ One-Class SVM (15%) │
└────────────────────────────────────────────────┘
         ↓
    Weighted Average Score + Majority Vote
         ↓
    NORMAL / ANOMALY
```

**How it combines:**

1. **Individual predictions**:
   ```python
   # Each model returns anomaly score
   iso_score = isolation_forest.decision_function(features)  # Higher = more normal
   xgb_score = xgboost.predict_proba(features)[:, 1]         # P(anomaly)
   hgb_score = histgb.predict_proba(features)[:, 1]          # P(anomaly)
   svm_score = one_class_svm.decision_function(features)     # Distance from boundary
   ```

2. **Weighted average formula**:
   ```python
   weights = {'isolation_forest': 0.35, 'xgboost': 0.30, 
              'histgb': 0.20, 'one_class_svm': 0.15}
   
   ensemble_score = Σ (weight_i × score_i) / Σ weight_i
   ```

3. **Majority voting**:
   ```python
   votes_anomaly = sum(1 for model in models if model.is_anomaly)
   is_anomaly = votes_anomaly >= len(models) / 2  # At least 2 of 4
   ```

4. **Final decision**:
   ```python
   # Both conditions must be met
   is_anomaly = (ensemble_score > threshold) AND (votes_anomaly >= 2)
   ```

---

### 4. OCR Ensemble (EasyOCR + TrOCR + PaddleOCR + Tesseract)

**Architecture**: Confidence-weighted text merging

```
Receipt Image
      ↓
┌─────────────────────────────────────┐
│ EasyOCR │ TrOCR │ PaddleOCR │ Tesseract │
└─────────────────────────────────────┘
      ↓
  Spatial Matching + Confidence Voting
      ↓
  Merged OCR Results
```

**How it combines:**

1. **Spatial matching**:
   - Find overlapping bounding boxes (IoU > 0.5)
   - Group text from same regions

2. **Text voting**:
   ```python
   for region in regions:
       candidates = get_texts_from_all_engines(region)
       
       # Score each text candidate
       text_scores = {}
       for text, conf, engine_weight in candidates:
           cleaned_text = fix_common_errors(text)
           text_scores[cleaned_text] += conf * engine_weight
       
       # Select highest scored text
       final_text = max(text_scores, key=text_scores.get)
   ```

3. **No single-engine dominance**:
   - Maximum weight for any engine = 40%
   - Ensures diverse perspectives

---

## Summary: Combining Strategies

| Ensemble | Method | Formula |
|----------|--------|---------|
| **Classification** | Stacking (Meta-learner) | `f_meta(P₁, P₂, P₃, conf₁, conf₂, ...)` |
| **Field Extraction** | Weighted Voting | `Σ(weight × conf × match) + agreement_bonus` |
| **Anomaly Detection** | Weighted Average + Majority Vote | `avg(w_i × s_i) AND votes ≥ 2` |
| **OCR** | Spatial Matching + Confidence Voting | `max(Σ(conf × weight))` per region |

---

## Part 2: Agentic Pipeline (LangGraph)

### What Makes It "Agentic"?

Traditional pipeline:
```
Input → Step 1 → Step 2 → Step 3 → Output
        (fails if any step fails)
```

Agentic pipeline:
```
Input → Agent 1 ──┬──→ Agent 2 ──┬──→ Agent 3 → Output
                  │              │
                  ↓              ↓
              Retry with     Fallback
              enhancement    strategy
```

**Key agentic properties:**
1. **State management** - Shared state passes through all nodes
2. **Conditional routing** - Different paths based on results
3. **Error handling** - Graceful degradation, not crashes
4. **Retry logic** - Automatic retry with image enhancement
5. **Human-in-the-loop** - Routes uncertain cases for review

---

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentState                                │
│  {image, classification, ocr_results, extracted_fields,         │
│   anomaly_result, decision, confidence_score, processing_log}   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────┐    ┌──────────┐    ┌─────┐    ┌─────────┐    ┌─────────┐    ┌───────┐
│ INGEST  │ →  │ CLASSIFY │ →  │ OCR │ →  │ EXTRACT │ →  │ ANOMALY │ →  │ ROUTE │
└─────────┘    └──────────┘    └─────┘    └─────────┘    └─────────┘    └───────┘
     ↓              ↓             ↓            ↓              ↓            ↓
  Load &        Is it a       Extract      LayoutLM +      4-model      Final
  validate      receipt?       text        Regex merge    ensemble    decision
```

---

### Node Definitions

#### 1. Ingestion Node
```python
def ingestion_node(state: AgentState) -> AgentState:
    """Load and prepare the image"""
    # Load from path or use provided image
    # Convert to RGB if needed
    # Validate image exists
    return state
```

#### 2. Classification Node
```python
def classification_node(state: AgentState) -> AgentState:
    """Determine document type"""
    result = doc_classifier.predict(image)
    state['classification'] = {
        'is_receipt': result['is_receipt'],
        'confidence': result['confidence'],
        'label': result['label']
    }
    return state
```

#### 3. OCR Node
```python
def ocr_node(state: AgentState) -> AgentState:
    """Extract text with positions"""
    ocr_results = receipt_ocr.extract_with_positions(image)
    state['ocr_results'] = ocr_results  # [{text, bbox, confidence}, ...]
    state['ocr_text'] = join_all_text(ocr_results)
    return state
```

#### 4. Extraction Node
```python
def extraction_node(state: AgentState) -> AgentState:
    """Extract structured fields"""
    # LayoutLM extraction
    layoutlm_fields = field_extractor.predict(image, ocr_results)
    
    # Regex fallback
    ocr_fields = receipt_ocr.postprocess_receipt(ocr_results)
    
    # Merge: prefer LayoutLM, fallback to regex
    fields = {
        'vendor': layoutlm_fields.get('vendor') or ocr_fields.get('vendor'),
        'date': layoutlm_fields.get('date') or ocr_fields.get('date'),
        'total': layoutlm_fields.get('total') or ocr_fields.get('total')
    }
    state['extracted_fields'] = fields
    return state
```

#### 5. Anomaly Node
```python
def anomaly_node(state: AgentState) -> AgentState:
    """Check for suspicious patterns"""
    result = anomaly_detector.predict(extracted_fields)
    state['anomaly_result'] = {
        'is_anomaly': result['is_anomaly'],
        'score': result['score'],
        'reasons': result['reasons']  # Human-readable explanations
    }
    return state
```

#### 6. Routing Node (Decision Agent)
```python
def routing_node(state: AgentState) -> AgentState:
    """Make final decision"""
    is_receipt = state['classification']['is_receipt']
    confidence = state['classification']['confidence']
    is_anomaly = state['anomaly_result']['is_anomaly']
    
    # Decision logic
    if not is_receipt:
        decision = "REJECT"
        reason = "Not a receipt/invoice"
    
    elif is_anomaly:
        decision = "REVIEW"
        reason = "Anomaly detected - requires human review"
    
    elif confidence > 0.9:
        decision = "APPROVE"
        reason = "High confidence, no anomalies"
    
    elif confidence > 0.7:
        decision = "APPROVE"
        reason = "Acceptable confidence"
    
    else:
        decision = "REVIEW"
        reason = "Low confidence - requires review"
    
    state['decision'] = decision
    return state
```

---

### Workflow Graph (LangGraph)

```python
from langgraph.graph import StateGraph, END

# Create graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("ingest", ingestion_node)
workflow.add_node("classify", classification_node)
workflow.add_node("ocr", ocr_node)
workflow.add_node("extract", extraction_node)
workflow.add_node("anomaly", anomaly_node)
workflow.add_node("route", routing_node)

# Define edges (flow)
workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "classify")
workflow.add_edge("classify", "ocr")
workflow.add_edge("ocr", "extract")
workflow.add_edge("extract", "anomaly")
workflow.add_edge("anomaly", "route")
workflow.add_edge("route", END)

# Compile
receipt_agent = workflow.compile()
```

---

### Conditional Branching

```python
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Early exit conditions"""
    
    # Error occurred - stop
    if state.get('error'):
        return "end"
    
    # Not a receipt - skip OCR/extraction
    if not state['classification']['is_receipt']:
        return "end"
    
    return "continue"

# Add conditional edge
workflow.add_conditional_edges(
    "classify",
    should_continue,
    {
        "continue": "ocr",
        "end": "route"  # Skip directly to decision
    }
)
```

---

### Retry Logic (Enhanced Workflow)

```python
def ocr_with_retry(state: EnhancedAgentState) -> EnhancedAgentState:
    """OCR with automatic retry on low confidence"""
    
    max_retries = 3
    retry_count = state.get('ocr_retry_count', 0)
    
    # First attempt
    ocr_results = receipt_ocr.extract_with_positions(state['image'])
    avg_confidence = mean([r['confidence'] for r in ocr_results])
    
    # Retry if confidence too low
    if avg_confidence < 0.7 and retry_count < max_retries:
        # Enhance image
        enhanced = enhance_image(state['image'])
        state['image'] = enhanced
        state['ocr_retry_count'] = retry_count + 1
        
        # Retry OCR
        ocr_results = receipt_ocr.extract_with_positions(enhanced)
    
    state['ocr_results'] = ocr_results
    return state

def enhance_image(image):
    """Apply enhancements for better OCR"""
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    return image
```

---

### Human-in-the-Loop Feedback

```python
class FeedbackCollector:
    """Collects human corrections and triggers retraining"""
    
    def __init__(self, retrain_threshold=5):
        self.feedback_queue = []
        self.retrain_threshold = retrain_threshold
    
    def add_feedback(self, receipt_id, correction):
        """Store human correction"""
        self.feedback_queue.append({
            'receipt_id': receipt_id,
            'original': correction['original'],
            'corrected': correction['corrected'],
            'field': correction['field']
        })
        
        # Trigger retraining if enough feedback
        if len(self.feedback_queue) >= self.retrain_threshold:
            self.trigger_retrain()
    
    def trigger_retrain(self):
        """Fine-tune models with new data"""
        # Update LayoutLM with corrected extractions
        # Retrain anomaly detector with new labels
        # Adjust ensemble weights based on accuracy
        pass
```

---

### Decision Flow Summary

```
                    ┌──────────────┐
                    │    INPUT     │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │   CLASSIFY   │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
            NO ←────│  Is Receipt? │────→ YES
            │       └──────────────┘       │
            ↓                              ↓
      ┌──────────┐                  ┌────────────┐
      │  REJECT  │                  │    OCR     │
      └──────────┘                  └──────┬─────┘
                                           ↓
                                    ┌────────────┐
                                    │  EXTRACT   │
                                    └──────┬─────┘
                                           ↓
                                    ┌────────────┐
                                    │  ANOMALY   │
                                    └──────┬─────┘
                                           ↓
                              ┌────────────────────┐
                      YES ←───│  Anomaly Detected? │───→ NO
                       │      └────────────────────┘      │
                       ↓                                  ↓
                 ┌──────────┐                    ┌────────────────┐
                 │  REVIEW  │            YES ←───│ Confidence>70%? │───→ NO
                 └──────────┘             │      └────────────────┘      │
                                          ↓                              ↓
                                    ┌──────────┐                  ┌──────────┐
                                    │ APPROVE  │                  │  REVIEW  │
                                    └──────────┘                  └──────────┘
```

---

## Quick Reference for Presentation

### "How do we combine predictions?"

| Component | One-liner |
|-----------|-----------|
| **Classification** | "XGBoost learns the optimal way to combine ViT and ResNet predictions" |
| **Field Extraction** | "Weighted voting - LayoutLM gets 35%, regex gets 25%, etc." |
| **Anomaly** | "Weighted average + majority vote - at least 2 of 4 must flag it" |
| **OCR** | "Find overlapping regions, vote on best text by confidence" |

### "What makes it agentic?"

1. **State management** - All data flows through shared state
2. **Conditional routing** - Non-receipts skip to rejection
3. **Retry logic** - Auto-enhances image if OCR fails
4. **Human feedback** - Corrections improve the system
5. **Explainability** - Every decision has a reason

---

*Document generated for Agentic Receipt Tool presentation - December 2025*



