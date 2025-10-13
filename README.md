<h1 align="center">
  SOS: Synthetic Object Segments Improve Detection, Segmentation, and Grounding
</h1>

<h2 align="center" style="color: #ef4444;">
  1 Million Diverse, Accurate Synthetic Dense-Annotated Images + 20M Synthetic Objects to Supercharge Grounding-DINO, Mask2Former, and Any Detectors / Segmentors / Grounding-VLMs
</h2>


<h2 align="center">
  <!-- <a href="https://generate-any-scene.github.io/">🌐 Website</a> | -->
  <a href="https://arxiv.org/abs/2510.09110">📑 Paper</a> |
  <a href="https://huggingface.co/collections/weikaih/sos-synthetic-object-segments-improves-detection-segmentat-682679751d20faa20800033c">🤗  Datasets: 1M images + 20M segments</a>
</h2>

<p align="center"><small>
  Weikai Huang<sup>1</sup>, Jieyu Zhang<sup>1</sup>,
  Taoyang Jia<sup>1</sup>, Chenhao Zheng<sup>1</sup>, Ziqi Gao<sup>1</sup>,
  Jae Sung Park<sup>1</sup>, Ranjay Krishna<sup>1,2</sup><br>
  <sup>1</sup>&nbsp University of Washington &nbsp
  <sup>2</sup>Allen Institute for AI
</small></p>

---

<p align="center">
  <img src="./assets/teaser.png" alt="Text-to-Image Results" width="800">
</p>
<p align="center">A scalable pipeline for composing high-quality synthetic object segments into richly annotated images for object detection, instance segmentation, and visual grounding.</p>

## 🌟 Highlights

**Why SOS?** A small amount of high-quality synthetic data can outperform orders of magnitude more real data:

- 🚀 **Efficient & Scalable**: Just **50K** SOS images match the gains from **20M** model-generated (GRIT) or **200K** human-annotated (V3Det) images on LVIS detection. We scale to 10 million diverse images with annotations, along with 20 million synthetic objects across 47,000+ categories from Flux.
- 🎯 **Accurate Annotations**: Object-centric composition provides pixel-perfect masks, boxes, and referring expressions—no noisy pseudo-labels
- 🎨 **Controllable Generation**: Synthesize targeted data for specific scenarios (e.g., intra-class referring, rare categories, domain-specific applications)
- 🔄 **Complementary to Real Data**: Adding SOS to existing datasets (COCO, LVIS, V3Det, GRIT) yields consistent additive gains across all benchmarks
- 💰 **Cost-Effective**: Generate unlimited training data from 20M object segments without expensive human annotation

---

# 📊 Released Datasets

We release the following datasets for research use:

| Dataset Name | # Images | # Categories | Description | Download |
|-------------|----------|--------------|-------------|----------|
| **FC-1M** | 1,000,000 | 1,600 | Frequent Categories | [🤗 HuggingFace](https://huggingface.co/datasets/weikaih/SOS-FC-1M) |
| **GC-1M** | 1,000,000 | 47,000+ | General Categories | [🤗 HuggingFace](https://huggingface.co/datasets/weikaih/SOS-GC-1M) |
| **SFC-200K** | 200,000 | 1,600 | Single-category Frequent Category — same category objects with varied attributes | [🤗 HuggingFace](https://huggingface.co/datasets/weikaih/SOS-SFC-200K) |
| **SGC-200K** | 200,000 | 47,000+ | Single-category General Category — same category objects with varied attributes | [🤗 HuggingFace](https://huggingface.co/datasets/weikaih/SOS-SGC-200K) |


<p align="center"><small><b>Examples</b> of dataset types:</small></p>
<table>
  <tr>
    <td align="center">
      <img src="./assets/fcgc.png" alt="FC/GC examples" width="400"><br>
      <sub><b>FC / GC</b></sub>
    </td>
    <td align="center">
      <img src="./assets/sfcsgc.png" alt="SFC/SGC examples" width="400"><br>
      <sub><b>SFC / SGC</b></sub>
    </td>
  </tr>
</table>

**All datasets include:**
- ✅ High-resolution images with photorealistic relighting and blending
- ✅ Pixel-perfect segmentation masks
- ✅ Tight bounding boxes
- ✅ Category labels
- ✅ Diverse referring expressions (attribute-based, spatial-based, and mixed)

**Note:** Other dataset variants (e.g., SOS-LVIS, MixCOCO) contain segments from existing datasets and cannot be released. Please use the code in this repository to compose your own datasets from the released object segments.

## Object Segments

We also release **20M synthetic object segments** used to compose the above datasets:

| Segment Set | # Segments | # Categories | Prompts/Category | Segments/Prompt | Download |
|-------------|------------|--------------|------------------|-----------------|----------|
| **FC Object Segments** | 10,000,000 | 1,600 | 200 | 3 | [🤗 SOS-FC-Object-Segments-10M](https://huggingface.co/datasets/weikaih/SOS-FC-Object-Segments-10M) |
| **GC Object Segments** | 10,000,000 | 47,000+ | 10 | 3 | [🤗 SOS-GC-Object-Segments-10M](https://huggingface.co/datasets/weikaih/SOS-GC-Object-Segments-10M) |

Browse all sets via the collection: [🤗 HuggingFace Collection](https://huggingface.co/collections/weikaih/sos-synthetic-object-segments-improves-detection-segmentat-682679751d20faa20800033c)

---

# 📦 Installation

*Notice*: We provide only minimal guidance for the core parts of the codebase for: image composing, relighting and blending, and referring expression generation. The full documentation (with an accompanying arXiv paper) covering additional tasks and case studies will be released soon.

## Environment Setup
Follow the steps below to set up the environment and use the repository:
```bash
# Clone the repository
git clone https://github.com/weikaih04/SOS
cd ./SOS

# Create and activate a Python virtual environment:
conda create -n sos python==3.10
conda activate sos

# Install the required dependencies for composing images with synthetic object segments:
pip install -r requirements.txt

# If you want to perform relighting and blending:
conda create -n sos-relight python==3.10
conda activate sos-relight
pip install -r requirements_relight_and_blend.txt

# If you want to generating referring expression:
conda create -n sos-ref python==3.10
conda activate sos-ref
pip install -r requirements_referring_expression_generation.txt
```


## Background Dataset (Optional)
If you want to relight images and didn't directly paste object segments into the background, just use a random image as the background and set the `hasBackground` to false in the `generate_batch.py`

You can download the BG-20K from this repo: https://github.com/JizhiziLi/GFM.git

# 🚀 Usage

## Composing Synthetic Images
We provide scripts to compose images with synthetic segments:

If you want to generate images for relighting and blending that only contain foreground object segments:
```bash
python scripts/generate_with_batch.py \
    --num_processes 100 \  # depends on your CPUs
    --total_images 100000 \
    --filtering_setting filter_0 \
    --image_save_path "/output/dataset_name/train" \
    --mask_save_path "/output/dataset_name/panoptic_train" \
    --annotation_path "/output/dataset_name/annotations" \
    --json_save_path "/output/dataset_name/annotations/panoptic_train.json"
```


### Key parameters
- --num_processes: Number of parallel workers to generate images; set based on CPU cores.
- --total_images: Total images to generate.
- --filtering_setting: One of filter_0..filter_4 (filter_4 = strictest). Controls segment quality filters.
- --image_save_path: Output path for rendered RGBA images (PNG).
- --mask_save_path: Output path for color panoptic masks (PNG).
- --annotation_path: Output folder for per-image JSONs and category maps.
- --json_save_path: Final merged COCO-style panoptic JSON path.

Important: At the end of scripts/generate_with_batch.py, available_object_datasets must point to your local copies of released FC/GC object segments and their metadata JSON. For example, if you downloaded SOS-FC-Object-Segments-10M to /data/fc_10m with metadata fc_object_segments_metadata.json, set:
- dataset_path="/data/fc_10m"
- synthetic_annotation_path="/data/fc_10m/fc_object_segments_metadata.json"
Similarly for GC: gc_object_segments_metadata.json




Notes
- We expect dataset_path to contain category/subcategory/ID.png structure as provided in our released object-segment datasets.
- The script writes per-image JSONs under annotation_path/separate_annotations and merges them into the final COCO-style panoptic JSON at json_save_path.

Minimal example
```bash
# Symlink your datasets to the default paths expected by the script (optional)
ln -s /data/fc_10m /fc_10m
ln -s /data/gc_10m /gc_10m

# Generate a tiny sample dataset locally
python scripts/generate_with_batch.py \
  --num_processes 4 \
  --total_images 20 \
  --filtering_setting filter_0 \
  --image_save_path "./out/train" \
  --mask_save_path "./out/panoptic_train" \
  --annotation_path "./out/annotations" \
  --json_save_path "./out/annotations/panoptic_train.json"
```

If you want to generate images that directly paste objects onto backgrounds, uncomment the `with bg process_image_worker` function in `scripts/generate_with_batch.py`.

## Relighting and Blending
Relight and blend images using IC-Light with mask-area-weighted blending to enhance photorealism while preserving object details and colors:

```bash
python relighting_and_blending/inference.py \
  --dataset_path "$DATASET_PATH" \
  --output_data_path "$OUTPUT_DATA_PATH" \
  --num_splits "$NUM_SPLITS" \
  --split "$SPLIT" \
  --index_json_path "" \
  --illuminate_prompts_path "$ILLUMINATE_PROMPTS_PATH" \
  --record_path "$RECORD_PATH"
```

Currently supports Google Cloud Storage access and local file system.

Notes
- Requires a CUDA GPU. Models load in half precision; 12GB+ VRAM recommended.
- Weights auto-download on first run:
  - Stable Diffusion components from stablediffusionapi/realistic-vision-v51
  - Background remover briaai/RMBG-1.4
  - IC-Light offset iclight_sd15_fc.safetensors (downloaded to ./models if missing)
- Input expectations:
  - dataset_path should point to the folder with RGBA foreground PNGs (e.g., ./out/train) named 0.png, 1.png, ...
  - A matching color panoptic mask must exist at the same id under dataset_path with "train" replaced by "panoptic_train" (e.g., ./out/panoptic_train/0.png)
- illuminate_prompts_path must be a JSON file containing an array of prompt strings for relighting.

Minimal example
````bash
# Create a tiny illumination prompt list
cat > ./illumination_prompt.json << 'JSON'
[
  "golden hour lighting, soft shadows",
  "overcast daylight, diffuse light",
  "studio softbox lighting"
]
JSON

# Relight a small sample from the composed outputs
python relighting_and_blending/inference.py \
  --dataset_path ./out/train \
  --output_data_path ./out/relit \
  --num_splits 1 \
  --split 0 \
  --illuminate_prompts_path ./illumination_prompt.json
````


## Referring Expression Generation
We use an OpenAI-compatible endpoint (vLLM) and query a local model.

Step 1) Start an OpenAI-compatible server (port 8080)
````bash
# Example: start vLLM OpenAI server with the model used in our script
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/QwQ-32B-AWQ \
  --host 0.0.0.0 \
  --port 8080
````
Notes
- Our script currently assumes base_url=http://localhost:8080/v1.
- Ensure your GPU/driver supports the chosen model; adjust model name if needed.

Step 2) Run the generator
````bash
# INPUT_FILE is the merged COCO-style JSON from the composing stage
# OUTPUT_DIR will contain jsonl shards (one per job): job_0.jsonl, ...
export OPENAI_API_KEY=dummy_key  # any non-empty string is accepted

python referring_expression_generation/inference.py \
  1 \
  0 \
  ./out/annotations/panoptic_train.json \
  ./out/refexp \
  --api_key "$OPENAI_API_KEY" \
  --num_workers 8
````
Outputs
- At least 9 expressions per image (balanced across attribute/spatial/reasoning, single/multi).
- Writes per-job jsonl files under OUTPUT_DIR.
- Supports local paths and GCS (gs://) for both inputs and outputs.



# 🔬 Method
<p align="center">
  <img src="./assets/pipeline.png" alt="SOS Pipeline" width="800">
</p>

Our pipeline consists of four key stages:

### 1. Object Segments Generation
- **Text Prompt Generation**: Use Qwen 2.5-32B to generate diverse text descriptions for each object category
- **Image Synthesis**: Prompt FLUX-1-dev to render single-object images on a plain white background with random viewpoints
- **Mask Extraction**: Apply DIS (Dichotomous Image Segmentation) to extract clean object masks
- **Result**: 20M high-quality segments covering:
  - **10M Frequent Categories**: 1.6K categories from LVIS, COCO, ADE20K (200 prompts × 3 seeds each)
  - **10M General Categories**: ~40K categories from LAION, GQA, Flickr30K (10 prompts × 3 seeds each)

### 2. Object Selection & Layout Generation
- **Object Count**: Sample 5–20 segments per image, matching real-photo distributions from COCO and SA-1B
- **Balanced Sampling**: Two-stage uniform sampling (category → segment) to avoid head-class bias
- **Size Distribution**: Assign segments to small/medium/large bins (40%/35%/25%) following COCO statistics
- **Spatial Layout**: Uniformly sample center coordinates with maximum overlap constraints
- **Performance**: Our layout (AP 9.16) outperforms random (9.07) and COCO-based (8.60) layouts

### 3. Relighting & Blending
- **Global Relighting**: Apply IC-Light diffusion model to:
  - Generate compatible backgrounds
  - Harmonize illumination across all objects
  - Suppress hard-edge artifacts from naive pasting

- **Mask-Area-Weighted Blending**: Re-blend original segments to address IC-Light failure modes:
  - **Problem 1**: IC-Light distorts fine details of small objects
  - **Problem 2**: IC-Light alters object colors (e.g., blue → red)
  - **Solution**: Blend with weight α ∈ [0,1] (higher for small objects to preserve details)

**Blending Comparison on LVIS-Mini:**
| Method | AP | Improvement |
|--------|----|----|
| Naive Paste | - | Hard edges, color mismatch |
| IC-Light Only | 36.3 | Global harmonization |
| **IC-Light + Blending** | **38.6** | **+2.3 AP** |

### 4. Ground Truth Generation
- **Segmentation Masks**: Compute final masks by subtracting occlusions from later-placed segments
- **Bounding Boxes**: Extract tight boxes directly from final masks
- **Referring Expressions**: Generate at least 9 expressions per image using QwQ-32B:
  - **Attribute-based**: "the red apple", "charcoal-grey cat"
  - **Spatial-based**: "dog to the right of the bike"
  - **Mixed-type**: "red object to the right of the child"
  - 3-6 expressions per type for dense annotation coverage

<p align="center">
     <img src="./assets/comparison.png" alt="Relighting and Blending Comparison" width="800">
</p>


# 📈 Results

## Task 1: Open-Vocabulary Object Detection

**Model**: MM-Grounding-DINO | **Benchmarks**: LVIS v1.0 Full Val, OdinW-35

<p align="center">
  <img src="./assets/ovd.png" alt="Open-Vocabulary Detection Results" width="800">
</p>

### Key Findings

#### 🎯 Small Amount of SOS Efficiently Brings Strong Gains
With only **50K** synthetic images, SOS delivers gains comparable to orders of magnitude more real data:

| Training Data | LVIS AP | AP<sub>rare</sub> | Gain vs Baseline |
|--------------|---------|-------------------|------------------|
| Object365+GoldG (Baseline) | 20.1 | 10.1 | - |
| + GRIT (20M images) | 27.1 | 17.1 | +7.0 AP |
| + V3Det (200K images) | 30.6 | 24.6 | +10.5 AP |
| **+ SOS-50K** | **29.8** | **23.5** | **+9.7 AP** |

**SOS-50K matches V3Det's gains with 400× fewer images!**

#### 📊 Scaling Up SOS Data Leads to Better Performance
Continuous improvements as we scale from 50K → 100K → 400K:

| SOS Scale | LVIS AP | AP<sub>rare</sub> | OdinW-35 mAP |
|-----------|---------|-------------------|--------------|
| 50K | 29.8 | 23.5 | 21.0 |
| 100K | 31.0 (+1.2) | 26.3 (+2.8) | 21.0 |
| 400K | **31.4 (+1.6)** | **27.9 (+1.6)** | **22.8 (+1.8)** |

#### 🔄 SOS is Complementary to Real Datasets
Adding SOS on top of large-scale real datasets yields additive gains:

| Training Data | LVIS AP | AP<sub>rare</sub> | OdinW-35 mAP |
|--------------|---------|-------------------|--------------|
| Object365+GoldG+V3Det+GRIT | 31.9 | 23.6 | - |
| **+ SOS-100K** | **33.2 (+1.3)** | **29.8 (+6.2)** | **+2.8** |

SOS introduces novel vocabulary and contextual variations not captured by existing real datasets.

## Task 2: Visual Grounding

**Model**: MM-Grounding-DINO | **Benchmarks**: RefCOCO/+/g, gRefCOCO, DoD

<p align="center">
  <img src="./assets/ref.png" alt="Visual Grounding Results" width="800">
</p>

### Key Findings

#### ⚠️ Existing Large Detection and Grounding Datasets Yield Only Marginal Improvements
Large-scale real datasets provide limited gains for referring expression tasks:

| Training Data | gRefCOCO P@1 | gRefCOCO N<sub>Acc</sub> | DoD FULL mAP |
|--------------|--------------|--------------------------|--------------|
| Object365+GoldG | - | 89.3 | - |
| + V3Det (200K) | +0.5 | +0.0 | - |
| + GRIT (20M) | - | - | +1.4 |

**Why?** V3Det lacks sentence-level supervision; GRIT uses noisy model-generated caption-box pairs.

#### ✨ SOS Provides Diverse, High-Quality Referring Expressions
SOS generates precise referring pairs from ground truth annotations without human labels:

| Training Data | gRefCOCO N<sub>Acc</sub> | DoD FULL mAP | Gain |
|--------------|--------------------------|--------------|------|
| Object365+GoldG | 89.3 | - | Baseline |
| **+ SOS-50K** | **93.9 (+4.6)** | **+1.0** | 50K images |
| **+ SOS-100K** | **97.7 (+8.4)** | **+3.8** | 100K images |

**Expression Types** (3-6 per type, balanced coverage):
- **Attribute-based**: "the red apple", "charcoal-grey cat"
- **Spatial-based**: "dog to the right of the bike"
- **Mixed-type**: "red object to the right of the child"

SOS's gains per example far outperform GRIT (20M) and V3Det (200K)!

## Task 3: Instance Segmentation

**Model**: APE (LVIS pre-trained) | **Benchmark**: LVIS v1.0 Val

<p align="center">
  <img src="./assets/results.png" alt="Instance Segmentation Results" width="800">
</p>

### Key Findings

#### 🎯 SOS Continuously Improves LVIS Segmentation
Two-stage fine-tuning: (1) Train on 50K SOS-LVIS → (2) Continue on LVIS train split

| Training Protocol | AP | AP<sub>rare</sub> | AP<sub>common</sub> | AP<sub>frequent</sub> |
|------------------|-------|-------------------|---------------------|----------------------|
| LVIS only | 46.96 | 40.87 | - | - |
| **SOS-50K → LVIS** | **48.48 (+1.52)** | **44.70 (+3.83)** | - | **(+0.31)** |

**Why the large rare-class gain?** Synthetic data can be generated to cover underrepresented classes, mitigating LVIS's long-tail imbalance. Frequent classes already have ample real examples and benefit less.

---

## Task 4: Small-Vocabulary, Limited-Data Regimes

**Model**: Mask2Former-ResNet-50 | **Benchmark**: COCO Instance Segmentation

### Key Findings

#### 💰 SOS Excels in Low-Data Regimes
Mixing real COCO segments with SOS synthetic segments (80 COCO categories):

| COCO Data Scale | COCO Only | COCO + SOS | Gain |
|----------------|-----------|------------|------|
| 1% (~1K images) | - | - | **+6.59 AP** |
| 10% (~10K images) | - | - | **~+3 AP** |
| 50% (~50K images) | - | - | **~+3 AP** |
| 100% (Full) | - | - | **~+3 AP** |

**Key Insight**: The boost is particularly dramatic at 1% COCO (+6.59 AP), and grows by roughly 3% at each subsequent data scale. SOS is most effective when real data is scarce!

---

## Task 5: Intra-Class Referring Expression

**Model**: MM-Grounding-DINO | **Benchmark**: Custom intra-class benchmark (COCO + OpenImages V7)

<p align="center">
  <img src="./assets/intra-class.png" alt="Intra-Class Referring Results" width="800">
</p>

### What is Intra-Class Referring?
A challenging visual grounding task requiring fine-grained attribute discrimination among same-category instances.

**Example**: In an image with multiple cars of different colors and makes, locate "the charcoal-grey sedan" (not just "car").

**Why it's hard**: Models often shortcut by ignoring attributes and relying solely on category nouns.

### Evaluation Metrics
- **Average Gap**: Average confidence margin between ground-truth box and highest-scoring same-category distractor
- **Positive Gap Ratio**: Percentage of images where ground-truth box receives highest confidence among same-category candidates

### Key Findings

#### 🎯 Targeted SOS Data Fixes Intra-Class Shortcuts

| Training Data | Average Gap | Positive Gap Ratio |
|--------------|-------------|-------------------|
| Object365+GoldG | 37.5 | ~80% |
| + GRIT (20M) | 34.6 (-2.9) | ~82% |
| + V3Det (200K) | 36.7 (-0.8) | ~83% |
| + GRIT + V3Det | 35.8 (-1.7) | ~85% |
| **+ SOS-SFC-50K + SOS-SGC-50K** | **40.6 (+3.1)** | **90%** |

**SOS-SFC/SGC**: Synthetic images with multiple instances of the same category but varied attributes (e.g., cars with different colors and makes).

**Key Insight**: Large-scale auxiliary data (GRIT, V3Det) yields negligible or even negative impact. Only targeted synthetic data tailored to intra-class attribute variation significantly improves performance!

---

## Comparison with Other Synthetic Pipelines

We compare SOS with established synthetic augmentation methods:

| Method | COCO AP (Mask2Former) | LVIS-Mini AP (MM-GDINO) | Relative Gain |
|--------|----------------------|------------------------|---------------|
| Simple Copy-Paste | 9.3 | 35.2 | Baseline |
| X-Paste (diffusion refinement) | 9.4 | 37.2 | +0.1 / +2.0 |
| **SOS (Ours)** | **12.79** | **38.6** | **+37.7% / +9.7%** |

### Why SOS Outperforms Copy-Paste Methods?

1. **🌍 Open-Vocabulary Coverage**: 20M segments spanning 40K+ categories vs. ~1.3K from LVIS/COCO
2. **🎨 Photorealistic Harmonization**: IC-Light relighting + mask-aware blending eliminates seam artifacts while preserving attributes
3. **🔄 Compositional Diversity**: Compose entire scenes from scratch → unlimited layout variations

---

## Ablation Study

**Setup**: Mask2Former on COCO instance segmentation, trained only on SOS, evaluated zero-shot

| Component | AP | Notes |
|-----------|----|----|
| Random Layout | 9.07 | Random center points, original sizes |
| COCO Layout | 8.60 | COCO bounding boxes constrain size/position |
| **Our Layout** | **9.16** | Size distribution + uniform sampling |
| **+ Relighting & Blending** | **12.79 (+39.7%)** | IC-Light + mask-area-weighted blending |
| | | |
| Real segments only | 7.03 | COCO segments |
| + Subject200K segments | 12.06 | Add synthetic segments |
| **+ SOS segments (Ours)** | **12.79 (+0.73)** | Higher quality synthetic segments |

### Key Takeaways
- ✅ **Layout matters**: Our statistical approach outperforms both random and COCO-constrained layouts
- ✅ **Relighting & blending are critical**: +39.7% AP uplift by eliminating edge artifacts
- ✅ **Segment quality impacts results**: SOS segments outperform other synthetic segment sources

<!-- *For figures, tables, and full details see the [paper PDF](./paper.pdf) and supplementary materials.*   -->

---

# 📧 Contact

* **Weikai Huang**: weikaih@cs.washington.edu
* **Jieyu Zhang**: jieyuz2@cs.washington.edu

---

<!--
# 📝 Citation

```bibtex
# Bibtex: stay tuned for ArXiv!
```
-->

---

# 🙏 Acknowledgments

We thank the authors of FLUX-1, IC-Light, DIS, Qwen, and QwQ for their excellent open-source models that made this work possible.
