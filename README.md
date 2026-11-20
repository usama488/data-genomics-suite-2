# 🩻 Medical Imaging AI (Streamlit)

Three assistants in one app:

| Module | Model | Data | Held-out result |
|---|---|---|---|
| 🩻 Chest X-ray | DenseNet-121 (TorchXRayVision, pretrained) | 8 public CXR datasets | published model, 18 findings |
| 🧠 Brain MRI tumour | MobileNetV3-Large fine-tuned here | Bhuvaji et al. 2020 (3,264 slices, 4 classes) | val 97.2%; official test folder 75.4% (dataset shift, explained in app) |
| 🔬 Skin lesion | MobileNetV3-Large fine-tuned here | HAM10000 (10,015 dermoscopy images, 7 classes, CC BY-NC 4.0) | test accuracy 82.1%, melanoma AUC 0.86, macro-F1 0.70 |

Brain and skin pages: probabilities, CAM + occlusion heat-maps, suggested action (urgency), a melanoma alert threshold, clinician review, PDF report, and a model card with the confusion matrix.

## Chest X-ray module

- **Input:** PNG / JPG / DICOM (VOI LUT, MONOCHROME1 handling, header with identifiers masked)
- **Image QC:** resolution, exposure/contrast, clipping, inverted-image detection, view (PA/AP/lateral), symmetry
- **Viewer:** window/level, gamma, CLAHE, histogram equalisation, edge enhancement, invert
- **AI:** DenseNet-121 from TorchXRayVision (Cohen et al., MIDL 2022), trained on 8 public datasets. It reports 18 findings with calibrated scores (0.5 = operating threshold). It is exported to ONNX and gives outputs identical to the original PyTorch model.
- **Explainability:** class activation maps (instant) and occlusion sensitivity (~10 s)
- **Triage:** STAT / Urgent / Routine / Normal, an adjustable cut-off, and a batch worklist with a heat-map
- **CTR tool:** manual cardiothoracic ratio, in cm when DICOM pixel spacing is available
- **Review & report:** the radiologist confirms or rejects each AI finding, then downloads a radiology PDF with images and the heat-map

The model weights are split into `cxr_model_part1.bin` + `cxr_model_part2.bin` (~14 MB each) so they can be uploaded to GitHub through the browser.

**Run:** `pip install -r requirements.txt && streamlit run app.py`. Deploy on Streamlit Cloud with main file `app.py` and Python 3.12. It works on Windows too, because PyTorch is not needed.

⚠️ This is for research and education. It is not a cleared medical device.
