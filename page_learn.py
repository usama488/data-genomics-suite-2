import pandas as pd
import streamlit as st

import cxr_model as M

TOPICS = {
    "How the model sees an X-ray": "The image is converted to grey, centre-cropped to a square, resized to 224×224 pixels and scaled to "
                                   "[-1024, 1024]. A DenseNet-121 convolutional neural network (121 layers, ~7 M parameters) turns it into "
                                   "1,024 feature maps of 7×7, averages them, and a final linear layer outputs one score per finding.",
    "Convolutional neural network (CNN)": "Stacks of small filters slide over the image. Early layers detect edges, middle layers textures "
                                          "and shapes (ribs, vessels), deep layers whole patterns (an effusion's meniscus, a heart silhouette).",
    "DenseNet": "Each layer receives the feature maps of all previous layers, which re-uses features and trains well on limited medical data.",
    "Multi-label output": "A radiograph can show several findings at once, so each of the 18 outputs is an independent sigmoid "
                          "(not a softmax that forces one answer).",
    "Operating threshold & calibrated score": "Each finding has its own threshold chosen on validation data. The app rescales the "
                                              "probability so that threshold sits at 0.5 for every finding: 0.5 = borderline, 0.7+ = confident.",
    "Sensitivity vs specificity": "Sensitivity = % of diseased patients flagged; specificity = % of healthy patients not flagged. "
                                  "Raising the cut-off increases specificity but misses more disease - the triage slider shows this trade-off.",
    "AUROC": "Area under the ROC curve: probability the model scores a random positive case higher than a random negative one "
             "(0.5 = chance, 1.0 = perfect). Published AUROCs for these models vary by finding and dataset.",
    "Class activation map (CAM)": "Multiplies the final-layer weights for one finding with the 7×7 feature maps → a coarse map of the "
                                  "regions that pushed the score up.",
    "Occlusion sensitivity": "Hide one patch at a time and re-run the model; regions whose removal lowers the score the most matter most.",
    "Shortcut learning": "Models can learn hospital-specific cues (chest tubes for pneumothorax, 'portable' text markers for sick patients) "
                         "instead of disease. Heat-maps help detect this.",
    "Dataset shift": "Performance drops on images unlike the training data: children, lateral views, different scanners, "
                     "or populations under-represented in the US/European training sets.",
    "Triage vs diagnosis": "Worklist triage AI re-orders studies so urgent ones are read first; the radiologist still reads every image.",
    "Regulation": "Clinical AI must be cleared as a medical device (FDA 510(k), CE-MDR) and locally validated before use on patients.",
    "Transfer learning": "The brain-MRI and skin models start from MobileNetV3 pretrained on 1.2 million everyday photos (ImageNet) "
                         "and are fine-tuned on medical images - far less data is needed than training from scratch.",
    "Softmax vs sigmoid": "Brain and skin models choose ONE class (softmax, probabilities sum to 1); the chest X-ray model scores "
                          "18 findings independently (sigmoid) because several can co-exist.",
    "Class imbalance": "HAM10000 is 67% benign moles. Without class weighting a model can reach 67% accuracy by always saying 'mole'. "
                       "That's why macro-F1 and per-class sensitivity (especially melanoma) matter more than accuracy.",
    "Validation vs test": "Validation data come from the same source as training; a separate test set shows how the model generalises. "
                          "The brain model scores 97% on validation but ~75% on the dataset's official test folder - dataset shift.",
    "Melanoma alert threshold": "Missing a melanoma is worse than an extra biopsy, so the app lets you flag melanoma at a probability "
                                "well below 50% to raise sensitivity.",
    "Cardiothoracic ratio": "Widest heart diameter ÷ widest inner thoracic diameter on a PA film; > 0.5 suggests cardiomegaly. "
                            "AP films magnify the heart.",
}


def render():
    st.title("📚 Learn: AI for medical imaging")
    st.caption("Model cards with accuracy for the brain-MRI and skin models are on their own pages (tab 📊 Model card & accuracy).")
    q = st.text_input("Search", placeholder="e.g. CAM, threshold, AUROC")
    for k, v in TOPICS.items():
        if not q or q.lower() in (k + v).lower():
            st.markdown(f"**{k}** — {v}")
    st.subheader("Chest X-ray model card")
    st.markdown("- **Model:** DenseNet-121, `densenet121-res224-all` from TorchXRayVision (Cohen et al., *MIDL* 2022), exported to ONNX "
                "(outputs verified identical to the original PyTorch model).\n"
                "- **Training data:** NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest, Google NIH labels, OpenI, RSNA Pneumonia (Kaggle).\n"
                "- **Input:** frontal (PA/AP) adult chest radiograph. **Not** for lateral views, children, CT or other body parts.\n"
                "- **Output:** 18 findings with per-finding operating thresholds.\n"
                "- **Intended use here:** education and research demonstration.")
    st.dataframe(pd.DataFrame({"finding": M.PATHOLOGIES, "operating threshold (raw prob.)": M.OP.round(4),
                               "triage group": [M.PRIORITY[f] for f in M.PATHOLOGIES],
                               "meaning": [M.DESCRIPTION[f] for f in M.PATHOLOGIES]}), hide_index=True)
