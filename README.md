---
title: Sugarcane Leaf Disease Detection
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# Sugarcane Leaf Disease Detection

**[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/tamimystic/sugercane-leaf-disease-ditection)**

This project is a deep learning-based diagnostic application for identifying sugarcane leaf diseases. Built with PyTorch and FastAPI, it classifies 15 different leaf conditions and provides transparent results using Explainable AI (XAI) techniques.

The application serves two primary audiences: users who need a diagnostic tool, and developers who want to understand or extend the underlying architecture.

---

## For Users

The application provides a simple web interface where you can upload or capture images of sugarcane leaves to get an instant diagnosis. 

### Features
- **Disease Classification:** Detects up to 15 different health conditions in sugarcane leaves.
- **Model Confidence:** Displays the probability scores for all 15 classes, not just the top prediction.
- **Visual Explanations (XAI):** 
  - **Grad-CAM++:** Generates a heatmap over the original image, showing which areas the model focused on.
  - **LIME Boundaries:** Highlights the specific edges and patterns that contributed to the model's decision.
- **Progressive Web App (PWA):** The interface can be installed as a standalone app on mobile devices.

### How to Run Locally

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Start the local server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

3. Open your browser and navigate to `http://localhost:8000`.

*Note: Generating LIME boundaries is computationally intensive. If you want faster predictions during testing, you can disable it in the configuration (see the developer section below).*

---

## For Developers

The project follows a modular architecture designed for scalability and production deployment. 

### Project Structure
- `app.py`: The FastAPI application entry point. Controls global configurations.
- `src/components/`: Contains modules for data preprocessing (resizing, tensor normalization) and model loading.
- `src/pipeline/`: Houses `prediction_pipeline.py`, which handles inference and the execution of XAI algorithms.
- `src/config/` & `src/entity/`: Manages configuration constants and model paths.
- `src/logger/` & `src/exception/`: Standardized modules for application logging and error handling.

### Tech Stack
- **Machine Learning:** PyTorch, DenseNet-121 (fine-tuned)
- **Explainability:** `grad-cam`, `lime`, `scikit-image`
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Vanilla JavaScript, HTML, CSS (Glassmorphism design)

### Explainable AI Details and Optimization
Implementing LIME in a web environment presents performance challenges due to the large number of forward passes required for image perturbation. To prevent CUDA OOM errors and reduce latency:
- The `PredictionPipeline` uses a custom prediction function that batches images (batch size of 16) during LIME evaluation.
- The `app.py` file exposes two configuration variables:
  - `ENABLE_LIME`: A boolean to quickly toggle LIME execution on or off.
  - `LIME_SAMPLES`: Controls the number of perturbations. It is set to 150 by default to balance execution speed and boundary accuracy.

### Extending the Project
To swap out the underlying model or adapt this for a different dataset:
1. Place your new weights in the `models/` directory.
2. Update the class list and paths in `src/entity/config_entity.py`.
3. If using a model architecture other than DenseNet, update the `target_layers` parameter for GradCAM initialization in `prediction_pipeline.py`.

---

**🔗 Try it out:** [https://huggingface.co/spaces/tamimystic/sugercane-leaf-disease-ditection](https://huggingface.co/spaces/tamimystic/sugercane-leaf-disease-ditection)

Built by [tamimystic](https://www.linkedin.com/in/tamimystic/).

