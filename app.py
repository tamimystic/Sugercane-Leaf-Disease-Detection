import gradio as gr
import base64
from io import BytesIO
from PIL import Image
from src.pipeline.prediction_pipeline import PredictionPipeline
import spaces
import json

# Initialize the pipeline
pipeline = PredictionPipeline()

@spaces.GPU(duration=15)
def predict_gradio(img):
    if img is None:
        return None, None, None, None
        
    try:
        # Convert PIL Image to bytes
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        # Run the pipeline which expects bytes
        result = pipeline.predict(img_bytes)
        
        # 1. Format Top Prediction HTML
        top_class = result["all_classes"][0]
        top_pred_html = f"""
        <div class="top-prediction-card">
            <h3 class="top-pred-label">Primary Diagnosis</h3>
            <h1 class="top-pred-name">{top_class['class']}</h1>
            <div class="top-pred-conf">{top_class['confidence']:.2f}% Confidence</div>
        </div>
        """
        
        # 2. Format remaining predictions Dataframe
        other_confidences = [[item["class"], f"{item['confidence']:.4f}%"] for item in result["all_classes"][1:]]
        
        # 3. Extract Grad-CAM Image
        grad_cam_img = None
        if result["images"].get("grad_cam"):
            grad_cam_data = result["images"]["grad_cam"].split(",")[1]
            grad_cam_img = Image.open(BytesIO(base64.b64decode(grad_cam_data)))
            
        # 4. Extract LIME Image
        lime_img = None
        if result["images"].get("lime"):
            lime_data = result["images"]["lime"].split(",")[1]
            lime_img = Image.open(BytesIO(base64.b64decode(lime_data)))
            
        return top_pred_html, other_confidences, grad_cam_img, lime_img
            
    except Exception as e:
        error_html = f"<div style='color:red; font-size:1.2rem;'>Error: {str(e)}</div>"
        return error_html, None, None, None

# Base theme to remove default Gradio styles
custom_theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    body_background_fill="transparent",
    block_background_fill="rgba(255,255,255,0.03)",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.1)",
    block_radius="16px",
    block_shadow="0 8px 15px rgba(0,0,0,0.2)"
)

# Deep Custom CSS for extreme premium animated UI
css = """
/* Core Dark Theme Background */
body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
}

/* Base Animations */
.gradio-container {
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Header Typography */
.header-title {
    font-weight: 900 !important;
    font-size: 3rem !important;
    background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    line-height: 1.2;
}
.header-desc {
    font-size: 1.2rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* Stunning Top Prediction Card */
.top-prediction-card {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 15px 35px rgba(16, 185, 129, 0.3);
    margin-bottom: 2rem;
    animation: popIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    border: 1px solid rgba(255,255,255,0.2);
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}
.top-pred-label {
    font-size: 1.1rem;
    color: #d1fae5;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
    margin-top: 0;
}
.top-pred-name {
    font-size: 3rem;
    font-weight: 900;
    color: white;
    margin: 10px 0;
    text-shadow: 0 4px 6px rgba(0,0,0,0.2);
    line-height: 1.1;
}
.top-pred-conf {
    font-size: 1.3rem;
    font-weight: 700;
    background: rgba(0,0,0,0.2);
    padding: 8px 25px;
    border-radius: 30px;
    display: inline-block;
    color: #f8fafc;
    border: 1px solid rgba(255,255,255,0.15);
}

/* Make internal image component buttons larger and clearer */
.image-container button[aria-label="Upload"]::after,
.image-container button[aria-label="upload"]::after {
    content: " Upload Image";
    font-size: 1.1rem;
    font-weight: 600;
    margin-left: 8px;
}
.image-container button[aria-label="Webcam"]::after,
.image-container button[aria-label="webcam"]::after,
.image-container button[aria-label="Camera"]::after,
.image-container button[aria-label="camera"]::after {
    content: " Open Camera";
    font-size: 1.1rem;
    font-weight: 600;
    margin-left: 8px;
}
.image-container button[aria-label="Upload"],
.image-container button[aria-label="upload"],
.image-container button[aria-label="Webcam"],
.image-container button[aria-label="webcam"],
.image-container button[aria-label="Camera"],
.image-container button[aria-label="camera"] {
    background: rgba(255,255,255,0.05) !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    margin: 0 10px !important;
    transition: all 0.3s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.image-container button[aria-label="Upload"]:hover,
.image-container button[aria-label="upload"]:hover,
.image-container button[aria-label="Webcam"]:hover,
.image-container button[aria-label="webcam"]:hover,
.image-container button[aria-label="Camera"]:hover,
.image-container button[aria-label="camera"]:hover {
    background: rgba(255,255,255,0.15) !important;
    transform: translateY(-2px);
}

/* Image Containers & XAI */
.xai-image {
    border-radius: 20px !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.02) !important;
}

/* Premium Buttons */
button.primary {
    background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    color: white !important;
    font-size: 1.3rem !important;
    font-weight: bold !important;
    padding: 18px !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 25px rgba(59, 130, 246, 0.6) !important;
}
button.secondary {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #f8fafc !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}
button.secondary:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    transform: translateY(-2px) !important;
}

/* Dark Mode Dataframe & Accordion */
table {
    border-collapse: collapse !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: transparent !important;
}
th {
    background: rgba(255,255,255,0.1) !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
    border: none !important;
}
td {
    background: rgba(255,255,255,0.03) !important;
    color: #e2e8f0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    font-weight: 500 !important;
}
.accordion {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
}
.accordion > button {
    color: #3b82f6 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

/* Footer styling */
.footer-container {
    width: 100%;
}
"""

with gr.Blocks(theme=custom_theme, title="Sugarcane Disease AI", css=css) as demo:
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; max-width: 900px; margin: 0 auto; padding: 40px 0;">
                <h1 class="header-title">Sugarcane Leaf Disease Detection</h1>
                <p class="header-desc">Upload an image of a sugarcane leaf to instantly detect health conditions using Deep Learning. The system also provides Explainable AI (XAI) visuals like Grad-CAM++ and LIME to show exactly why it made its decision.</p>
            </div>
            """
        )
    
    with gr.Row(equal_height=True):
        # Left Panel: Input
        with gr.Column(scale=4):
            img_input = gr.Image(type="pil", label="", show_label=False, sources=["upload", "webcam"], height=550, elem_classes="xai-image image-container")
            
            with gr.Row():
                clear_btn = gr.ClearButton(value="Reset", components=[img_input], variant="secondary")
                submit_btn = gr.Button("Analyze Image", variant="primary")
                
        # Right Panel: Output
        with gr.Column(scale=6):
            top_pred_output = gr.HTML()
            
            # Show all other possibilities directly below without accordion
            other_preds = gr.Dataframe(headers=["Condition", "Confidence"], interactive=False)

    # Bottom Row: XAI Images
    with gr.Row():
        gr.Markdown("<h2 style='text-align:center; width:100%; color:#10b981; margin-top: 20px; font-weight:800;'>Visual Explanation: Why did the model make this prediction?</h2>")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h3 style='text-align:center; color:#94a3b8; font-weight:700;'>Grad-CAM++ (Heatmap)</h3>")
            cam_output = gr.Image(show_label=False, interactive=False, elem_classes="xai-image")
        with gr.Column():
            gr.Markdown("<h3 style='text-align:center; color:#94a3b8; font-weight:700;'>LIME (Boundary Analysis)</h3>")
            lime_output = gr.Image(show_label=False, interactive=False, elem_classes="xai-image")

    # Footer
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: center; width: 100%; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1); color: #94a3b8; font-size: 1rem; letter-spacing: 0.5px;">
                Powered by <a href="https://www.linkedin.com/in/tamimystic" target="_blank" style="color: #10b981; font-weight: 700; text-decoration: none;">tamimystic</a>
            </div>
            """
        )
    
    # Wire the Clear Button to clear outputs as well
    clear_btn.add([top_pred_output, cam_output, lime_output, other_preds])
    
    # Wire the Analyze Button
    submit_btn.click(
        fn=predict_gradio,
        inputs=img_input,
        outputs=[top_pred_output, other_preds, cam_output, lime_output]
    )

if __name__ == "__main__":
    demo.launch()
