import gradio as gr
import base64
from io import BytesIO
from PIL import Image
from src.pipeline.prediction_pipeline import PredictionPipeline
import spaces
import json

# Initialize the pipeline
pipeline = PredictionPipeline()

@spaces.GPU
def predict_gradio(img):
    if img is None:
        return None, None, None
        
    try:
        # Convert PIL Image to bytes
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        # Run the pipeline which expects bytes
        result = pipeline.predict(img_bytes)
        
        # Format predictions for the Label component (Gradio expects 0-1)
        confidences = {item["class"]: item["confidence"] / 100.0 for item in result["all_classes"]}
        
        # Extract Grad-CAM Image
        grad_cam_img = None
        if result["images"].get("grad_cam"):
            grad_cam_data = result["images"]["grad_cam"].split(",")[1]
            grad_cam_img = Image.open(BytesIO(base64.b64decode(grad_cam_data)))
            
        # Extract LIME Image
        lime_img = None
        if result["images"].get("lime"):
            lime_data = result["images"]["lime"].split(",")[1]
            lime_img = Image.open(BytesIO(base64.b64decode(lime_data)))
            
        return confidences, grad_cam_img, lime_img
            
    except Exception as e:
        return {f"Error: {str(e)}": 1.0}, None, None

# Define a custom theme to make it look professional
custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_radius="16px",
    block_shadow="*shadow_drop_lg"
)

# Define the Gradio Interface
with gr.Blocks(theme=custom_theme, title="Sugarcane Disease AI") as demo:
    # Header Section
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px 0;">
                <h1 style="font-weight: 800; font-size: 2.5rem; color: #059669; margin-bottom: 0.5rem;">🌿 Sugarcane Leaf Disease AI</h1>
                <p style="font-size: 1.1rem; color: #475569;">Upload an image of a sugarcane leaf to instantly detect up to 15 different health conditions using Deep Learning. The AI also provides Explainable AI (XAI) visuals like Grad-CAM++ and LIME to show exactly why it made its decision.</p>
            </div>
            """
        )
    
    # Main Content Section
    with gr.Row():
        
        # Left Column: Input and Controls
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Image")
            img_input = gr.Image(type="pil", label="Sugarcane Leaf", height=400)
            
            with gr.Row():
                clear_btn = gr.ClearButton(value="🔄 Reset", components=[img_input], variant="secondary")
                submit_btn = gr.Button("🔍 Analyze Image", variant="primary")
                
        # Right Column: Results
        with gr.Column(scale=2):
            gr.Markdown("### 2. Analysis Results")
            
            with gr.Tabs():
                # Tab 1: AI Prediction
                with gr.TabItem("📊 AI Prediction"):
                    # Showing all 15 classes as requested
                    label_output = gr.Label(label="Detection Confidence (All 15 Classes)", num_top_classes=15)
                
                # Tab 2: Explainable AI
                with gr.TabItem("🧠 Explainable AI (XAI)"):
                    gr.Markdown("*These images show which parts of the leaf the AI looked at to make its decision.*")
                    with gr.Row():
                        cam_output = gr.Image(label="Grad-CAM++ (Heatmap)", interactive=False)
                        lime_output = gr.Image(label="LIME (Boundary Analysis)", interactive=False)
    
    # Wire the Clear Button to clear outputs as well
    clear_btn.add([label_output, cam_output, lime_output])
    
    # Wire the Analyze Button
    submit_btn.click(
        fn=predict_gradio,
        inputs=img_input,
        outputs=[label_output, cam_output, lime_output]
    )

if __name__ == "__main__":
    demo.launch()
