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
        return None, None, None
        
    try:
        # Convert PIL Image to bytes
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

        # Run the pipeline which expects bytes
        result = pipeline.predict(img_bytes)
        
        # Format predictions for the Dataframe (exact float percentage)
        confidences = [[item["class"], f"{item['confidence']:.4f}%"] for item in result["all_classes"]]
        
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

# Custom CSS for animations and a more professional look
css = """
/* Fade in animation for the whole app */
.gradio-container {
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Button hover effects */
button.primary {
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
}

/* Image expansion and hover effects */
.image-container img {
    transition: transform 0.3s ease !important;
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}
.image-container:hover img {
    transform: scale(1.01) !important;
}

/* Footer styling */
.footer-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
    width: 100%;
}
.footer {
    color: #64748b;
    font-size: 0.95rem;
}
.footer a {
    color: #059669;
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover {
    text-decoration: underline;
}
"""

# Define the Gradio Interface
with gr.Blocks(theme=custom_theme, title="Sugarcane Disease AI", css=css) as demo:
    # Header Section
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px 0;">
                <h1 style="font-weight: 800; font-size: 2.5rem; color: #059669; margin-bottom: 0.5rem;">Sugarcane Leaf Disease Detection</h1>
                <p style="font-size: 1.1rem; color: #475569;">Upload an image of a sugarcane leaf to instantly detect up to 15 different health conditions using Deep Learning. The system also provides Explainable AI (XAI) visuals like Grad-CAM++ and LIME to show exactly why it made its decision.</p>
            </div>
            """
        )
    
    # Main Content Section
    with gr.Row():
        
        # Left Column: Input and Controls
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Image")
            img_input = gr.Image(type="pil", label="Sugarcane Leaf", sources=["upload", "webcam"], height=400, elem_classes="image-container")
            
            with gr.Row():
                clear_btn = gr.ClearButton(value="Reset", components=[img_input], variant="secondary")
                submit_btn = gr.Button("Analyze Image", variant="primary")
                
        # Right Column: Results
        with gr.Column(scale=2):
            gr.Markdown("### 2. Analysis Results")
            
            with gr.Tabs():
                # Tab 1: AI Prediction
                with gr.TabItem("Prediction"):
                    label_output = gr.Dataframe(headers=["Condition", "Confidence"], interactive=False)
                
                # Tab 2: Explainable AI
                with gr.TabItem("Explainable AI (XAI)"):
                    gr.Markdown("*These images show which parts of the leaf the model focused on to make its decision.*")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Grad-CAM++ (Heatmap)")
                            cam_output = gr.Image(show_label=False, interactive=False, elem_classes="image-container")
                        with gr.Column():
                            gr.Markdown("#### LIME (Boundary Analysis)")
                            lime_output = gr.Image(show_label=False, interactive=False, elem_classes="image-container")
    
    # Footer Section
    with gr.Row(elem_classes="footer-container"):
        gr.Markdown(
            """
            <div class="footer">
                Powered by <a href="https://www.linkedin.com/in/tamimystic" target="_blank">tamimystic</a>
            </div>
            """
        )
    
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
