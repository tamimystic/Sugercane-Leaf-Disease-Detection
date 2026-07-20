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

# Define the Gradio Interface
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown(
        """
        # 🌿 Sugarcane Leaf Disease Detection
        Upload an image of a sugarcane leaf to instantly detect up to 15 different health conditions. 
        The AI also provides Explainable AI (XAI) visuals like Grad-CAM++ and LIME to show exactly why it made its decision.
        """
    )
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Sugarcane Leaf Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column():
            label_output = gr.Label(label="Prediction Confidence", num_top_classes=5)
            
    with gr.Row():
        cam_output = gr.Image(label="Grad-CAM++ (Focus Area)")
        lime_output = gr.Image(label="LIME (Boundary Analysis)")
        
    # Wire the button to the prediction function
    submit_btn.click(
        fn=predict_gradio,
        inputs=img_input,
        outputs=[label_output, cam_output, lime_output]
    )

if __name__ == "__main__":
    demo.launch()
