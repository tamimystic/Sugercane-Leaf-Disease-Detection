from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import gradio as gr
import uvicorn
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Sugarcane Leaf Disease Classifier")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# LIME configs
ENABLE_LIME = True
LIME_SAMPLES = 50 # Reduced for CPU speed on HF Spaces

pipeline = PredictionPipeline(enable_lime=ENABLE_LIME, lime_samples=LIME_SAMPLES)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = pipeline.predict(contents)
        return JSONResponse(content={"data": result, "status": "success"})
    except Exception as e:
        return JSONResponse(content={"error": str(e), "status": "error"}, status_code=500)

# Dummy Gradio app to satisfy Hugging Face Gradio SDK requirements
demo = gr.Blocks()
with demo:
    gr.Markdown("Backend Server is running.")

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
