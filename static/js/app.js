document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const cameraInput = document.getElementById('cameraInput');
    
    const actionButtons = document.getElementById('actionButtons');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('resultContainer');
    
    const predictionResult = document.getElementById('predictionResult');
    const confidenceBars = document.getElementById('confidenceBars');
    const origImg = document.getElementById('origImg');
    const gradCamImg = document.getElementById('gradCamImg');
    const limeImg = document.getElementById('limeImg');

    const errorContainer = document.getElementById('errorContainer');
    const errorText = document.getElementById('errorText');
    const resetErrorBtn = document.getElementById('resetErrorBtn');

    let selectedFile = null;

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            showError("Please upload a valid image file (JPG, PNG).");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            
            // Hide upload buttons, show preview
            actionButtons.classList.add('hidden');
            previewArea.classList.remove('hidden');
            
            // Enable Predict and Reset buttons
            analyzeBtn.disabled = false;
            resetBtn.disabled = false;
            
            // Ensure results are hidden if selecting a new image after an error
            resultContainer.classList.add('hidden');
            errorContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
    });

    cameraInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Disable Predict button during processing, but keep Reset enabled
        analyzeBtn.disabled = true;
        loader.classList.remove('hidden');
        errorContainer.classList.add('hidden');
        resultContainer.classList.add('hidden'); // hide previous results if any

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const jsonResp = await response.json();
            
            if (!response.ok || jsonResp.status === 'error') {
                throw new Error(jsonResp.error || 'Failed to analyze image');
            }

            const data = jsonResp.data;

            loader.classList.add('hidden');
            resultContainer.classList.remove('hidden');
            
            // 1. Final Diagnosis
            predictionResult.textContent = data.predicted_class;
            
            // 2. All 15 Confidence Bars (or fallback if server not restarted)
            confidenceBars.innerHTML = '';
            const classesList = data.all_classes || data.top_classes || [];
            classesList.forEach((item, index) => {
                const confHtml = `
                <div class="conf-item">
                    <div class="conf-label">
                        <span class="conf-name">${item.class}</span>
                        <span>${item.confidence.toFixed(2)}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width: 0%" data-width="${item.confidence}%"></div>
                    </div>
                </div>`;
                confidenceBars.insertAdjacentHTML('beforeend', confHtml);
            });

            setTimeout(() => {
                document.querySelectorAll('.conf-bar-fill').forEach(bar => {
                    bar.style.width = bar.getAttribute('data-width');
                });
            }, 100);

            // 3. Images
            origImg.src = data.images.original;
            gradCamImg.src = data.images.grad_cam;
            
            if (data.images.lime) {
                document.getElementById('limeContainer').classList.remove('hidden');
                limeImg.src = data.images.lime;
            } else {
                document.getElementById('limeContainer').classList.add('hidden');
            }

        } catch (error) {
            loader.classList.add('hidden');
            errorContainer.classList.remove('hidden');
            analyzeBtn.disabled = false; // re-enable so user can try again
            errorText.textContent = error.message;
        }
    });

    function resetUI() {
        selectedFile = null;
        fileInput.value = '';
        cameraInput.value = '';
        imagePreview.src = '';
        
        previewArea.classList.add('hidden');
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
        loader.classList.add('hidden');
        
        actionButtons.classList.remove('hidden');
        
        analyzeBtn.disabled = true;
        resetBtn.disabled = true;
    }

    resetBtn.addEventListener('click', resetUI);
    resetErrorBtn.addEventListener('click', resetUI);
    
    function showError(msg) {
        errorContainer.classList.remove('hidden');
        errorText.textContent = msg;
    }

    let deferredPrompt;
    const installAppBtn = document.getElementById('installAppBtn');

    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        installAppBtn.classList.remove('hidden');
    });

    installAppBtn.addEventListener('click', async () => {
        if (deferredPrompt) {
            deferredPrompt.prompt();
            const { outcome } = await deferredPrompt.userChoice;
            if (outcome === 'accepted') {
                installAppBtn.classList.add('hidden');
            }
            deferredPrompt = null;
        }
    });
    
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', () => {
            navigator.serviceWorker.register('/static/service-worker.js');
        });
    }
});
