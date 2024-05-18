from flask import Flask, request, render_template, send_file
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'original_image' not in request.files or 'watermark_image' not in request.files:
        return 'No file part'
    
    original_image = Image.open(request.files['original_image'])
    watermark_image = Image.open(request.files['watermark_image'])
    
    watermarked_image = embed_watermark(original_image, watermark_image)
    
    # Save watermarked image to a bytes buffer
    buffer = io.BytesIO()
    watermarked_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='watermarked_image.png')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No file part'
    
    image = Image.open(request.files['image'])
    watermark = extract_watermark(image)
    
    # Save extracted watermark to a bytes buffer
    buffer = io.BytesIO()
    watermark.save(buffer, format="PNG")
    buffer.seek(0)
    
    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='extracted_watermark.png')

def embed_watermark(original_image, watermark_image):
    original = np.array(original_image)
    watermark = np.array(watermark_image.resize(original_image.size))
    
    # Ensure the watermark is not visible
    watermark = watermark // 32 * 32  # Reduce watermark intensity

    watermarked = original // 32 * 32 + watermark // 32
    return Image.fromarray(watermarked.astype(np.uint8))

def extract_watermark(image):
    array = np.array(image)
    watermark = (array % 32) * 32  # Extract least significant bits
    return Image.fromarray(watermark.astype(np.uint8))

if __name__ == "__main__":
    app.run(debug=True)
