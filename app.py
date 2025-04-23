from flask import Flask, render_template, request, redirect, url_for
import os
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline (this will download the model on first run)
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    print("Stable Diffusion model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

def generate_linkedin_description(prompt):
    """Generate LinkedIn description based on user prompt"""
    description = f"""
🚀 Passionate professional with expertise in {prompt}. 
💼 Currently looking for new opportunities to leverage my skills.
📈 Dedicated to continuous learning and professional growth.
✨ Let's connect and explore how we can collaborate!

#Professional #Networking #CareerGrowth #Opportunities #LinkedIn
"""
    return description.strip()

def generate_image(prompt):
    """Generate image using local Stable Diffusion model"""
    if pipe is None:
        print("Image generation not available - model not loaded")
        return None
    
    try:
        # Generate image
        image = pipe(
            prompt=f"professional LinkedIn banner image about {prompt}",
            width=1024,
            height=512,
            num_inference_steps=30
        ).images[0]
        
        # Save the generated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        return filename
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    description = None
    image_url = None
    
    if request.method == 'POST':
        prompt = request.form['prompt']
        description = generate_linkedin_description(prompt)
        image_filename = generate_image(prompt)
        if image_filename:
            image_url = url_for('static', filename=f'images/{image_filename}')
    
    return render_template('index.html', description=description, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)