import gradio as gr
from transformers import StableDiffusionPipeline
import numpy as np
import cv2
from PIL import Image
import os

# Initialize Stable Diffusion model
model_name = "CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(model_name)

def generate_image_from_text(prompt):
    image = pipeline(prompt).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path

def create_video_from_image(image_path):
    # Define video parameters
    frame_width = 640
    frame_height = 480
    fps = 1  # 1 frame per second

    # Create a temporary directory to store the video
    temp_dir = "frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Resize and add image to video
    frame = cv2.imread(image_path)
    if frame is not None:
        frame = cv2.resize(frame, (frame_width, frame_height))
        out = cv2.VideoWriter('generated_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
        out.write(frame)
        out.release()

    # Clean up temporary files
    os.rmdir(temp_dir)

    return 'generated_video.avi'

def generate_image_and_video(prompt):
    # Generate image from prompt
    image_path = generate_image_from_text(prompt)
    
    # Create video from image
    video_path = create_video_from_image(image_path)
    
    return image_path, video_path

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Mesozoic Era Video Generator")
    prompt_input = gr.Textbox(label="Enter a prompt", placeholder="Describe what you want to generate")
    image_output = gr.Image(label="Generated Image", type="filepath")
    video_output = gr.Video(label="Generated Video", type="filepath")
    
    generate_button = gr.Button("Generate")
    generate_button.click(fn=generate_image_and_video, inputs=prompt_input, outputs=[image_output, video_output])

demo.launch()
