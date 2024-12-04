# Emotion-to-Image Generation

This project integrates emotion detection and image generation to create a system that analyzes user input, identifies emotions, and generates a corresponding image based on the detected emotion. The project utilizes Hugging Face models for emotion detection and Stable Diffusion for generating images.

## Features

- **Emotion Detection**: Uses a pre-trained emotion detection model to identify emotions in text input.
- **Image Generation**: Generates images based on the detected emotion using the Stable Diffusion model.
- **Flask API**: A backend Flask API that processes user input and delivers generated images in response.

## Tech Stack

- **Flask**: Web framework for building the API.
- **Hugging Face Transformers**: Used for emotion detection and conversational AI models.
- **Diffusers**: For generating images with the Stable Diffusion model.
- **Torch**: Framework for machine learning (PyTorch).
- **Pillow**: For handling image processing and saving.
- **Python 3.x**: Programming language used.

## Requirements

Ensure you have Python 3.6+ installed, then install the necessary dependencies using the following:

```bash
git clone https://github.com/mrayushmehrotra/emotion-to-image-gen.git
cd emotion-to-image-gen
python3 -m venv . 
source bin/activate 
pip install flask transformers diffusers torch pillow
python3 app.py
```



## Example Request in curl
```bash
curl -X POST http://127.0.0.1:5000/generate_emotion_image \
-H "Content-Type: application/json" \
-d '{"text": "I feel so happy and excited today!"}'
```

## Example Response 
```bash 
{
  "input_text": "I feel so happy and excited today!",
  "detected_emotion": "joy",
  "image": "data:image/png;base64,<base64-encoded-image>"
}
```