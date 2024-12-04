import base64
import io

import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load models
conversation_model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(conversation_model_name)
conversation_model = AutoModelForCausalLM.from_pretrained(conversation_model_name)

emotion_detector = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
)

# Load Stable Diffusion model
sd_model_id = "CompVis/stable-diffusion-v1-4"
sd_pipe = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token="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",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipe.to(device)


# Detect emotion from text
def detect_emotion(input_text):
    emotions = emotion_detector(input_text)
    highest_emotion = max(emotions[0], key=lambda x: x["score"])
    return highest_emotion["label"]


# Generate an image based on the emotion
def generate_image(emotion):
    prompts = {
        "joy": "a vibrant and colorful scene filled with happiness and laughter",
        "sadness": "a serene, peaceful landscape with soft colors",
        "anger": "a dramatic storm with intense lighting and dark clouds",
        "fear": "a mysterious forest at night with a hint of eeriness",
        "surprise": "a spectacular fireworks display in a clear night sky",
        "love": "a warm sunset over a romantic beach setting",
    }
    prompt = prompts.get(emotion, "a generic serene landscape")
    image = sd_pipe(prompt, guidance_scale=8.5)["sample"][0]
    return image


@app.route("/generate_emotion_image", methods=["POST"])
def generate_emotion_image():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data["text"]
    # Detect emotion
    emotion = detect_emotion(input_text)

    # Generate image based on emotion
    image = generate_image(emotion)

    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return jsonify(
        {"input_text": input_text, "detected_emotion": emotion, "image": image_base64}
    )


if __name__ == "__main__":
    app.run(debug=True)
