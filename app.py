import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
import torch
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast

# Create app
app = tk.Tk()
app.geometry("532x642")
app.title("Stable Images")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(height=512, width=512)
lmain.place(x=10, y=10)

modelid = 'CompVis/stable-diffusion-v1-4'
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth="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")
device = "cpu"
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    image.save("generatedImage.png")
    img = ImageTk.PhotoImage(image)

    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to prevent image from being garbage collected

trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
