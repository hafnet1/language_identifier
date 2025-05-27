import gradio as gr
from fastai.vision.all import *
from pathlib import Path
import pathlib


# Fix for Windows loading a Unix-trained learner
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

current_dir = Path.cwd()

# Load the model
print("Loading model...")
def get_label(path):
    return path.name.split('_')[0]
learn = load_learner(current_dir / 'language_identifier_model.pkl')
print("Model loaded.")

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Language Identifier"
description = "A computer vision model that identifies the language of black text on white background"
examples = ['']
interpretation='default'

gr.Interface(fn=predict,inputs=gr.Image(),outputs=gr.Label(num_top_classes=5),title=title,description=description).launch(share=True)