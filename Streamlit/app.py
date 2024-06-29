#Check
import os
import torch
from PIL import Image
import io
import streamlit as st

from transformers import AutoTokenizer,AutoProcessor
from transformers import Blip2ForConditionalGeneration

def load_image_upload():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
    
def load_blip_prompt(image):
    # prompt = "Question: which city is this? Answer:"
    # prompt = input ("Enter the Question: ")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b",torch_dtype=torch.float16)
    model= Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = st.text_input('Enter the Question: ')
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    st.markdown(generated_text)

def main():
    st.title('Image Search Question Answering using BLIP_2')

    img = load_image_upload()
    if img != None :
        load_blip_prompt(img)
#         load_clip_similar(img)

if __name__ == '__main__':
    main()