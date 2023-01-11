
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi import File, UploadFile

from http import HTTPStatus

app = FastAPI()
    
    
    



model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

def predict_step(image_paths):
#    images = []
#    for image_path in image_paths:
#       i_image = Image.open(image_path)
#       if i_image.mode != "RGB":
#          i_image = i_image.convert(mode="RGB")

#       images.append(i_image)
   pixel_values = feature_extractor(images=image_paths, return_tensors="pt").pixel_values
   pixel_values = pixel_values.to(device)
   output_ids = model.generate(pixel_values, **gen_kwargs)
   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
   preds = [pred.strip() for pred in preds]
   return preds



@app.get('/')
def root():
    response={
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
    }
    return response

from pathlib import Path

@app.post('/transformer/')
async def infer(data: UploadFile=File(...)):
    with open('image.jpg', 'wb') as f:
        content = await data.read()
        f.write(content)
        f.close()
        
    img = Image.open('image.jpg')
    preds = predict_step(img)

    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'prediction': preds
    }

    return response
    