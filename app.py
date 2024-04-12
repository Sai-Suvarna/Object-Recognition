import uvicorn
from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import cv2
import pickle
import numpy as np
import os
import requests
import subprocess
import shutil
from fastapi.responses import RedirectResponse
import easyocr
import psycopg2
import os
from dotenv import load_dotenv
import pathlib
import textwrap

import google.generativeai as genai
import PIL.Image

from IPython.display import display
from IPython.display import Markdown
from google.cloud import vision
from google.cloud.vision_v1 import types
import os

from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import requests
import string
from bs4 import BeautifulSoup


load_dotenv()

UPLOAD_FOLDER='static'

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')



@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("open.html", {"request": request})

@app.get('/upload')
def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get('/result')
def index(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})




def fetch_from_knowledge_graph(query):
    api_key = "AIzaSyBHQghHp4b6DdrlAftGluFOe6a6WQQpAn0"  # Replace with your actual API key
    api_endpoint = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": query,
        "key": api_key
    }
    response = requests.get(api_endpoint, params=params)
    data = response.json()
    return data

def display_knowledge_graph_data(data, query,upimage):
    results = []
    unique_names = set()  # Maintain a set of unique names
    
    if "itemListElement" in data:
        for item in data["itemListElement"]:
            name = item["result"]["name"]
            print(name)
            if name.lower() == query.lower() and name not in unique_names:  # Check if the name is unique
                item_image = item["result"].get("image", {}).get("contentUrl", "No image available")
                if item_image != "No image available":
                    
                    description = item["result"].get("detailedDescription", {}).get("articleBody", "No detailed description available")

                    detailed_description = item["result"].get("detailedDescription", {}).get("url", "No detailed description available")

                    result_dict = {
                        "Name": name,
                        "Description": description,
                        "Detailed Description": detailed_description,
                        "item_image": item_image
                    }
                    results.append(result_dict)
                    unique_names.add(name)
                else:
                    model1 = genai.GenerativeModel('gemini-pro')
                    query="Give me a description of 60 words about" + name
                    response = model1.generate_content(query)
                    description=response.text

                    detailed_description = item["result"].get("detailedDescription", {}).get("url", "No detailed description available")
                    item_image=upimage

                    result_dict = {
                        "Name": name,
                        "Description": description,
                        "Detailed Description": detailed_description,
                        "item_image": item_image
                    }
                    results.append(result_dict)
                    unique_names.add(name)

    return results



@app.post("/upload_image", response_class=HTMLResponse)
async def upload_image( request: Request,image_file: UploadFile = File(...)):
    image_path = f"{UPLOAD_FOLDER}/{image_file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    
    with open(save_path, "wb") as image:
        content = await image_file.read()
        image.write(content)
         
    img = PIL.Image.open(image_path)

    response = model.generate_content(["Identify the only some important things that are in the image.I should have the response only consist of names of all the objects name in a single word for each one without any stopwords in the object names separated by comma in the image", img], stream=True)
    response.resolve()
    
    res = response.text.split(',')
    res = [word.strip() for word in res if word.strip()]
    
    object_results = []
    for obj in res:
        data = fetch_from_knowledge_graph(obj)
        object_data = display_knowledge_graph_data(data, obj,save_path)
        object_results.extend(object_data)
    
    print(object_results)

    
    context = {
        "request": request,
        "res":res,
        "object_results":object_results,
        "image_path": image_path
        
    }


    return templates.TemplateResponse("result.html", context)



    
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)