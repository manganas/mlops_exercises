from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
import re

app = FastAPI()

class ItemEnum(Enum):
	alexnet='alexnet'
	resnet='resnet'
	lenet='lenet'


@app.get('/')
def root():
	'''
	Health check
	'''
	response = {
	'message': HTTPStatus.OK.phrase,
	'status-code': HTTPStatus.OK,
	}
	return response

@app.get('/restric_items/{item_id}')
def read_item(item_id: ItemEnum):
	return {'item_id': item_id}

@app.get("/query_items")
def read_item(item_id: int):
	return {'item_id': item_id}



database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
	username_db = database['username']
	password_db = database['password']
	if username not in username_db and password not in password_db:
		with open('database.csv', 'a') as f:
			f.write(f"{username}, {password} \n")
		username_db.append(username)
		password_db.append(password)

		return 'login saved'

	return 'user or password exists'

from pydantic import BaseModel
from typing import Union

class Item(BaseModel):
	email:str
	domain_match: str

@app.get("/check_domain/")
def is_domain(data: Item):
	return {data.email: data.domain_match}

@app.get("/text_model/")
def contains_email(data: str):
	regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
	response={
		'input':data,
		'message': HTTPStatus.OK.phrase,
		'status-code': HTTPStatus.OK,
		"is_email": re.fullmatch(regex, data) is not None
		}
	return response


from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from typing import Optional
import cv2


@app.post("/cv_model/")
async def cv_model(data: UploadFile=File(...), h: Optional[int]=28, w: Optional[int]=28):
	with open('image.jpg', 'wb') as image:
		content = await data.read()
		image.write(content)
		image.close()

	img = cv2.imread('image.jpg')
	res = cv2.resize(img, (h,w))

	cv2.save('image_resized.jpg', res)

	response = {
		'input': data,
		'output': FileResponse('image_resized.jpg'),
		'message': HTTPStatus.OK.phrase,
		'status-code': HTTPStatus.OK,
	}

	return response













