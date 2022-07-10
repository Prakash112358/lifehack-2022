from cv_service import CVService, MockCVService
from fastapi import FastAPI,BackgroundTasks,Form,File, UploadFile,responses, Response
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
from os import getcwd

helptable = []
elderly = []
volunteers = []
locations = ["Woodlands Ring Road, Block 10, #05-330", "Queensway Road, Block 2, #01-333", "Swee Heng Road, Block 420, #07-090", "Potong Pasir Road, Block 120, #12-530"]
class Volunteers():
    def __init__(self,id,p,e,m,a):
        self.pid = id
        self.password = p
        self.email = e
        self.password = p
        self.mobile = m
        self.personalqn = a
    def update_password(self,p):
        self.password = p
    
class Person():
    def __init__(self, id,n,d,m,location,age,image):
        self.pid = id
        self.name = n
        self.dob = d
        self.mobile = m
        self.location = location
        self.age = age
        self.image = image
        self.status = "ok"

    def get_pid(self):
        return self.pid
    def get_name(self):
        return self.name
    def get_dob(self):
        return self.dob
    def get_mobile(self):
        return self.mobile
    def get_location(self):
        return self.location
    def get_age(self):
        return self.age
    def update_status(self,status):
        self.status = status
    def display(self):
        print("Person ID: ", self.pid)
        print("Name: ", self.name)
        print("DOB: ", self.dob)
        print("Mobile: ", self.mobile)
        print("Location: ", self.location)
        print("Age: ", self.age)
        print("Image: ", self.image)
elderly1 = Person('S501234A', "John", "01/01/1950", "99998888", "Woodlands Ring Road, Block 10, #05-330", "72", "./static/S501234A.jpg")    
elderly2 = Person('S501234B', "Jane", "01/01/1950", "99998888", "Queensway Road, Block 2, #01-333", "72", "./static/S501234B.jpg")
elderly.append(elderly1)
elderly.append(elderly2)
CV_MODEL_DIR = ''
cv_service = MockCVService(model_dir=CV_MODEL_DIR) #initialize model
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/images")
async def getimage(user_id: str):
    img = cv2.imread(user_id + "status.jpg")
    return Response(content=img, media_type="image/jpg")
    
@app.post("/upload/file")
async def upload_file(file: UploadFile = File(...),user_id: str=Form(...)):
    #db = []
    PATH_FILES = getcwd() + '/static/'
    filename = PATH_FILES + user_id + "status.jpg"
    with open(filename, "wb") as f:
        contents = await file.read()
        f.write(contents)
        f.close()
    #db.append(contents)

    img = cv2.imread(file.filename)
    
    # SAVE FILE ORIGINAL
    prediction = cv_service.targets_from_image(img)
    print(prediction[0])
    
    if prediction[0][0][1] == 1:
        for i in elderly:
            if i.get_pid() == user_id:
                i.update_status("help")
                helptable.append(i)
                return {"message": "help"}
                
    else:
        return {"message": "ok"}
        #remove old details from table
        #helptable.pop(helptable.index(user_id))
        #add new details to table
        #helptable.append(user_id)
        #print(elderly)
@app.post("/rescued")
async def rescued(user_id: str=Form(...)):
    for i in helptable:
        if i.get_pid() == user_id:
            i.update_status("ok")
            helptable.pop(helptable.index(i))
            return {"message": "ok"}
    return {"message": "error"}
@app.get("/help")
async def showhelp(background_tasks: BackgroundTasks):
    return helptable

@app.post("/login")
async def login(user_id: str,password: str,background_tasks: BackgroundTasks):
    for i in volunteers:
        if i.pid == user_id and i.password == password:
            return "login successful"
    return "login failed"
@app.post("/login/resetpassword")
async def resetpassword(user_id: str,answer:str,password: str,background_tasks: BackgroundTasks):
    for i in volunteers:
        if i.pid == user_id and i.personalqn == answer:
            i.update_password(password)
            return "password reset successful"
    return "password reset failed"

