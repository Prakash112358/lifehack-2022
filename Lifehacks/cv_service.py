'''

Before running anything, remember to:

1. !git clone https://github.com/ultralytics/yolov5
2. cd yolov5
3. !pip install -r requirements.txt 
4. Run this py file


'''

from typing import List, Any
from tilsdk.cv.types import *
#import onnxruntime as ort
import torch
import os
import sys


class CVService:
	def __init__(self, model_dir):
	
		self.model_dir = model_dir
		self.new_model = torch.hub.load('','custom', path=self.model_dir,force_reload=True,source='local')
		self.new_model.conf = 0.5 ########################### MUST TUNE THIS ########################
		print(" model loaded")
       
	 # Convert Pascal_Voc bb to Coco b
	def _pascal_voc_to_coco(self, x1, y1, x2, y2):
		return [x1,y1, x2 - x1 + 1, y2 - y1 + 1]
	def targets_from_image(self, img) -> List[DetectedObject]:    
        
		#T: Participant to complete.
        
		# Load old model with just weights
		
		# hubconf.py should be same dir as current
		results = self.new_model(img)
		temp = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
		yes = eval(temp)
	
		all_info = []
	
		for a in range(len(yes)):

			voc_bbox = (yes[a]['xmin'],yes[a]['ymin'],yes[a]['xmax'],yes[a]['ymax']) # put in tuple the xmin, ymin, xmax, ymax
			#print(voc_bbox)
			ymax = voc_bbox[3]
			xmax = voc_bbox[2]
			
			newbbox = self._pascal_voc_to_coco(voc_bbox[0], voc_bbox[1], voc_bbox[2], voc_bbox[3])
			#print("bbox: ",newbbox)
			xcenter = (xmax - newbbox[0]) / 2
			ycenter = (ymax - newbbox[1]) / 2
			
			newbbox[0] = xcenter
			newbbox[1] = ycenter
			
			#print(newbbox)
			
			score = round(float(yes[a]['confidence']),4)
			categoryy = yes[a]['class'] + 1
			#print("score: ",score) 
			if categoryy == 2:
				categoryy = 0
      		
      	
			lst = []
			for i in range(0, len(newbbox)):
				lst.append(float(newbbox[i]))
				lst[i] = round(lst[i], 1)
        		
			all_info.append((categoryy, score, lst))
			#print(all_info)
	
		return all_info 


class MockCVService:

	def __init__(self, model_dir:str):
		self.model_dir = model_dir
		self.cv = CVService("./yolov5/exp14(1).pt")


	def targets_from_image(self, img:Any) -> List[DetectedObject]:
        	
		tup = self.cv.targets_from_image(img) #Tuple stores all_info needed
        
        
		lst = []
		
		for i in range(0, len(tup)):  # Itr through all bbox  	
			bbox = BoundingBox(tup[i][2][0],tup[i][2][1],tup[i][2][2],tup[i][2][3])
			#print(bbox)
			target_id = 1   ####### How to find target_id #### ??
			
			obj = DetectedObject(target_id, tup[i][0], bbox) 
			lst.append(obj)
			
        	
		#print(lst)
		return lst,img

        
        
       
#### INDEPENDENT TESTING #######
'''

mock = MockCVService("./exp14(1).pt")

predictions = []  

prediction = mock.targets_from_image("./video50.jpg")# Call cv_service for the prediction
predictions.append(prediction)   
predictions.append([DetectedObject(id=1, cls=1, bbox=BoundingBox(x=119.4, y=53.0, w=239.9, h=106.9)), DetectedObject(id=1, cls=1, bbox=BoundingBox(x=37.7, y=37.5, w=76.4, h=76.0))])

print("Predictions: ", predictions)

newList = []       
for i in range(0, len(predictions)): #itr through all images
	tempList = []
	for j in range(0, len(predictions[i])): # itr through all bounding box in an image
          tempList.append(predictions[i][j].bbox.w * predictions[i][j].bbox.h) #width * height to find area
	print("TempList = ", tempList)

	maximum = 0
	for j in range(0, len(tempList)):
          if tempList[j] > maximum:
                maximum = tempList[j] 
	newList.append((maximum, i))
                
newList.sort()
print("newList = ", newList)
bestResult = newList[-1] 
toSubmit = predictions[bestResult[1]]			 

print(toSubmit)

                # Tomorrow take photo of background images from maze
                # Tomorrow take shots & save them of robot in different angles trying to predict the image
'''                

