import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker


model=YOLO('yolov8n.pt')


#area1=[(312,388),(289,390),(474,469),(497,462)]

#area2=[(279,392),(250,397),(423,477),(454,469)]

area1=[(86,330),(925,330),(1000,377),(3,377)]

area2=[(170,280),(830,280),(900,315),(98,315)]

# Mouse callback function to display pixel position
mouse_position = (0, 0)  # To store mouse position globally

def display_pixel_position(event, x, y, flags, param):
    global mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position = (x, y)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', display_pixel_position)

cap=cv2.VideoCapture('peoplecount.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
tracker=Tracker()
entering={}
exiting={}
entered=set()
exited=set()

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    # count += 1
    # if count % 2 != 0:
    #     continue
    frame=cv2.resize(frame,(1020,500))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
    #print(f".....results....:{results}")
    
    #results[0]: Accesses the first result in the results list. If you process multiple frames at once, results may contain multiple Results objects. Here, we assume one frame, so we take results[0].
    #results[0].boxes: The .boxes attribute contains the bounding box data. Each box usually includes:
    #1=Class label (representing the detected object category, e.g., "person").
    #2=Confidence score (indicating how confident the model is in this detection),
    #3=Coordinates of the bounding box (like [x1, y1, x2, y2]),
    #.data: Accesses the underlying data as a tensor, array, or similar format. In YOLOv8, .data provides a raw tensor with the bounding box details, confidence, and class information.
    
#     results = [
#     # This would be results[0] - information for the first (and typically only) image in batch
#     {
#         'boxes': {
#             'data': [
#                 [50, 100, 200, 300, 0.95, 0],   # Person 1: x1, y1, x2, y2, confidence, class_id
#                 [250, 150, 400, 450, 0.88, 0],  # Person 2: x1, y1, x2, y2, confidence, class_id
#                 [300, 500, 450, 700, 0.85, 1]   # Car: x1, y1, x2, y2, confidence, class_id
#             ]
#         }
#     }
# ]
    a=results[0].boxes.data
    #print(f".....results[0]....:{results[0].boxes}")
    
    #converts tensor to pandas for easy manipulation
    px=pd.DataFrame(a).astype("float")
    #print(f".....px.....:{px}")
    person_list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            person_list.append([x1,y1,x2,y2])
            
            
    persons=tracker.update(person_list)
    
    for person in persons:
        x3,y3,x4,y4,id=person
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
        # entering
        result2=cv2.pointPolygonTest(np.array(area2,np.int32), (x4,y4), False)    
        if result2>-1:
            entering[id]=(x4,y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
            cv2.putText(frame,str(),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
            
        if id in entering:
            result1=cv2.pointPolygonTest(np.array(area1,np.int32), (x4,y4), False)
            if result1>-1:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1) 
                entered.add(id)
                break
        
        # exiting
        result3=cv2.pointPolygonTest(np.array(area1,np.int32), (x4,y4), False)    
        if result3>-1:
            exiting[id]=(x4,y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
            cv2.putText(frame,str(),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
            
        if id in exiting:
            result4=cv2.pointPolygonTest(np.array(area2,np.int32), (x4,y4), False)
            if result4>-1:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1) 
                exited.add(id)
                break
   
            
    #print(people_entering)    
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
    
    cv2.putText(frame,str(f"Entered:{len(entered)}"),(10,20),cv2.FONT_HERSHEY_COMPLEX,(0.6),(0,255,0),2)
    cv2.putText(frame,str(f"Exited:{len(exited)}"),(10,60),cv2.FONT_HERSHEY_COMPLEX,(0.6),(0,0,255),2)

    cv2.putText(frame, str(f"Mouse Pos: {mouse_position}"), (800, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

