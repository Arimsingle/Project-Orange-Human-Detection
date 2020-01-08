import cv2
import numpy as np
import time
import os
from matplotlib import pyplot as plt 
from playsound import playsound #Sound
import requests #Message

# import yolov3 and name
net = cv2.dnn.readNet("yolov3/yolov3-tiny.weights", "yolov3/yolov3-tiny.cfg")
classes = []
with open("name/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# วีดีโอ
cap = cv2.VideoCapture(0)
cap_show = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(os.path.join('Orange.mp4'))
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
count_orange_down = 0
orange_price = 0
c_good = 0
c_orange_roi = 0
c_Not_good = 0
Img_Orange = 'Image/Orange' #Address Orange 
Img_Human = 'Image/Human' #Address Human

#แจ้งเตื่อนผ่านไลน์
def notifyFile(filename):
    file = {'imageFile':open(filename,'rb')}
    payload = {'message':'พบคนเข้าสวนส้มค้าบบบบบ (>.<)'}
    return _lineNotify(payload,file)
def _lineNotify(payload,file=None):
    import requests
    url = 'https://notify-api.line.me/api/notify'
    token = '1yVfOUsQeTDZKQ9dhVb2rBM4fA9DEqO7pf4T5eE1BXn'	
    headers = {'Authorization':'Bearer '+token}
    return requests.post(url, headers=headers , data = payload, files=file)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=33, #White
    varThreshold=15, #Black
    detectShadows=True
)
#รูป
count_image_orange = 1 #Count Image Orange
count_image_human = 1 #Count Image Human
#รันนนนน 
while True:
    _, frame = cap.read()
    #frame = cv2.resize(frame,(1280,800))
    frame_show = frame.copy()
    frame_hed = frame
    frame_Ori = frame
    frame_id += 1 
    height, width, channels = frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    contours_max = []
    BW = fgbg.apply(frame)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)          
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame_Ori, (x, y), (x + w, y + h), (51, 51, 51), 2)
            cv2.putText(frame_Ori, label + " " + str(round(confidence, 2)), (x, y-10), font, 3, (51, 51, 51), 3)
            cv2.circle(frame ,(center_x, center_y),3,(51, 51, 51),-1)
            if(frame_id % 10 == 0):
                if(label == 'orange'):
                    print('Orange Scaned')
                    #playsound('scanner_sound.mp3')
                    cv2.putText(frame_Ori,'.',(130,312),font,3,(0, 0, 255),3,cv2.LINE_AA)
                    playsound('Sounds/scanner_sound.mp3')
                    cv2.imwrite(os.path.join(Img_Orange,"Orange_" + str(count_image_orange) + ".jpg"), frame_show[y:y+h,x:x+w])
                    cv2.waitKey(100)
                    Img_orange=cv2.imread(os.path.join(Img_Orange,"Orange_" + str(count_image_orange) + ".jpg"),cv2.IMREAD_COLOR)
                    Img_orange_roi=cv2.imread(os.path.join(Img_Orange,"Orange_" + str(count_image_orange) + ".jpg"),cv2.IMREAD_COLOR)
                    blurred = cv2.GaussianBlur(Img_orange, (5,5),3)
                    hsv_orange = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    lower_orage=np.array([0,80,70]) #ช่วงสีส้ม
                    up_orage=np.array([23,255,255])
                    mask_orange = cv2.inRange(hsv_orange,lower_orage,up_orage)
                    contours, hierarchy = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    n = len(contours)
                    print('There are %d contours'%n)
                    Img_orange_c=Img_orange
                    cv2.drawContours(Img_orange_c,contours, -1,(255,0,0),1)
                    cv2.imshow("Image", Img_orange_c)
                    count_image_orange+=1 #เลขชื่อภาำพ
                    for i in range(0,n):
                        cnt=contours[i]
                        perimeters = cv2.arcLength(cnt,True)
                        contours_max.append(perimeters)
                    perimeter=max(contours_max)
                    if perimeter > 500:
                        print("Orange Goog")
                        playsound('Sounds/Good.mp3')
                        c_good+=1
                        orange_price +=100
                        cv2.waitKey(100)
                    elif perimeter < 500:
                        print("Orange Not Good")
                        playsound('Sounds/Not_good.mp3')
                        c_Not_good+=1
                        orange_price+=50
                        cv2.waitKey(100)
                    print("Orange Good:%d Orange Not good:%d"%(orange_price,c_Not_good))
                    print("Price:%d"%orange_price)
                    if(center_y > 312):
                        count_orange_down+=1
                        playsound('Sounds/Down.mp3')
                        cv2.line(frame,(500,320),(130,320),(0, 0, 255),10)
                        print("Count:%d"%count_orange_down)
                    Img_orange_roi_gray = cv2.cvtColor(src = Img_orange_roi,code=cv2.COLOR_BGR2GRAY)
                    Img_orange_roi_blur = cv2.GaussianBlur(src=Img_orange_roi_gray,ksize=(5,5),sigmaX=0)
                    theash_val,Img_orange_roi_thresh = cv2.threshold(Img_orange_roi_blur,0,255,cv2.THRESH_OTSU)
                    #cv2.imwrite(os.path.join(Img_Orange,"Orange_roi" + str(count_image_orange) + ".jpg"), Img_orange_roi_thresh)
                    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
                    closing = cv2.morphologyEx(Img_orange_roi_thresh, cv2.MORPH_CLOSE,se)
                    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE,se)
                    contours2, hierarchy2 = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    n2 = len(contours2)
                    Img_orange_roi_color = Img_orange_roi
                    cv2.drawContours(Img_orange_roi_color,contours2, -1,(0,0,255),1)
                    cv2.imshow("Image_roi", Img_orange_roi_color)
                    for i in range(0,n2):
                        cnt2=contours2[i]
                        perimeters2 = cv2.arcLength(cnt2,True)
                        print(perimeters2)
                    if perimeters2 > 410 and perimeters2 < 700:
                        c_orange_roi+=1
            if(label=='person'):
               if(frame_id % 30 == 0):
                   cv2.imwrite(os.path.join(Img_Human,"Human_" + str(count_image_human) + ".jpg"), frame_show[y:y+h,x:x+w])
                   cv2.waitKey(100)
                   notifyFile("Image/Human/Human_" + str(count_image_human) + ".jpg")
                   count_image_human+=1
               print('Person Scaned')
               cv2.imshow("frame_BW", BW[y:y+h,x:x+w])
    cv2.line(frame,(500,320),(130,320),(0, 104, 255),2)
    cv2.putText(frame, ".|Orange&Human Detection|.", (88, 40), font, 2, (0, 50, 255), 2)
    cv2.putText(frame, ".|Orange&Human Detection|.", (90, 40), font, 2, (0, 150, 255), 2)
    cv2.putText(frame,'Donw:'+str(count_orange_down),(150,312),font,2,(0, 104, 255),1,cv2.LINE_AA)
    cv2.putText(frame, "Good:" + str(c_good), (30, 450), font, 2, (0, 104, 255), 1)
    cv2.putText(frame, "Not Good:" + str(c_Not_good), (220, 450), font, 2, (0, 104, 255), 1)
    cv2.putText(frame, "Price:" + str(orange_price), (450, 450), font, 2, (0, 104, 255), 1)
    cv2.imshow("frame_Ori", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    file1 = open("Image/Quality.txt","w+")
    L = ["Orange Good:%d \nOrange Not Good:%d \nDown:%d \nTotal Price:%d \nNo quality:%d"%((c_good),c_Not_good,count_orange_down,(orange_price-(c_orange_roi*100)),c_orange_roi)]
    file1.writelines(L)
 
cap.release()
cv2.destroyAllWindows()


