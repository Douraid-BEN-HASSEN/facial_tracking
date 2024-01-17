from ultralytics import YOLO
import cv2
import math

# start webcam
cap = cv2.VideoCapture(1)

# VARS
detect = 'face' # person or face
confidence_threshold = 0.7 # thresold
distance_threshold = 50
scale_percent = 50 # image size
zoomLevel = 1.0
cursorX, cursorY = -1, -1

# models
model = YOLO("faceModel.pt")

# object classes
classNames = ["face"]

while cap.isOpened():
    success, img = cap.read()
    cursor = img.copy()

    # init VARS
    if cursorX == -1 and cursorY == -1:
        cursorX, cursorY = int(img.shape[1]/2), int(img.shape[0]/2)
    valX, valY = -1, -1

    # predict
    results = model(img, stream=True, verbose=False)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            if confidence < confidence_threshold or classNames[cls] != classNames[0]:
                continue
            
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            targetX, targetY, centerX, centerY = int((x1+x2)/2), int((y1+y2)/2), int(img.shape[1]/2), int(img.shape[0]/2)
            distance = math.sqrt((targetX - centerX)**2 + (targetY - centerY)**2)

            if distance > distance_threshold:
                valX = int((cursor.shape[1]/2)/zoomLevel)
                valY = int((cursor.shape[0]/2)/zoomLevel)
                if (targetY-valY >= 0 and (targetY+valY) <= cursor.shape[0] and (targetX-valX) >= 0 and (targetX+valX) <= cursor.shape[1]):
                    cursorX = targetX
                    cursorY = targetY
                    
            point_coordinates = (targetX, targetY)
            point_color = (255, 165, 0)
            point_thickness = -1
            cv2.circle(img, point_coordinates, 2, point_color, point_thickness)

            point_coordinates = (centerX, centerY)
            cv2.circle(img, point_coordinates, 2, point_color, point_thickness)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 3)

            # object details
            org = [x1, y1-10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            color = (255, 165, 0)
            thickness = 2

            cv2.putText(img, classNames[cls] + ' ' + str((confidence*100)) + '%', org, font, fontScale, color, thickness)
            break

    print('y1 => {}, y2 => {}, x1 => {}, x2 => {}, zoom => {}'.format(cursorY-valY, cursorY+valY, cursorX-valX,cursorX+valX, zoomLevel))
    if valX > 0 and valY > 0:
        cursor = cursor[cursorY-valY:cursorY+valY, cursorX-valX:cursorX+valX]

    cursorWidth = int(cursor.shape[1]*scale_percent / 100*zoomLevel)
    cursorHeight = int(cursor.shape[0]*scale_percent / 100*zoomLevel)
    cursorDim = (cursorWidth, cursorHeight)

    # resize cursor
    cursor = cv2.resize(cursor, cursorDim, interpolation = cv2.INTER_AREA)
    cv2.imshow('cursor', cursor)
            
    imageWidth = int(img.shape[1] * scale_percent / 100)
    imageHeight = int(img.shape[0] * scale_percent / 100)
    imageDim = (imageWidth, imageHeight)    
    
    # resize image
    img = cv2.resize(img, imageDim, interpolation = cv2.INTER_AREA)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('+'):
        print('keypressed => +')
        zoomLevel += 0.25
    elif cv2.waitKey(1) == ord('-'):
        print('keypressed => -')
        zoomLevel -= 0.25
        
cap.release()
cv2.destroyAllWindows()