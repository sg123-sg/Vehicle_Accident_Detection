# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import cv2

'''CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]'''

# There is a fix index for class, location and confidence
CLASSES = ["", "", "", "", "",
    "", "", "car", "", "", "", "",
    "", "", "", "", "", "",
    "", "", ""]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES)))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

print("[INFO] starting video stream...")
vs = cv2.VideoCapture('video500kb.mp4')
#vs = cv2.VideoCapture('yt1s.com - Reckless Driver Missed Exit Causes Two Semi Trucks to Crash Spectacularly_360p.mp4')
print(vs.get(3), vs.get(4))
#print(vs.set(3,320), vs.set(4,240))
fps = FPS().start()
flag = 0
# loop over the frames from the video stream

while (True):
    ret, frame = vs.read()
    #cv2.imwrite('test.jpg',frame)
    #frame = cv2.imread('test.jpg')
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if(confidence > 0.2): #Confidence of prediction
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = CLASSES[idx]
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            #print (confidence)

            if label == 'car':
                w = startX + int((endX - startX) / 2)
                h = startY + int((endY - startY) / 2)
                print((w, h))
                cv2.circle(frame, (w, h), 5, (0, 255, 0), 5)
                if (endX < 160):
                    cv2.putText(frame, 'car', (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    flag = 1

                if (flag == 1):
                    if (endX > 160):
                        print('Reckless Detection.')
                        cv2.putText(frame, "Reckless Detected.", (20, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    fps.update()

    
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

    
