import cv2, time, pandas
from datetime import datetime

firstFrame = None
statusList = [None, None]
timesList = []
df = pandas.DataFrame(columns = ["Start","End"])

vid = cv2.VideoCapture(0)

while True:
    check, frame = vid.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if firstFrame is None:
        firstFrame = gray
        continue

    deltaFrame = cv2.absdiff(firstFrame,gray)
    threshFrame = cv2.threshold(deltaFrame, 30, 255, cv2.THRESH_BINARY)[1]
    threshFrame = cv2.dilate(threshFrame, None, iterations = 2)

    (cnts,_) = cv2.findContours(threshFrame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000: #pixels
            continue
        status = 1 #when motion is detected change status to 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    statusList.append(status)

    #statusList = statusList[-2:]

    #detection start time - object appears
    if statusList[-1] == 1 and statusList[-2] == 0:
        timesList.append(datetime.now())

    #detection end time - object appears
    if statusList[-1] == 0 and statusList[-2] == 1:
        timesList.append(datetime.now())

    #Open webcam windows with specified names
    #cv2.imshow("Gray frame", gray)
    #cv2.imshow("Delta frame", deltaFrame)
    cv2.imshow("Thresh frame", threshFrame)
    cv2.imshow("Color frame", frame)

    #to exit press q
    keyToExit = cv2.waitKey(1)

    if keyToExit == ord('q'):
        if status == 1:
            timesList.append(datetime.now())
        break
#end of while loop

print(statusList)
print(timesList)

for i in range(0, len(timesList), 2):
    df = df.append({"Start": timesList[i], "End": timesList[i+1]},ignore_index = True)

df.to_csv("Detected times.csv")

vid.release()
cv2.destroyAllWindows()
