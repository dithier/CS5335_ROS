import cv2

filename = "C:/New/Videos/Play.mp4"

# Opens the Video file
cap = cv2.VideoCapture(filename)
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('frame' + str(i) + '.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()