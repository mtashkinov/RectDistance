import cv2

video = cv2.VideoCapture(0)
name = 234

while True:
    flag, frame = video.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(frame, 1)
    cv2.imshow('frame', gray)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('w'):
        cv2.imwrite(str(name) + '.png', gray)
        name += 1
    elif k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
