import cv2
import dlib
import math
Blink_ratio_thresold = 5.7

def midpoint(point1,point2):
    return((point1.x + point2.x)/2,(point1.y+point2.y)/2)

def euclidean_distance(point1,point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
def get_blink_ratio(eye_points,facial_landamrks):
    corner_left = (facial_landamrks.part(eye_points[0]).x,
                   facial_landamrks.part(eye_points[0]).y)

    corner_right = (facial_landamrks.part(eye_points[3]).x,
                   facial_landamrks.part(eye_points[3]).y)
    
    center_top = midpoint(facial_landamrks.part(eye_points[1]),
                         facial_landamrks.part(eye_points[2]))

    center_bottom = midpoint(facial_landamrks.part(eye_points[5]),
                              facial_landamrks.part(eye_points[4]))

    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)
    
    ratio = horizontal_length / vertical_length

    return ratio

cap=cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\openCV\Blink_detector\shape_predictor_68_face_landmarks.dat")
left_eye_landmarks = [36,37,38,39,40,41]
right_eye_landmarks=[42,43,44,45,46,47]
while True:
    retval,frame = cap.read()
    if not retval:
        print("Can't recieve frame")
        break

    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces,_,_ = detector.run(image = gray_img, upsample_num_times = 0, 
                       adjust_threshold = 0.0)
    for face in faces:
        landmarks=predictor(gray_img,face)

        left_eye_ratio = get_blink_ratio(left_eye_landmarks,landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks,landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio > Blink_ratio_thresold:
            cv2.putText(frame,"BLINKING",(10,50),cv2.FONT_HERSHEY_SIMPLEX,
            2,(255,0,0),2)
    
    cv2.imshow('BlinkDetector',frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()