import cv2
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier("haarcascade_smile.xml")

class VideoCamera(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)

    def __del__(self):
        self.video.releast()
    
    def get_frame(self):
        ret, frame=self.video.read()

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            roi_gray=gray[y:y+h,x:x+w]
            smiles=smile_cascade.detectMultiScale(roi_gray,
                scaleFactor=1.5,
                minNeighbors=15,
                minSize=(25,25))
            
            for i in smiles:
                if len(smiles)>1:
                    cv2.putText(frame,"Smiling",(30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3,
                        cv2.LINE_AA)
            break

        ret, jpeg=cv2.imencode('.jpg',frame)
        return jpeg.tobytes()

        
