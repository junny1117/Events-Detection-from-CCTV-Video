import cv2

class VideoStream:
    def __init__(self, source):
        
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source {source}")

    def get_frame(self):
       
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def __del__(self):

        if self.cap.isOpened():
            self.cap.release()
