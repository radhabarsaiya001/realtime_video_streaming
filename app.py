import cv2
import time 
from flask import Flask, render_template, Response
import numpy as np

video_path =  r'rtsp://admin:vinayan@123@192.168.1.120:554/1/1'
cap = cv2.VideoCapture(video_path)
app = Flask(__name__)
def gen_frames():
    while(cap.isOpened()):         
        ret, frame = cap.read() 
        logo = cv2.imread('logo.png')
        logo_resize = cv2.resize(logo, (100,100), interpolation=cv2.INTER_AREA)
        rows,cols, channels = logo_resize.shape
        logo_gray = cv2.cvtColor(logo_resize,cv2.COLOR_BGR2GRAY)
        ret,mask = cv2.threshold(logo_gray,220,255,cv2.THRESH_BINARY)
        inversemask = cv2.bitwise_not(mask)
        if not ret: 
            break
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_data = f"fps{int(fps)}"
            roi = frame[0:rows, 0:cols]
            # print("roi is : ",roi)
            
            background = cv2.bitwise_and(roi,roi,mask =mask)
            foreground = cv2.bitwise_and(logo_resize,logo_resize,mask= inversemask)
            dst = cv2.add(background,foreground)
            frame[0:rows, 0:cols] = dst
            cv2.putText(frame, fps_data, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Hii, It's Vinayan", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        except :
            pass
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
