import cv2
import numpy as np
import redis
import time
import json
import base64
from records import database_model

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
db_obj = database_model()
db_obj.create_table()

def process_frames():
    while True:
        face_data_json = redis_client.lpop('face_recognition_queue')

        if face_data_json is None:
            time.sleep(0.01)
            continue

        data = json.loads(face_data_json)
        frame_bytes_data = data['frame']
        mapped_faces = data['mapped_faces']
        detected_human_id = mapped_faces[0]['label']
        position = mapped_faces[0]['position']
        confidence = mapped_faces[0]['similarity_score']

        # Convert the byte data back into a NumPy array and decode as an image
        frame_bytes = base64.b64decode(frame_bytes_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        #  **************************** STORE DATA IN DB *******************
        db_obj.insert_data(frame,detected_human_id,position,confidence)

if __name__ == "__main__":
    process_frames()
