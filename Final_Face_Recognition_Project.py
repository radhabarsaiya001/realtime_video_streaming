import cv2
import multiprocessing
from mtcnn_ort import MTCNN
from numpy import asarray, expand_dims
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from keras_facenet import FaceNet
from numpy import asarray, expand_dims, load
import time
import redis
import json
import base64
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)


# Load pre-trained models
svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_face_model.pkl")
label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_label_encoder.pkl")
model = FaceNet()
detector = MTCNN()
trained_faces = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\face_embeddings_wth_labels.npz")


# Process a single frame for face detection
def detect_faces(frame_queue, embedding_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)
        frame_face_positions = []

        # Process each face detected in the frame
        for face in faces:
            x, y, w, h = face['box']
            if face['confidence'] > 0.98:
                frame_face_positions.append(face['box'])

        if len(frame_face_positions) > 0:
            embedding_queue.put((frame, frame_face_positions))


# Capture frames from the RTSP stream
def capture_frames(rtsp_url, frame_queue):
    # cap = cv2.VideoCapture(rtsp_url)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

# Extract face embeddings using FaceNet
def extract_embeddings(embedding_queue, recognition_queue):
    while True:
        data = embedding_queue.get()
        if data is None:
            break

        frame, face_positions = data
        frame_face_arrs = []
        for frame_face_position in face_positions:
            x, y, w, h = frame_face_position
            face_arr = frame[y:y + h, x:x + w]
            face_arr = cv2.resize(face_arr, (160, 160))
            frame_face_arrs.append(face_arr)

        frame_face_arrs = asarray(frame_face_arrs)
        frame_all_embeddings = []

        for face in frame_face_arrs:
            samples = expand_dims(face, axis=0)
            yhat = model.embeddings(samples)
            frame_all_embeddings.append(yhat[0])

        # Encode the frame in JPEG format (convert to bytes)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        if len(frame_all_embeddings) > 0:
            # recognition_queue.put((frame, face_positions,asarray(frame_all_embeddings)))
            recognition_queue.put((frame_bytes, face_positions,asarray(frame_all_embeddings)))

# Perform classification and similarity checks on embeddings
def recognize_faces(recognition_queue):
    mapped_faces = []
    
    while True:
        data = recognition_queue.get()
        if data is None:
            break

        # frame, face_positions, frame_all_embeddings = data
        frame_bytes, face_positions, frame_all_embeddings = data

        # Cache trained embeddings for efficient lookup
        trained_face_labels = trained_faces['arr_1']
        trained_face_embeddings = trained_faces['arr_0']

        for idx, embedding in enumerate(frame_all_embeddings):
            sample_embedding = expand_dims(embedding, axis=0)
            
            # Predict class using the SVC model
            predicted_class = svc_model.predict(sample_embedding)
            predicted_label = label_encoder.inverse_transform(predicted_class)[0]

            # Efficiently find the matching label and corresponding embedding
            if predicted_label in trained_face_labels:
                label_idx = trained_face_labels.tolist().index(predicted_label)
                training_embedding = trained_face_embeddings[label_idx]

                training_embedding = expand_dims(training_embedding, axis=0)
            else:
                # If label not found in trained data, skip processing
                continue

            # Calculate similarity score
            similarity_score = cosine_similarity(training_embedding, sample_embedding)[0][0]
            similarity_score = round(similarity_score * 100, 3)

            # Handle face position and mapping
            if idx < len(face_positions):
                x, y, w, h = face_positions[idx]
                face_info = {
                    'position': (x, y, w, h),
                    'label': predicted_label if similarity_score >= 60 else "Unknown",
                    'similarity_score': similarity_score
                }
                mapped_faces.append(face_info)

        # Convert frame bytes to a base64 string
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')


        # Store the frame and mapped_faces in Redis
        frame_data = {
            'frame': frame_base64,  ## Store the frame as a base64 string
            'mapped_faces': mapped_faces
        }

        redis_client.rpush('face_recognition_queue', json.dumps(frame_data))
        print("data send successfully!!")

        mapped_faces = []

        # return frame, mapped_faces
        # print("***************************")
        # print(frame_data)
        # time.sleep(2)
        # print(mapped_faces)
        # # # yield frame, mapped_faces
        # mapped_faces = []


# Main function to start the processes
def main():
    rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"

    # Create multiprocessing queues
    frame_queue = multiprocessing.Queue(maxsize=5)
    embedding_queue = multiprocessing.Queue(maxsize=5)
    recognition_queue = multiprocessing.Queue(maxsize=5)

    # Start the frame capture process
    capture_process = multiprocessing.Process(target=capture_frames, args=(rtsp_url, frame_queue))
    capture_process.start()

    # Start the face detection process
    detection_process = multiprocessing.Process(target=detect_faces, args=(frame_queue, embedding_queue))
    detection_process.start()

    # Start the face embedding extraction process
    embedding_process = multiprocessing.Process(target=extract_embeddings, args=(embedding_queue, recognition_queue))
    embedding_process.start()

    # Start the face recognition process
    recognition_process = multiprocessing.Process(target=recognize_faces, args=(recognition_queue,))
    recognition_process.start()

    # Wait for the processes to finish
    capture_process.join()
    frame_queue.put(None)  # Signal the processes to exit
    detection_process.join()
    embedding_queue.put(None)  # Signal the embedding process to exit
    embedding_process.join()
    recognition_queue.put(None)  # Signal the recognition process to exit
    recognition_process.join()

if __name__ == "__main__":
    main()