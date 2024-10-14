import cv2
from numpy import asarray, expand_dims, load
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from keras_facenet import FaceNet

# Load pre-trained models
# svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_face_model.pkl")
# label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_label_encoder.pkl")
# Face_embedding = FaceNet()
# detector = MTCNN()
# data = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\face_embeddings_wth_labels.npz")

class FRS:
    def __init__(self, detector, Face_embedding, svc_model, label_encoder, data, rtsp_url, frame_queue, embedding_queue):
        self.detector = detector
        self.Face_embedding = Face_embedding
        self.svc_model = svc_model
        self.label_encoder = label_encoder
        self.data = data 
        self.frame_queue = frame_queue
        self.embedding_queue = embedding_queue
        self.rtsp_url = rtsp_url
        # self.recognition_queue = recognition_queue

    # Process a single frame for face detection
    def detect_faces(self):
        # frame = self.frame_queue.get()
        frame = self.capture_frames()
        if frame is None:
            print("Frame not valid!!")
            pass
        # Detect faces using MTCNN
        faces = self.detector.detect_faces(frame)
        frame_face_arr = []
        frame_face_positions = []

        # Process each face detected in the frame
        for face in faces:
            x, y, w, h = face['box']
            if face['confidence'] > 0.98:
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # face_arr = frame[y:y + h, x:x + w]
                # face_arr = cv2.resize(face_arr, (160, 160))
                # frame_face_arr.append(face_arr)
                frame_face_positions.append(face['box'])
    
        # if len(frame_face_arr) > 0:
        #     frame_face_arr = asarray(frame_face_arr)
            # self.embedding_queue.put(frame_face_arr)

        face_arr = frame[y:y + h, x:x + w]
        face_arr = cv2.resize(face_arr, (160, 160))
        frame_face_arr = asarray(frame_face_arr)
        return frame, frame_face_positions
    
    # Capture frames from the RTSP stream
    def capture_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        # cap = cv2.VideoCapture(0)
        # while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280,720))
        if not ret:
            print("Failed to grab frame")
            # break
            pass
        # if not self.frame_queue.full():
        #     self.frame_queue.put(frame)
        
        return frame
        # cap.release()

    # Extract face embeddings using FaceNet
    def extract_embeddings(self):
        frame_face_arr = self.embedding_queue.get()
        if frame_face_arr is None:
            print("Not valid face array!!")
            pass
        frame_all_embedding = []
        for face in frame_face_arr:
            samples = expand_dims(face, axis=0)
            print("******************************")
            print(self.Face_embedding)
            print("******************************************")
            yhat = self.Face_embedding.embeddings(samples)
            frame_all_embedding.append(yhat[0])
        if len(frame_all_embedding) > 0:
            # self.recognition_queue.put(asarray(frame_all_embedding))
            return asarray(frame_all_embedding)

    # Perform classification and similarity checks on embeddings
    def recognize_faces(self):
        # frame_all_embedding = self.recognition_queue.get()
        frame_all_embedding = self.extract_embeddings()
        if frame_all_embedding is None:
            print("Not valid Embedding !!")
            pass

        for embedding in frame_all_embedding:
            sample_embedding = expand_dims(embedding, axis=0)
            predicted_class = self.svc_model.predict(sample_embedding)
            predicted_label = self.label_encoder.inverse_transform(predicted_class)[0]
            cnt = 0
            for i in self.data['arr_1']:
                if i == predicted_label:
                    training_embedding = self.data['arr_0'][cnt][:]
                    training_embedding = expand_dims(training_embedding, axis=0)
                    break
                cnt += 1
            similarity_score = cosine_similarity(training_embedding, sample_embedding)
            similarity_score = round(similarity_score[0][0] * 100, 3)
            if similarity_score < 60:
                # print('Unknown', f"{similarity_score}%")
                predicted_label = "Unknown"
                return predicted_label, similarity_score
            else:
                # print(predicted_label, f"{similarity_score}%")
                return predicted_label, similarity_score



    # Main function to start the processes
    def main(self):
        
        # Start the frame capture process
        # capture_process = multiprocessing.Process(target=self.capture_frames)
        # capture_process.start()

        # Start the face detection process
        # detection_process = multiprocessing.Process(target=self.detect_faces, args=(frame_queue, embedding_queue))
        # detection_process.start()
        detection_process = multiprocessing.Process(target=self.detect_faces)
        detection_process.start()

        # Start the face embedding extraction process
        # embedding_process = multiprocessing.Process(target=extract_embeddings, args=(embedding_queue, recognition_queue))
        # embedding_process.start()

        # Start the face recognition process
        recognition_process = multiprocessing.Process(target=self.recognize_faces)
        recognition_process.start()

        # # Wait for the processes to finish
        # capture_process.join()
        # frame_queue.put(None)  # Signal the processes to exit
        detection_process.join()
        # embedding_queue.put(None)  # Signal the embedding process to exit
        # embedding_process.join()
        # recognition_queue.put(None)  # Signal the recognition process to exit
        recognition_process.join()

if __name__ == "__main__":
    self.main()


# svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_face_model.pkl")
# label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_label_encoder.pkl")
# Face_embedding = FaceNet()
# detector = MTCNN()
# data = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\face_embeddings_wth_labels.npz")
# rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"
#  # Create multiprocessing queues
# frame_queue = multiprocessing.Queue(maxsize=5)
# embedding_queue = multiprocessing.Queue(maxsize=5)
# # recognition_queue = multiprocessing.Queue(maxsize=5)

# FRS_obj = FRS(detector, Face_embedding, svc_model, label_encoder, data, rtsp_url, frame_queue, embedding_queue)
 

# import cv2
# from numpy import asarray, expand_dims, load
# from sklearn.metrics.pairwise import cosine_similarity
# import multiprocessing
# from keras_facenet import FaceNet
# from mtcnn import MTCNN
# import joblib

# class FRS:
#     def __init__(self, detector, Face_embedding, svc_model, label_encoder, data, rtsp_url, frame_queue, embedding_queue):
#         self.detector = detector
#         self.Face_embedding = Face_embedding
#         self.svc_model = svc_model
#         self.label_encoder = label_encoder
#         self.data = data
#         self.frame_queue = frame_queue
#         self.embedding_queue = embedding_queue
#         self.rtsp_url = rtsp_url

#     # Process a single frame for face detection
#     def detect_faces(self):
#         while True:
#             frame = self.capture_frames()
#             if frame is None:
#                 print("Frame not valid!")
#                 continue

#             faces = self.detector.detect_faces(frame)
#             frame_face_arr = []
#             frame_face_positions = []

#             for face in faces:
#                 x, y, w, h = face['box']
#                 if face['confidence'] > 0.97:
#                     face_arr = frame[y:y + h, x:x + w]
#                     face_arr = cv2.resize(face_arr, (160, 160))
#                     frame_face_arr.append(face_arr)
#                     frame_face_positions.append(face['box'])

#             if len(frame_face_arr) > 0:
#                 self.embedding_queue.put(asarray(frame_face_arr))

#     # Capture frames from the RTSP stream
#     def capture_frames(self):
#         cap = cv2.VideoCapture(self.rtsp_url)
#         ret, frame = cap.read()
#         cap.release()
#         if not ret:
#             print("Failed to grab frame")
#             return None
#         frame = cv2.resize(frame, (1280, 720))
#         return frame

#     # Extract face embeddings using FaceNet
#     def extract_embeddings(self):
#         frame_face_arr = self.embedding_queue.get()
#         if frame_face_arr is None:
#             print("Not valid face array!")
#             return None

#         frame_all_embedding = []
#         for face in frame_face_arr:
#             samples = expand_dims(face, axis=0)
#             yhat = self.Face_embedding.embeddings(samples)
#             frame_all_embedding.append(yhat[0])
#         return asarray(frame_all_embedding)

#     # Perform classification and similarity checks on embeddings
#     def recognize_faces(self):
#         frame_all_embedding = self.extract_embeddings()
#         if frame_all_embedding is None:
#             print("Not valid Embedding!")
#             return None

#         for embedding in frame_all_embedding:
#             sample_embedding = expand_dims(embedding, axis=0)
#             predicted_class = self.svc_model.predict(sample_embedding)
#             predicted_label = self.label_encoder.inverse_transform(predicted_class)[0]

#             training_embedding = None
#             for i, label in enumerate(self.data['arr_1']):
#                 if label == predicted_label:
#                     training_embedding = self.data['arr_0'][i]
#                     break

#             if training_embedding is not None:
#                 training_embedding = expand_dims(training_embedding, axis=0)
#                 similarity_score = cosine_similarity(training_embedding, sample_embedding)
#                 similarity_score = round(similarity_score[0][0] * 100, 3)

#                 if similarity_score < 60:
#                     return "Unknown", similarity_score
#                 else:
#                     return predicted_label, similarity_score

#         return "Unknown", 0  # If no faces were recognized

#     # Main function to start the processes
#     def main(self):
#         detection_process = multiprocessing.Process(target=self.detect_faces)
#         recognition_process = multiprocessing.Process(target=self.recognize_faces)

#         # Start face detection and recognition processes
#         detection_process.start()
#         recognition_process.start()

#         # Join processes to ensure they complete before exiting
#         detection_process.join()
#         recognition_process.join()


# if __name__ == "__main__":
#     # Load pre-trained models
#     svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_face_model.pkl")
#     label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_label_encoder.pkl")
#     Face_embedding = FaceNet()
#     detector = MTCNN()
#     data = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\face_embeddings_wth_labels.npz")
#     rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"

#     # Create multiprocessing queues
#     frame_queue = multiprocessing.Queue(maxsize=5)
#     embedding_queue = multiprocessing.Queue(maxsize=5)

#     # Create FRS object and start the process
#     FRS_obj = FRS(detector, Face_embedding, svc_model, label_encoder, data, rtsp_url, frame_queue, embedding_queue)
#     FRS_obj.main()







