from records import FRS
import joblib
from mtcnn_ort import MTCNN
# from keras_facenet import FaceNet
import multiprocessing
import time
import cv2
from numpy import load

svc_model = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_face_model.pkl")
label_encoder = joblib.load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\model_pickel_files\optimized_label_encoder.pkl")
Face_embedding = FaceNet()
detector = MTCNN()
data = load(r"C:\Users\hp\Downloads\Vinayan_New\Real Time Face Recognition using Kafka\weight_model_files\face_embeddings_wth_labels.npz")
rtsp_url = "rtsp://admin:vinayan@123@192.168.1.64:554/1/1"
 # Create multiprocessing queues
frame_queue = multiprocessing.Queue(maxsize=5)
embedding_queue = multiprocessing.Queue(maxsize=5)
# recognition_queue = multiprocessing.Queue(maxsize=5)

FRS_obj = FRS(detector, Face_embedding, svc_model, label_encoder, data, rtsp_url, frame_queue, embedding_queue)
              
# while True:
#     # detect_faces_obj = FRS.detect_faces()
    
#     frame, frame_face_positions = FRS_obj.detect_faces()
#     cv2.imshow("win", frame)
#     print("*************************")
#     print(frame_face_positions)
#     extract_embeddings_obj = FRS.extract_embeddings()
#     predicted_label, similarity_score = FRS.recognize_faces()
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     print(predicted_label,similarity_score)
#     # time.sleep(5)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()



# if __name__ == "__main__":
#     obj = FRS_obj.main()
#     print(obj)