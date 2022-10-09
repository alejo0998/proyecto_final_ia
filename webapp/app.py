from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import gc
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Asd"


@app.route("/send_video", methods=["POST"])
def send_video():
    video = request.files.get('video')
    idSign = request.form.get('idSign')
    categorySign = request.form.get('categorySign')
    response = predict(video, idSign, categorySign)
    print("FINALIZANDO")
    if video is None:
        return "No se envio el video"
    str_response = str(response[0].copy)
    del response
    gc.collect
    return str_response
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)



IMAGE_HEIGHT , IMAGE_WIDTH = 320, 180 
SEQUENCE_LENGTH = 30        #ACÁ DECÍA 20, LE SUBÍ A 150 PORQUE LA MAYORÍA LOS FILMAMOS A 30 FPS Y DURAN COMO MUCHO 5 SEG
DATASET_DIR = '../media' 

class Video():

  def __init__(self):
    self.frames = []
    self.keypoints = []
    self.label = ""
    self.path = ""


def mediapipe_detection(image, model):
    #image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #image = image / 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                       # Image is no longer writeable
    results = model.process(image)                      # Make prediction
    image.flags.writeable = True                        # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def zero(i,t):
  print("no encontro " + str(t))
  a = np.empty(0)
  a.resize(i)
  return a

def frames_extraction(video_memory, categoria):
    print("Estoy extrayendo los frames")
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''
    # Declare a list to store video frames.
    #frames_list = []
    results = []
    video = Video()
    keypoints = []
    video.label = categoria
    # Read the Video File using the VideoCapture object.
    nombre_archivo = video_memory.filename
    video_memory.save(dst=nombre_archivo)
    video_reader = cv2.VideoCapture(nombre_archivo)
    os.remove(nombre_archivo)
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    # Iterate through the Video Frames.
    mp_holistic = mp.solutions.holistic # Holistic model
    print("ANTES")
    for frame_counter in range(SEQUENCE_LENGTH):
      #t = threading.Thread(target=magia_2, args=(video_reader, skip_frames_window, frame_counter,  ))
      #t.start()
      video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
      success, frame = video_reader.read()
      if not success:
          return "Error"
      resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
      with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        _, results = mediapipe_detection(resized_frame, holistic)
        del holistic
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else zero(33*4,"pose")
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else zero(468*3,"cara")
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else zero(21*3,"mano izq")
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else zero(21*3,"mano der")
        pose_arr = pose[:33*4].copy()
        face_arr = face[:468*3].copy()
        lh_arr = lh[:21*3].copy()
        rh_arr = rh[:21*3].copy()
        del pose
        del face
        del lh
        del rh
        keypoint = np.concatenate([pose_arr, face_arr, lh_arr, rh_arr])
        keypoint_arr = keypoint[:1662].copy()
        del keypoint
        keypoints.append(keypoint_arr)
        gc.collect()
    del mp_holistic
    gc.collect()
    return keypoints

def predict(video, sena, categoria):
    categoria = categoria
    model_name = categoria
    file_name = './modelos/' + model_name + ".h5"
    model = tf.keras.models.load_model(file_name, compile = False)
    print("Cargue el modelo")
    keypoints = frames_extraction(video, categoria)
    print("Extraje los frames")
    test_keypoints = list(keypoints)
    list_test = list()
    list_test.append(test_keypoints)
    lista = np.array(list_test)
    nueva_lista = lista.copy()
    del keypoints
    del list_test
    del lista
    predictions = model.predict(nueva_lista).copy()
    del nueva_lista
    print("Ya predije")
    del nueva_lista
    gc.collect()
    return predictions