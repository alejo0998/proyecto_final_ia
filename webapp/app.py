import multiprocessing
from flask import Flask, request
import gc
import os
import dask.bag as db

semaforo = multiprocessing.Semaphore(1); #Crear variable semáforo
semaforo_2 = multiprocessing.Semaphore(1); #Crear variable semáforo

app = Flask(__name__)

@app.route("/")
def home():
    return "Asd"

IMAGE_HEIGHT , IMAGE_WIDTH = 320, 180 
SEQUENCE_LENGTH = 30
DATASET_DIR = '../media' 

def return_in_queue(queue, func, it):
    queue.put(func(it))

def run_in_subprocess(it, func):    
    semaforo.acquire()
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=return_in_queue, args=(queue, func, it))
    process.start()
    ret = queue.get()
    process.join()
    semaforo.release()

    return ret

def mediapipe_detection(image, model):
    import cv2
    #image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #image = image / 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                       # Image is no longer writeable
    results = model.process(image)                      # Make prediction
    image.flags.writeable = True                        # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results



def zero(i,t):
    import numpy as np
    print("no encontro " + str(t))
    return np.zeros(i)

def frames_extraction(nombre_archivo):
    import numpy as np
    import mediapipe as mp
    import cv2
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

    keypoints = list()
    # Read the Video File using the VideoCapture object.

    video_reader = cv2.VideoCapture(nombre_archivo)
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    # Iterate through the Video Frames.
    mp_holistic = mp.solutions.holistic # Holistic model
    print("ANTES")
    for frame_counter in range(SEQUENCE_LENGTH):
      print(frame_counter)
      video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
      success, frame = video_reader.read()
      if not success:
          return "Error"
      with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        _, results = mediapipe_detection(cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)), holistic)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else zero(33*4,"pose")
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else zero(468*3,"cara")
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else zero(21*3,"mano izq")
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else zero(21*3,"mano der")
        keypoints.append(np.concatenate([pose, face, lh, rh]))
        print (keypoints)
    video_reader.release()
    return keypoints
    
def frames_extraction_web(nombre_archivo):
    import numpy as np
    import mediapipe as mp
    import cv2
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

    keypoints = list()
    # Read the Video File using the VideoCapture object.

    video_reader = cv2.VideoCapture(nombre_archivo)
    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames_count = 60
    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    # Iterate through the Video Frames.
    mp_holistic = mp.solutions.holistic # Holistic model
    pos = 1
    print("ANTES")
    for frame_counter in range(SEQUENCE_LENGTH):
      print(frame_counter)
      video_reader.set(120, frame_counter*2 )
      pos = pos + frame_counter*2
      success, frame = video_reader.read()
      if not success:
          return "Error"
      with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        _, results = mediapipe_detection(cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)), holistic)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else zero(33*4,"pose")
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else zero(468*3,"cara")
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else zero(21*3,"mano izq")
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else zero(21*3,"mano der")
        keypoints.append(np.concatenate([pose, face, lh, rh]))
        print (keypoints)
    video_reader.release()
    return keypoints


@app.route("/send_video", methods=["POST"])
def send_video():
    import tensorflow as tf
    import numpy as np
    semaforo_2.acquire()
    video = request.files.get('video')
    position = request.form.get('position')
    category = request.form.get('category')
    nombre_archivo = video.filename
    video.save(dst=nombre_archivo)
    print(nombre_archivo) 
    result = db.from_sequence([nombre_archivo], partition_size=1).map(run_in_subprocess, frames_extraction_web)
    keypoints = result.compute()
    file_name = './modelos/' + str(category) + ".h5"
    predictions = tf.keras.models.load_model(file_name, compile = True).predict(np.asarray(keypoints))
    print("FINALIZANDO")
    os.remove(nombre_archivo)
    del result
    del keypoints
    gc.collect()
    if video is None:
        return "No se envio el video"
    semaforo_2.release()    
    max = np.argmax(predictions[0])
    return str(max == int(position))
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


