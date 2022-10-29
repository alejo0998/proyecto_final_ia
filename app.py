import multiprocessing
from flask import Flask, request
import gc
import os
import dask.bag as db
from flask_cors import CORS


#semaforo = multiprocessing.Semaphore(1); #Crear variable semáforo
#semaforo_2 = multiprocessing.Semaphore(1); #Crear variable semáforo

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Asd"

IMAGE_HEIGHT , IMAGE_WIDTH = 1280, 720 
SEQUENCE_LENGTH = 30
DATASET_DIR = '../media' 


def return_in_queue(queue, func, it):
    queue.put(func(it))

def run_in_subprocess(it, func):    
    #semaforo.acquire()
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=return_in_queue, args=(queue, func, it))
    process.start()
    ret = queue.get()
    process.join()
    #semaforo.release()

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
    errores = []
    errores.append(0)
    errores.append(0)
    errores.append(0)
    errores.append(0)
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
        if results.pose_landmarks is None:
            errores[0]+=1
        if results.face_landmarks is None:
            errores[1]+=1
        if results.left_hand_landmarks is None:
            errores[2]+=1
        if results.right_hand_landmarks is None:
            errores[3]+=1
        keypoints.append(np.concatenate([pose, face, lh, rh]))
        print (keypoints)
    video_reader.release()
    keypoints.append(errores)
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
    errores = []
    errores.append(0)
    errores.append(0)
    errores.append(0)
    errores.append(0)
    keypoints = list()
    # Read the Video File using the VideoCapture object.

    video_reader = cv2.VideoCapture(nombre_archivo)
    # Get the total number of frames in the video.
    #video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    #video_frames_count = 60
    # Calculate the the interval after which frames will be added to the list.
    #skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    # Iterate through the Video Frames.
    mp_holistic = mp.solutions.holistic # Holistic model
    pos = 1
    print("ANTES")
    for frame_counter in range(SEQUENCE_LENGTH):
      print(frame_counter)
      success, frame = video_reader.read()
      if not success:
          return "Error"
      with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        _, results = mediapipe_detection(cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)), holistic)
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else zero(33*4,"pose")
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else zero(468*3,"cara")
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else zero(21*3,"mano izq")
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else zero(21*3,"mano der")
        if results.pose_landmarks is None:
            errores[0]+=1
        if results.face_landmarks is None:
            errores[1]+=1
        if results.left_hand_landmarks is None:
            errores[2]+=1
        if results.right_hand_landmarks is None:
            errores[3]+=1
        keypoints.append(np.concatenate([pose, face, lh, rh]))
        print (keypoints)
    video_reader.release()
    keypoints.append(errores)
    return keypoints


@app.route("/send_video", methods=["POST"])
def send_video():
    import tensorflow as tf
    import numpy as np
    #semaforo_2.acquire()
    video = request.files.get('video')
    position = request.form.get('position')
    category = request.form.get('category')
    web = request.form.get('web')
    nombre_archivo = video.filename
    video.save(dst=nombre_archivo)
    print(nombre_archivo)
    result = None
    if video is None:
        return "No se envio el video"
    if web == 'true': 
        result = db.from_sequence([nombre_archivo], partition_size=1).map(run_in_subprocess, frames_extraction_web)
    else:
        result = db.from_sequence([nombre_archivo], partition_size=1).map(run_in_subprocess, frames_extraction)
    respuesta = result.compute()
    file_name = './modelos/' + str(category) + ".h5"
    cantidad_errores = None
    if len(respuesta[0])==31:
        cantidad_errores = respuesta[0].pop(30)
    predictions = tf.keras.models.load_model(file_name, compile = True).predict(np.asarray(respuesta))
    print("FINALIZANDO")
    os.remove(nombre_archivo)
    message = ''
    i = 0
    if cantidad_errores:
        for cantidad in cantidad_errores:
            if cantidad>15:
                if i == 0:
                    message += "No se pudo detectar la pose \n"
                elif i ==1:
                    message += "No se pudo detectar la cara \n"
                elif i == 2:
                    message += "No se pudo detectar la mano izquierda \n"
                elif i == 3:
                    message += "No se pudo detectar la mano derecha \n"
            i+=1
    if message != '':
        respuesta = {
        'response': message,
        'validation': 'REINTENTAR',
        'prediction': str(predictions)
        }
        print(respuesta)
        return respuesta
    #semaforo_2.release()    
    max = np.argmax(predictions[0])
    booleano = (max == int(position))
    respuesta = {
        'response': "La seña realizada es correcta" if booleano else "La seña realizada es incorrecta",
        'validation': 'CORRECTA' if booleano else 'INCORRECTA',
        'prediction': str(predictions)
    }
    print(respuesta)
    return respuesta
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


