import argparse # para leer argumentos de la linea de comandos
import datetime # para obtener la fecha y hora actual del sistema
import os # para el manejo de llamadas al SO
import cv2 # para opencv
import time # para controlar el tiempo
import shutil # para ver espacio libre del disco 


# función para comprobar el tamaño libre del sistema en bytes 
def check_free_space_in_bytes(path="."):
  # el tamaño total y usado no nos interesa, sólo nos quedamos con el tamaño libre
  _, _, free = shutil.disk_usage(path)
  return free

def check_enough_space_in_bytes(path, threshold):
  # obtenemos el tamaño libre
  free_space = check_free_space_in_bytes(path)
  
  #print(free_space)
  #print(free_space/(1024**3))

  # si el tamaño es mayor, podemos grabar, si es menor, no podemos grabar
  if (free_space >= threshold):
    return True
  else:
    return False

# ejemplo de uso: python3 recordingCrontabVideo.py -duration 60 -frameWidth 1280 -frameHeigth 720 -fps 30 -focusValue 40 -path /home/beesound/videoRecording/videos-piqueras
# argumentos pasados en la llamada al programa
# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-duration", type=int, required=True)
parser.add_argument("-frameWidth", type=int, required=True)
parser.add_argument("-frameHeigth", type=int, required=True)
parser.add_argument("-focusValue", type=int, required=False)
parser.add_argument("-fps", type=int, required=True)
parser.add_argument("-path", type=str, required=False)


# Establece umask a 0, para que los nuevos archivos hereden permisos abiertos
os.umask(0)

# Read arguments from command line
args = parser.parse_args()

with open("/tmp/debug_camara.log", "a") as f:
    f.write(f"Script iniciado a las {datetime.datetime.now()}\n")

#Abrir cámara con el backend de Linux
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# se intenta abrir la cámara, en caso contrario se sale del programa
if (not cap.isOpened()):
#  if (args.verbose):
#    print("Error: No se pudo acceder a la cámara.")
    exit()

# desperdiciamos 20 fotogramas para asegurar que se establecen los parametros adecuados
for _ in range(20):
  cap.read()


if (not args.focusValue): # Comprobamos si se manda el argumento del ajuste foc>
  # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
  os.system("v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=1")

else: # En caso de que no se mande, se establece como autoenfoque y en caso con>
  # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
  # cap.set(cv2.CAP_PROP_FOCUS, args.focusValue)
  os.system("v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0")

  for _ in range(3):
    os.system(f"v4l2-ctl -d /dev/video0 -c focus_absolute={args.focusValue}")
    time.sleep(0.1)

os.system("v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0")
#os.system("v4l2-ctl -d /dev/video0 -c auto_exposure=1") # 1 = Manual
#os.system("v4l2-ctl -d /dev/video0 -c exposure_time_absolute=156") # Clave par>
os.system("v4l2-ctl -d /dev/video0 -c gain=100")
#os.system("v4l2-ctl -d /dev/video0 -c white_balance_automatic=0")

# Establecer los parametros de la camara
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frameHeigth)
cap.set(cv2.CAP_PROP_FPS, args.fps)
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

# desperdiciamos 20 fotogramas para asegurar que se establecen los parámetros adecuados
for _ in range(20):
  cap.read()


try:
  while True: # BUCLE INFINITO

    # obtener dia, mes y anio
    first_time = datetime.datetime.now()

    # generar nombre directorio anio-mes-dia
    if not args.path:
      path = str(first_time.year) + "-" + "{:02d}".format(first_time.month) + "-" + "{:02d}".format(first_time.day)
    else:
      path = args.path + "/" + str(first_time.year) + "-" + "{:02d}".format(first_time.month) + "-" + "{:02d}".format(first_time.day)

    # generar carpeta con el dia
    # Comprobar si el directorio existe, en caso contrario crearlo
    if not os.path.exists(path):
      os.makedirs(path)
      os.chmod(path, 0o777)

    now = datetime.datetime.now()
    dia_path = os.path.join(args.path, now.strftime("%Y-%m-%d"))

    if not os.path.exists(dia_path):
      os.makedirs(dia_path)
      os.chmod(dia_path, 0o777)

    # Generación del nombre del fichero
    file_name = path + "/" + now.strftime("%Y-%m-%d") + "_" + first_time.strftime("%H-%M-%S") + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # O el que prefieras
    
    #comprobamos que hay espacio libre en el disco. Por seguridad para permitir a otros servicios que hacen uso del disco y que no se bloqueen, la comprobación se hace con un umbral de 5GB. De esta forma si se superan el espacio libre de 5GB, se graba, en caso contrario no.
    if(check_enough_space_in_bytes(args.path, 5*(1024**3))):
      out = cv2.VideoWriter(file_name, fourcc, args.fps, (args.frameWidth, args.frameHeigth))

      # Grabación por frames para asegurar unos determinados fps a los segundos del argumento duration
      frames_totales = args.duration * args.fps

      for _ in range(frames_totales):
        ret, frame = cap.read()
        if not ret:
          break

        out.write(frame)

      out.release() # Cierre del fichero actual y generación del siguiente video
      os.chmod(file_name, 0o777)

finally:
  cap.release()
  cv2.destroyAllWindows()