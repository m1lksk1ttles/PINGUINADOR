import cv2
from ultralytics import YOLO
import supervision as sv

MODEL_PATH = "best.pt" 
CAMERA_INDEX = 0

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print("LES PRESENTO.... EL PINGUINADOR")

while True:
    ret, frame = cap.read()
    if not ret:
        print("conecta la camara pñts")
        break

    try:
        results = model(frame)[0]
        
        detections = sv.Detections.from_ultralytics(results)

        # Anotar la imagen
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections)

        # Mostrar resultados
        cv2.imshow("YOLOv8 - Offline Detection", frame)

    except Exception as e:
        print(f"EHHH ERRORFATAL: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
