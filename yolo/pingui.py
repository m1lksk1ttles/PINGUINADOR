from inference import get_model
import supervision as sv
import cv2
import os

API_KEY = "qzCvC6cNTjMmxHf43jFn" 
MODEL_ID = "pingui-test/2"  
CAMERA_INDEX = 0  

model = get_model(model_id=MODEL_ID, api_key=API_KEY)

cap = cv2.VideoCapture(CAMERA_INDEX)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print(" CON USTEDES... EL PINGUINADOR")

while True:
    ret, frame = cap.read()
    if not ret:
        print("conecta la camara pñts")
        break

    try:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections)

        cv2.imshow("YOLOv8 - Roboflow", frame)
    except Exception as e:
        print(f"EHHH ERROR FATAL: {e}")

    # salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()