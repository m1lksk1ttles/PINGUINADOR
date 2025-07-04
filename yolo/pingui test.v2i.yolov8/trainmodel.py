from ultralytics import YOLO

# Carga un modelo base (elige el que tu Raspberry Pi pueda manejar después)
# Puedes usar yolov8n.pt (nano) o yolov8s.pt (small)
model = YOLO("yolov8n.pt")  # o yolov8s.pt si tienes más RAM

# Entrena el modelo usando el dataset descargado de Roboflow
# Asegúrate de que el archivo data.yaml esté en el mismo folder o indica la ruta completa
results = model.train(
    data="data.yaml",     # archivo con rutas y clases
    epochs=150,            # puedes ajustarlo según tu tiempo
    imgsz=416,            # tamaño de imagen, puedes bajarlo a 416 para Raspberry
    batch=32,              # ajusta según tu RAM
    device="cpu",         # usa CPU para evitar errores si no hay GPU
)

# El modelo entrenado quedará en: runs/detect/train/weights/best.pt
print("Entrenamiento terminado. Modelo guardado en: runs/detect/train/weights/best.pt")