from inference import get_model
import supervision as sv
import cv2

# define the image path to use for inference
image_file = "TESTFOTO.jpg"
image = cv2.imread(image_file)

# load a pre-trained model with CPUExecutionProvider only
model = get_model(
    model_id="pingui-test/2",
    api_key="qzCvC6cNTjMmxHf43jFn"
)


# run inference on the image
results = model.infer(image)[0]

# parse results into Supervision Detections
detections = sv.Detections.from_inference(results)

# annotate the image
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# show result
sv.plot_image(annotated_image)
