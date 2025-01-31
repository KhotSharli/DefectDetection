from Defect_Detection_Pipeline import predict_defect

prediction, confidence = predict_defect(
    image_path="path/to/image.jpg",
    model_path="defect_detection_resnet50.pth",
    model_arch="resnet50"
)
print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")