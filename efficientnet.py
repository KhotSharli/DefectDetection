from Defect_Detection_Pipeline import predict_defect

prediction, confidence = predict_defect(
    image_path="path/to/image.jpg",
    model_path="defect_detection_efficientnet.pth",
    model_arch="efficientnet"
)
print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")