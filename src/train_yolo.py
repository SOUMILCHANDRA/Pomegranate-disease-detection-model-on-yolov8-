from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # Pointing to the dataset directory which contains train/val/test splits
    results = model.train(data='e:/SIH2/dataset_merged', epochs=10, imgsz=224, batch=16, project='pomegranate_disease_model', name='yolov8n_cls_run_v2')

    # Validate
    metrics = model.val()
    print("Validation Metrics:", metrics)

    # Export the model
    success = model.export(format='onnx')
    print("Model exported to ONNX:", success)

if __name__ == '__main__':
    train_model()
