class ModelConfig:
    # Training parameters
    BATCH_SIZE = 64   # Batch size
    MAX_EPOCHS = 450  # Number of Epochs
    LR = 1e-3         # Learning Rate
    IMG_SIZE = 28     # If not loading from pickles, then the input images are resized to this size

    NETWORK_NAME = "CNN"   # Either CNN, SmallMobileNet or MobileNetV2
