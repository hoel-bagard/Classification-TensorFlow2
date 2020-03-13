class DataConfig:
    # Training config
    DATA_PATH = '../data/'
    USE_CHECKPOINT = True                    # Whether to save checkpoints or not
    KEEP_CHECKPOINTS = False                          # Whether to remove the checkpoint dir
    CHECKPOINT_DIR = '../checkpoint/Test/'  # Path to checkpoint dir
    USE_TB = True                          # Whether generate a TensorBoard or not
    KEEP_TB = False                          # Whether to remove the TensorBoard dir
    TB_DIR = '../logs/Test'               # Path to TensorBoard dir
    VAL_FREQ = 1                            # How often to compute validation images and metrics
    RECORD_DELAY = 0                      # Metrics before that epoch are not recorded
