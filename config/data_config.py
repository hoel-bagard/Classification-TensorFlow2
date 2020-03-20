class DataConfig:
    # Training config
    DATA_PATH = '../data/'
    USE_CHECKPOINT = True                    # Whether to save checkpoints or not
    KEEP_CHECKPOINTS = False                          # Whether to remove the checkpoint dir
    CHECKPT_SAVE_FREQ = 10                   # How often to save checkpoints (if they are better than the previous one)
    CHECKPOINT_DIR = '../checkpoint/CNN/'  # Path to checkpoint dir
    USE_TB = True                          # Whether generate a TensorBoard or not
    KEEP_TB = False                          # Whether to remove the TensorBoard dir
    TB_DIR = '../logs/CNN'               # Path to TensorBoard dir
    VAL_FREQ = 10                            # How often to compute validation images and metrics
    RECORD_DELAY = 10                      # Metrics before that epoch are not recorded
