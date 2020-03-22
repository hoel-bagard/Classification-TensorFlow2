class DataConfig:
    # Training config
    DATA_PATH = '../data/'
    DATASET = 'CatVsDog'
    USE_CHECKPOINT = True                    # Whether to save checkpoints or not
    KEEP_CHECKPOINTS = True                          # Whether to remove the checkpoint dir
    CHECKPT_SAVE_FREQ = 2                   # How often to save checkpoints (if they are better than the previous one)
    CHECKPOINT_DIR = '../checkpoint/'+DATASET  # Path to checkpoint dir
    USE_TB = True                          # Whether generate a TensorBoard or not
    KEEP_TB = True                          # Whether to remove the TensorBoard dir
    TB_DIR = '../logs/'+DATASET               # Path to TensorBoard dir
    VAL_FREQ = 2                            # How often to compute validation images and metrics
    RECORD_DELAY = 2                      # Metrics before that epoch are not recorded
