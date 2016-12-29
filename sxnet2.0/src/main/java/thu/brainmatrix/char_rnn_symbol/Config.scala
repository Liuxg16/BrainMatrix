package thu.brainmatrix.char_rnn_symbol


/**
 * @author liuxianggen
 * @date 20160718
 * @brief provide some global setting for charr_rnn
 * @param
 * @return
 * @example
 * @note
 */
object Config {
    
    val INPUT_FILE_NAME = "./seqData/input.txt"
//    val INPUT_FILE_NAME = "./seqData/ptb.train.txt"
    val VOCAB_FILE_NAME = "./seqData/vocab.txt"
    val SEQ_LENGTH    = 32
    val UNKNOW_CHAR = '\0'
    val DROPOUT       = 0
    val BATCH_SIZE    = 32
    val DIM_HIDDEN    = 64
    val DIM_EMBED     = 64
    val LSTM_N_LAYER  = 3
    val N_EPOCH       = 2 // 21
    val LEARNING_RATE = 0.001f
    val MOMENTUM      = 0f
    val WEIGHT_DECAY  = 0.000001f
    val CLIP_GRADIENT = 1
    val N_GPU         = 0
    val USE_GPU       = true
    val DATA_TRAIN_RATIO = 0.9
}