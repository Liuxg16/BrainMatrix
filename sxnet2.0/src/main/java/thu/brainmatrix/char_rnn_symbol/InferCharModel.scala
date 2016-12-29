package thu.brainmatrix.char_rnn_symbol

import thu.brainmatrix.Context
import thu.brainmatrix.NDArray
import thu.brainmatrix.Shape
import Config._
class InferCharModel(numLstmLayer: Int, n_alphabet: Int, numHidden: Int,
                           numEmbed: Int, argParams: Map[String, NDArray],
                           ctx: Context = Context.cpu(), dropout: Float = 0f) {
    private val symbol = Lstm.LSTM_forward(numLstmLayer,SEQ_LENGTH, numHidden, numEmbed, n_alphabet, DROPOUT)
    private val batchSize = 1
    val initC = (for (l <- 0 until LSTM_N_LAYER)
                          yield (s"_l${l}_init_c" -> Shape(batchSize, DIM_HIDDEN))).toMap
    val initH = (for (l <- 0 until LSTM_N_LAYER)
                          yield (s"_l${l}_init_h" -> Shape(batchSize, DIM_HIDDEN))).toMap
    val dataShape = Map("data" -> Shape(batchSize,n_alphabet),"label" -> Shape(batchSize,1))
    private val inputShape = initC ++ initH ++ dataShape
    private val executor = symbol.simpleBind(ctx = ctx, shapeDict = inputShape)

    for (key <- this.executor.argDict.keys) {
      if (!inputShape.contains(key) && argParams.contains(key) && key != "softmax_label") {
        argParams(key).copyTo(this.executor.argDict(key))
      }
    }

    private var stateName = (Array[String]() /: (0 until numLstmLayer)) { (acc, i) =>
      acc :+ s"l${i}_init_c"  :+ s"l${i}_init_h"
    }

    private val statesDict = stateName.zip(this.executor.outputs.drop(1)).toMap
    private val inputArr = NDArray.zeros(dataShape("data"))

    def forward(inputData: NDArray, newSeq: Boolean = false): Array[Float] = {
      if (newSeq == true) {
        for (key <- this.statesDict.keys) {
          this.executor.argDict(key).set(0f)
        }
      }
      inputData.copyTo(this.executor.argDict("data"))
      this.executor.forward()
      for (key <- this.statesDict.keys) {
        this.statesDict(key).copyTo(this.executor.argDict(key))
      }
      val prob = this.executor.outputs(0).toArray
      prob
    }
    
   def dispose(){
	   this.executor.dispose()
   } 
   
  }
