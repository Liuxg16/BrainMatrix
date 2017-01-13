package thu.brainmatrix.rnn

import thu.brainmatrix.Symbol
import thu.brainmatrix.Executor

/**
 * @author Depeng Liang
 */
object Lstm {

  final case class LSTMState(c: Symbol, h: Symbol)
  final case class LSTMParam(i2hWeight: Symbol, i2hBias: Symbol,
                             h2hWeight: Symbol, h2hBias: Symbol)

  // LSTM Cell symbol
  def lstm(numHidden: Int, inData: Symbol, prevState: LSTMState,
           param: LSTMParam, seqIdx: Int, layerIdx: Int, dropout: Float = 0f): LSTMState = {
    val inDataa = {
      if (dropout > 0f) Symbol.Dropout()(Map("data" -> inData, "p" -> dropout))
      else inData
    }
    val i2h = Symbol.FullyConnected(s"t${seqIdx}_l${layerIdx}_i2h")(Map("data" -> inDataa,
                                                       "weight" -> param.i2hWeight,
                                                       "bias" -> param.i2hBias,
                                                       "num_hidden" -> numHidden * 4))
    val h2h = Symbol.FullyConnected(s"t${seqIdx}_l${layerIdx}_h2h")(Map("data" -> prevState.h,
                                                       "weight" -> param.h2hWeight,
                                                       "bias" -> param.h2hBias,
                                                       "num_hidden" -> numHidden * 4))
    val gates = i2h + h2h
    val sliceGates = Symbol.SliceChannel(s"t${seqIdx}_l${layerIdx}_slice")(Array(gates),
        Map("num_outputs" -> 4))
    val ingate = Symbol.Activation()(Map("data" -> sliceGates.get(0), "act_type" -> "sigmoid"))
    val inTransform = Symbol.Activation()(Map("data" -> sliceGates.get(1), "act_type" -> "tanh"))
    val forgetGate = Symbol.Activation()(Map("data" -> sliceGates.get(2), "act_type" -> "sigmoid"))
    val outGate = Symbol.Activation()(Map("data" -> sliceGates.get(3), "act_type" -> "sigmoid"))
    val nextC = (forgetGate * prevState.c) + (ingate * inTransform)
    val nextH = outGate * Symbol.Activation()(Map("data" -> nextC, "act_type" -> "tanh"))
    LSTMState(c = nextC, h = nextH)
  }

  // we define a new unrolling function here because the original
  // one in lstm.py concats all the labels at the last layer together,
  // making the mini-batch size of the label different from the data.
  // I think the existing data-parallelization code need some modification
  // to allow this situation to work properly
  def lstmUnroll(numLstmLayer: Int, seqLen: Int, inputSize: Int, numHidden: Int,
                 numEmbed: Int, numLabel: Int, dropout: Float = 0f): Symbol = {
    val embedWeight = Symbol.Variable("embed_weight")
    val clsWeight = Symbol.Variable("cls_weight")
    val clsBias = Symbol.Variable("cls_bias")

    var paramCells = Array[LSTMParam]()
    var lastStates = Array[LSTMState]()
    for (i <- 0 until numLstmLayer) {
      paramCells = paramCells :+ LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
                                           i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
                                           h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
                                           h2hBias = Symbol.Variable(s"l${i}_h2h_bias"))
      lastStates = lastStates :+ LSTMState(c = Symbol.Variable(s"l${i}_init_c"),
                                           h = Symbol.Variable(s"l${i}_init_h"))
    }
    assert(lastStates.length == numLstmLayer)

    // embeding layer
    val data = Symbol.Variable("data")
    var label = Symbol.Variable("softmax_label")
    val embed = Symbol.Embedding("embed")(Map("data" -> data, "input_dim" -> inputSize,
                                           "weight" -> embedWeight, "output_dim" -> numEmbed))
    val wordvec = Symbol.SliceChannel()(Array(embed),
      Map("num_outputs" -> seqLen, "squeeze_axis" -> true))

    var hiddenAll = Array[Symbol]()
    var dpRatio = 0f
    var hidden: Symbol = null
    for (seqIdx <- 0 until seqLen) {
      hidden = wordvec.get(seqIdx)
      // stack LSTM
      for (i <- 0 until numLstmLayer) {
        if (i == 0) dpRatio = 0f else dpRatio = dropout
        val nextState = lstm(numHidden, inData = hidden,
                             prevState = lastStates(i),
                             param = paramCells(i),
                             seqIdx = seqIdx, layerIdx = i, dropout = dpRatio)
        hidden = nextState.h
        lastStates(i) = nextState
      }
      // decoder
      if (dropout > 0f) hidden = Symbol.Dropout()(Map("data" -> hidden, "p" -> dropout))
      hiddenAll = hiddenAll :+ hidden
    }
    val hiddenConcat = Symbol.Concat()(hiddenAll, Map("dim" -> 0))
    val pred = Symbol.FullyConnected("pred")(Map("data" -> hiddenConcat, "num_hidden" -> numLabel,
                                            "weight" -> clsWeight, "bias" -> clsBias))
    label = Symbol.transpose(label)
    label = Symbol.Reshape()(Map("data" -> label, "target_shape" -> "(0,)"))
//    val sm = Symbol.SoftmaxOutput("softmax")(Map("data" -> pred, "label" -> label))
    
    val smce = Symbol.Softmax_cross_entropy(pred, label) + Symbol.sum(Symbol.square(embedWeight))
    val loss = Symbol.MakeLoss("lossdd")(Map("data"->smce)) 
    loss
//    Symbol.Group(loss,sm)
//    sm
  }

  def lstmInferenceSymbol(numLstmLayer: Int, inputSize: Int, numHidden: Int,
                          numEmbed: Int, numLabel: Int, dropout: Float = 0f): Symbol = {
    val seqIdx = 0
    val embedWeight = Symbol.Variable("embed_weight")
    val clsWeight = Symbol.Variable("cls_weight")
    val clsBias = Symbol.Variable("cls_bias")

    var paramCells = Array[LSTMParam]()
    var lastStates = Array[LSTMState]()
    for (i <- 0 until numLstmLayer) {
      paramCells = paramCells :+ LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
                                           i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
                                           h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
                                           h2hBias = Symbol.Variable(s"l${i}_h2h_bias"))
      lastStates = lastStates :+ LSTMState(c = Symbol.Variable(s"l${i}_init_c"),
                                           h = Symbol.Variable(s"l${i}_init_h"))
    }
    assert(lastStates.length == numLstmLayer)

    val data = Symbol.Variable("data")

    var hidden = Symbol.Embedding("embed")(Map("data" -> data, "input_dim" -> inputSize,
                                           "weight" -> embedWeight, "output_dim" -> numEmbed))

    var dpRatio = 0f
    // stack LSTM
    for (i <- 0 until numLstmLayer) {
      if (i == 0) dpRatio = 0f else dpRatio = dropout
      val nextState = lstm(numHidden, inData = hidden,
                           prevState = lastStates(i),
                           param = paramCells(i),
                           seqIdx = seqIdx, layerIdx = i, dropout = dpRatio)
      hidden = nextState.h
      lastStates(i) = nextState
    }
    // decoder
    if (dropout > 0f) hidden = Symbol.Dropout()(Map("data" -> hidden, "p" -> dropout))
    val fc = Symbol.FullyConnected("pred")(Map("data" -> hidden, "num_hidden" -> numLabel,
                                      "weight" -> clsWeight, "bias" -> clsBias))
    val sm = Symbol.SoftmaxOutput("softmax")(Map("data" -> fc))
    var output = Array(sm)
    for (state <- lastStates) {
      output = output :+ state.c
      output = output :+ state.h
    }
    Symbol.Group(output: _*)
  }
  
  // we define a new unrolling function here because the original
  // one in lstm.py concats all the labels at the last layer together,
  // making the mini-batch size of the label different from the data.
  // I think the existing data-parallelization code need some modification
  // to allow this situation to work properly
  def bi_lstmUnroll(numLstmLayer: Int, seqLen: Int, inputSize: Int, numHidden: Int,
                 numEmbed: Int, numLabel: Int, dropout: Float = 0f): Symbol = {
	    val embedWeight = Symbol.Variable("embed_weight")
	    val clsWeight = Symbol.Variable("cls_weight")
	    val clsBias = Symbol.Variable("cls_bias")
	
	    var lastStates = Array[LSTMState]()
	    for (i <- 0 until numLstmLayer) {
	      lastStates = lastStates :+ LSTMState(c = Symbol.Variable(s"l${i}_init_c"),
	                                           h = Symbol.Variable(s"l${i}_init_h"))
	    }
	    var i = 0
	    val forward_paramCells = LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
	                                           i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
	                                           h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
	                                           h2hBias = Symbol.Variable(s"l${i}_h2h_bias"))
	    i = 1
	    val backward_paramCells = LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
	                                           i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
	                                           h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
	                                           h2hBias = Symbol.Variable(s"l${i}_h2h_bias"))
	    
	    
	    assert(lastStates.length == numLstmLayer)
	
	    // embeding layer
	    val data = Symbol.Variable("data")
	    var label = Symbol.Variable("softmax_label")
	    val embed = Symbol.Embedding("embed")(Map("data" -> data, "input_dim" -> inputSize,
	                                           "weight" -> embedWeight, "output_dim" -> numEmbed))
	    val wordvec = Symbol.SliceChannel()(Array(embed),
	      Map("num_outputs" -> seqLen, "squeeze_axis" -> true))

 		var forward_hidden =  Array[Symbol]()
    	var hidden: Symbol = null
	    for (seqIdx <- 0 until seqLen) {
	      	hidden = wordvec.get(seqIdx)
	        val nextState = lstm(numHidden, inData = hidden,
	                             prevState = lastStates(0),
	                             param = forward_paramCells,
	                             seqIdx = seqIdx, layerIdx = 0, dropout = dropout)
	        hidden = nextState.h
	        lastStates(0) = nextState
	        forward_hidden :+= hidden
	    }
	    
	    //backward
	    var backward_hidden =  Array[Symbol]()
	    for (seqIdx <- 0 until seqLen) {
	    	val k = seqLen - seqIdx - 1
	      	hidden = wordvec.get(k)
	        val nextState = lstm(numHidden, inData = hidden,
	                             prevState = lastStates(1),
	                             param = backward_paramCells,
	                             seqIdx = seqIdx, layerIdx = 1, dropout = dropout)
	        hidden = nextState.h
	        lastStates(1) = nextState
	        backward_hidden = hidden +:backward_hidden
	    }

	    var hiddenAll = Array[Symbol]()
	    for (seqIdx <- 0 until seqLen) {
	    	hiddenAll :+= Symbol.Concat()(Array(forward_hidden(seqIdx),backward_hidden(seqIdx)), Map("dim"->1))
 	    }
	    val hiddenConcat = Symbol.Concat()(hiddenAll, Map("dim" -> 0))
    	val pred = Symbol.FullyConnected("pred")(Map("data" -> hiddenConcat, "num_hidden" -> numLabel,
                                            "weight" -> clsWeight, "bias" -> clsBias))
    	label = Symbol.transpose(label)
    	label = Symbol.Reshape()(Map("data" -> label, "target_shape" -> "(0,)"))
    	val sm = Symbol.SoftmaxOutput("softmax")(Map("data" -> pred, "label" -> label))
    	sm
  }
  
  
  
  
}
