package thu.brainmatrix.char_rnn_symbol
import thu.brainmatrix.NDArray
import thu.brainmatrix.Base
//import thu.brainmatrix.ReshapeAccuracy
import thu.brainmatrix.Shape
import thu.brainmatrix.Accuracy
import thu.brainmatrix.Context
import thu.brainmatrix.FeedForward
import thu.brainmatrix.optimizer.Adam
import thu.brainmatrix.Xavier
import thu.brainmatrix.Symbol
import thu.brainmatrix.Model
import thu.brainmatrix.Callback
import thu.brainmatrix.util.mathTool
import thu.brainmatrix.CustomMetric
import thu.brainmatrix.io.NDArrayLSTMIter
import thu.brainmatrix.EpochEndCallback
import Config._
import scala.io.Source
import scala.collection.mutable.ListBuffer


object Training {
    
    def main(args:Array[String]){
//        sampleCharLSTM
//        trainCharLSTM
//        train_vec_CharLSTM
//    	trainCharRNN
    	train_vec_CharLSTM_lxg
    }
    
    def trainCharLSTM{
    	
      val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
      val n_alphabet = vocab.size
      val lstm = Lstm.LSTMNet(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
//      lstm.listArguments().foreach {println}
//      println(lstm.debug())
      val source  = Source.fromFile(INPUT_FILE_NAME) 
      val seq_input = source.mkString
      val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
      val text_train = seq_input.take(len_train)
      val text_val = seq_input.drop(len_train)
      val traindata = seq_IO.lstmDataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      val valdata = seq_IO.lstmDataIter(text = text_val,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      Base.INPUTSHAPE_AUXILIARY = Map("_l0_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l0_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN))
//      val modelBase = new FeedForward(lstm, Context.cpu(), numEpoch = N_EPOCH,optimizer = new SGD(learningRate = LEARNING_RATE, momentum = MOMENTUM, wd = WEIGHT_DECAY),name = "lstm")
////        modelBase.fit(traindata, traindata,new ReconsAccuracy())
//       modelBase.fit(traindata,valdata,new Accuracy())
//       modelBase.saveModelParams(s"./model/charLSTM.params_${N_EPOCH}")
      
    }
    
    def train_vec_CharLSTM{
    	
      val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
      val n_alphabet = vocab.size
      val lstm = Lstm.LSTMNet(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
//      lstm.listArguments().foreach {println}
//      println(lstm.debug())
      val source  = Source.fromFile(INPUT_FILE_NAME) 
      val seq_input = source.mkString
      val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
      val text_train = seq_input.take(len_train)
      val text_val = seq_input.drop(len_train)
      val traindata = seq_IO.lstm_vec_DataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH,vocab_len = n_alphabet)
      val valdata = seq_IO.lstm_vec_DataIter(text = text_val,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH,vocab_len = n_alphabet)
      
//      for(j<-0 until 80){
//	      val databatch = traindata.next()
//	      val label1 = databatch.label(0)
//      	  println(label1)
//      }
      Base.INPUTSHAPE_AUXILIARY = Map("_l0_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l0_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN))
//      val modelBase = new FeedForward(lstm, Context.cpu(), numEpoch = N_EPOCH,optimizer = new SGD(learningRate = LEARNING_RATE, momentum = MOMENTUM, wd = WEIGHT_DECAY),name = "lstm")
//       modelBase.fit(traindata,valdata,new Accuracy())
//       modelBase.saveModelParams(s"./model/charLSTM.params_${N_EPOCH}")
      
    }
    
    
    
    def train_vec_CharLSTM_lxg{
      val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME) 
      val n_alphabet = vocab.size
      val lstm = Lstm.LSTM(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
//      lstm.listArguments().foreach {println}
//      println(lstm.debug())
      val source  = Source.fromFile(INPUT_FILE_NAME) 
      val seq_input = source.mkString
      val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
      val text_train = seq_input.take(len_train)
      val text_val = seq_input.drop(len_train)
      val traindata = seq_IO.RNN_OneHot_DataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      val valdata = seq_IO.RNN_OneHot_DataIter(text = text_val,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      val  h = (0 until LSTM_N_LAYER).map(idx =>
    	  (s"_l${idx}_init_h",Shape(BATCH_SIZE,DIM_HIDDEN))
    	  ).toMap
   	  val  c = (0 until LSTM_N_LAYER).map(idx =>
    	  (s"_l${idx}_init_c",Shape(BATCH_SIZE,DIM_HIDDEN))
    	  ).toMap
      
      val ctx = if (N_GPU == -1) Context.cpu() else Context.gpu(N_GPU)
    	  
   	  val datasAndLabels = traindata.provideData ++ traindata.provideLabel ++ h ++ c
      val (argShapes, outputShapes, auxShapes) = lstm.inferShape(datasAndLabels)

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

      val argNames = lstm.listArguments()
      val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
      val auxNames = lstm.listAuxiliaryStates()
      val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap

      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabels.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap

      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabels.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val data = argDict("data")
      val label = argDict("label")

      val executor = lstm.bind(ctx, argDict, gradDict)

      val opt = new Adam(learningRate = LEARNING_RATE, wd = 0.0001f)

      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new CustomMetric(mathTool.perplexity, "perplexity")
      val batchEndCallback = new Callback.Speedometer(BATCH_SIZE, 50)
      val epochEndCallback = doCheckpoint("./model/obama")

      for (epoch <- 0 until N_EPOCH) {
        // Training phase
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false
        // Iterate over training data.
        traindata.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && traindata.hasNext) {
            val dataBatch = traindata.next()

            data.set(dataBatch.data(0))
            label.set(dataBatch.label(0))
            executor.forward(isTrain = true)
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }

            // evaluate at end, so out_cpu_array can lazy copy
            evalMetric.update(dataBatch.label, executor.outputs)

            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            traindata.reset()
          }
          // this epoch is done
          epochDone = true
        }
        val (name, value) = evalMetric.get
        println(s"Epoch[$epoch] Train-$name=$value")
        val toc = System.currentTimeMillis
        println(s"Epoch[$epoch] Time cost=${toc - tic}")

        epochEndCallback.invoke(epoch, lstm, argDict, auxDict)
      }
     
      executor.dispose()
      
    }
    
    
    def trainCharRNN{
    	
      val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
      val n_alphabet = vocab.size
      val lstm = Lstm.LSTM(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
//      lstm.listArguments().foreach {println}
//      println(lstm.debug())
      val source  = Source.fromFile(INPUT_FILE_NAME) 
      val seq_input = source.mkString
      val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
      val text_train = seq_input.take(len_train)
      val text_val = seq_input.drop(len_train)
      val traindata = seq_IO.RNN_OneHot_DataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      val valdata = seq_IO.RNN_OneHot_DataIter(text = text_val,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      val  h = (0 until LSTM_N_LAYER).map(idx =>{
    	  (s"_l${idx}_init_h",Shape(BATCH_SIZE,DIM_HIDDEN))
    	  }).toMap
   	  val  c = (0 until LSTM_N_LAYER).map(idx =>{
    	  (s"_l${idx}_init_c",Shape(BATCH_SIZE,DIM_HIDDEN))
    	  }).toMap
      
      Base.INPUTSHAPE_AUXILIARY = h ++ c
//      val modelBase = new FeedForward(lstm, Context.defaultCtx, numEpoch = N_EPOCH,optimizer = new SGD(learningRate = LEARNING_RATE, momentum = MOMENTUM, wd = WEIGHT_DECAY),name = "lstm")
////        modelBase.fit(traindata, traindata,new ReconsAccuracy())
//       modelBase.fit(traindata,valdata,new ReshapeAccuracy())
//       modelBase.saveModelParams(s"./model/charLSTM.params_${N_EPOCH}")
      
    };
    
    def doCheckpoint(prefix: String): EpochEndCallback = new EpochEndCallback {
    override def invoke(epoch: Int, symbol: Symbol,
                        argParams: Map[String, NDArray],
                        auxStates: Map[String, NDArray]): Unit = {
      Model.saveCheckpoint(prefix, epoch + 1, symbol, argParams, auxStates)
    }
  }
}