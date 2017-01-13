
package thu.brainmatrix.char_rnn_symbol

import thu.brainmatrix.char_rnn_symbol.Config._
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import scala.io.Source
import thu.brainmatrix.FeedForward
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context.ctx2Array
import thu.brainmatrix.char_rnn_symbol.seq_IO

class lstmSuite extends FunSuite with BeforeAndAfterAll  {
	
  	test("mlp proccess text data"){
        val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
        val n_alphabet = vocab.size
//      val lstm = Lstm.LSTM(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
        val data = Symbol.CreateVariable("data")
        val label = Symbol.CreateVariable("sp_label")
 		val fc1 = Symbol.FullyConnected()(Map("data" -> data, "name" -> "fc1", "num_hidden" -> 128))
	    val act1 = Symbol.Activation()(Map("data" -> fc1, "name" -> "relu1", "act_type" -> "relu"))
	    val fc2 = Symbol.FullyConnected()(Map("data" -> act1, "name" -> "fc2", "num_hidden" -> 64))
 		val act2 = Symbol.Activation()(Map("data" -> fc2, "name" -> "relu2", "act_type" -> "relu"))
 		val fc3 = Symbol.FullyConnected()(Map("data" -> act2, "name" -> "fc3", "num_hidden" -> 24))
 		val linearRO = Symbol.LinearRegressionOutput()(Map("data"->fc3,"label"->label))
 		
// 		SoftmaxOutput(Map("data" -> fc3, "name" -> "sm"))
// 		println(linearRO.debug())
 		val source  = Source.fromFile(INPUT_FILE_NAME) 
        val seq_input = source.mkString
        val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
        val text_train = seq_input.take(len_train)
        val text_val = seq_input.drop(len_train)
      
        val traindata = seq_IO.Str2Char_NDArrayIterator(text = text_train,labelName = "sp_label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
        val modelBase = new FeedForward(linearRO, Context.cpu(), numEpoch = N_EPOCH,
                optimizer = new SGD(learningRate = LEARNING_RATE, momentum = MOMENTUM, wd = WEIGHT_DECAY))
     
//        modelBase.fit(traindata, traindata,new ReconsAccuracy())
  }
  
  	test("vocab reverse"){
		val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
	  	var bacov = for((k,v)<- vocab) yield (v,k)
	  	bacov = bacov.updated(5, '?')
	  	println(bacov(5))
	  	println(bacov)
  }
  
  	test("data&label"){
	  	val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
	  var bacov = for((k,v)<- vocab) yield (v,k)
  	  bacov = bacov.updated(bacov.size-1, '?')
      val n_alphabet = vocab.size
      val lstm = Lstm.LSTMNet(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
      val source  = Source.fromFile(INPUT_FILE_NAME) 
      val seq_input = source.mkString
      val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
      val text_train = seq_input.take(len_train)
      val text_val = seq_input.drop(len_train)
      val traindata = seq_IO.lstmDataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      for(i<- 0 until 17) traindata.next()
      for(j<-0 until 2){
	      val databatch = traindata.next()
	      val data = databatch.data
	      val label = databatch.label
	      val dataText = data.map(x =>bacov(x(0,0).toInt)).mkString
	      val labelText = label.map(x =>bacov(x(0).toInt)).mkString
	      println("------------------------------")
	      println(dataText)
		  println("------------------------------")
		   println(labelText)
      }
   }
  
  	test("lstm_vec_DataIter"){
  	  val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
	  var bacov = for((k,v)<- vocab) yield (v,k)
  	  bacov = bacov.updated(bacov.size-1, '?')
  	  println(bacov)
      val n_alphabet = vocab.size
      val lstm = Lstm.LSTMNet(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
      val source  = Source.fromFile(INPUT_FILE_NAME) 
      val seq_input = source.mkString
      val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
      val text_train = seq_input.take(len_train)
      val text_val = seq_input.drop(len_train)
      val traindata = seq_IO.lstm_vec_DataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH,vocab_len = n_alphabet)
      
      for(i<- 0 until 19) traindata.next()
      for(j<-0 until 19){
	      val databatch = traindata.next()
	      val data = databatch.data
	      val label = databatch.label
	      
	      val dataText = data.map(x =>{
//	     	  val temp = 
	     	  bacov(NDArray.argmaxChannel(x).toArray(0).toInt)
	      }).mkString
	      println("------------------------------")
	      val labelText = label.map(x =>bacov(x(0).toInt)).mkString
	      
	      println(dataText)
		  println("------------------------------")
		   println(labelText)
      }
    }
  	
  	test("RNN_OneHot_DataIter"){
  	  val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
	  var bacov = for((k,v)<- vocab) yield (v,k)
  	  bacov = bacov.updated(bacov.size-1, '?')
  	  println(bacov)
      val n_alphabet = vocab.size
      val lstm = Lstm.LSTM(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
      val source  = Source.fromFile(INPUT_FILE_NAME) 
      val seq_input = source.mkString
      val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
      val text_train = seq_input.take(len_train)
      val text_val = seq_input.drop(len_train)
      val traindata = seq_IO.RNN_OneHot_DataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      
//      for(i<- 0 until 19) traindata.next()
//      for(j<-0 until 19){
      val databatch = traindata.next()
      val data = databatch.data(0)
      val label = databatch.label(0)
      var data_text = ""
      for(i<-0 until BATCH_SIZE){
    	  	  val seq = NDArray.array(data.slice(i).toArray,Shape(SEQ_LENGTH,n_alphabet))
    	  	  val a = NDArray.argmaxChannel(seq)
    	  	  data_text += a.toArray.map(x => bacov(x.toInt)).foldRight("")(_+_)
      }
      
//	     	  val temp = 
//	     	  bacov(NDArray.argmaxChannel(x).toArray(0).toInt)
//	      }).mkString
	      println("------------------------------------------------------------")
	      val labelText = label.toArray.map(x => bacov(x.toInt)).foldRight("")(_+_)
//	      
	      println(data_text)
		  println("-------------------------------------------------------------")
		   println(labelText)
//      }
      }
  	
  	
  	
  	test("2layer-lstm") {
  		
  		val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
      	val n_alphabet = vocab.size
      	val lstm = Lstm.LSTMNet(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
      	
      	val source  = Source.fromFile(INPUT_FILE_NAME) 
	    val seq_input = source.mkString
	    val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
		val text_train = seq_input.take(len_train)
  		val text_val = seq_input.drop(len_train)
      	val traindata = seq_IO.lstmDataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
      	val aux_input= Map("_l0_init_h"->Shape(16,64),"_l0_init_c"->Shape(16,64),"_l1_init_h"->Shape(16,64),"_l1_init_c"->Shape(16,64)) ++ traindata.provideData ++ traindata.provideLabel
      	//val map_infer = for((x,y)<-aux_input) yield (x,Random.uniform(0f, 0.1f, y))
    	val executor = lstm.simpleBind(ctx = Context.cpu(),gradReq = "write",shapeDict = aux_input)
    	
		executor.forward(true)
		val out0 = executor.outputs(0)
		val out15 = executor.outputs(29)
		val out2 = executor.outputs(SEQ_LENGTH-1)
		println(out0)
	    println(out15)		
	    println("----------------------------------------------")
	    println(out2)
//  		executor.backward()
//  		println("----------------------------------------------")
//	    println(executor.gradArrays(0))
    	println("end...")
    	
  	}
  	
  	
  	test("1 layer-lstm") {
  		
  		val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
      	val n_alphabet = vocab.size
      	val lstm = Lstm.LSTMNet(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
      	lstm.listArguments().foreach {println}
      	println(lstm.debug())
      	val source  = Source.fromFile(INPUT_FILE_NAME) 
	    val seq_input = source.mkString
	    val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
		val text_train = seq_input.take(len_train)
  		val text_val = seq_input.drop(len_train)
      	val traindata = seq_IO.lstm_vec_DataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH,vocab_len = n_alphabet)
      	val aux_input= Map("_l0_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l0_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN)) ++ traindata.provideData ++ traindata.provideLabel
      	//val map_infer = for((x,y)<-aux_input) yield (x,Random.uniform(0f, 0.1f, y))
      	println(aux_input)
    	val executor = lstm.simpleBind(ctx = Context.cpu(),gradReq = "write",shapeDict = aux_input)
    	
		executor.forward(true)
		val out0 = executor.outputs(0)
		val out15 = executor.outputs(29)
		val out2 = executor.outputs(SEQ_LENGTH-1)
		println(out0)
	    println(out15)		
	    println("----------------------------------------------")
	    println(out2)
  		executor.backward()
  		println("----------------------------------------------")
//	    (executor.gradArrays).foreach {println}
    	println("end...")
    	
  	}
  	
  
  	test("inspect file"){
  		val source  = Source.fromFile(INPUT_FILE_NAME) 
  		val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
	    val seq_input = source.mkString.map(vocab)
	    println(seq_input.take(100))
  	}
  	
  	
  	test("check params"){
  		val pretrained = NDArray.load2Map(s"./model/charLSTM.params_${N_EPOCH}")
    	println(pretrained.keys)
    	println(pretrained("argParams::_pred_0_weight"))
  	}
  	
  	 test("debugTraining"){
			
	  	val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
      	val n_alphabet = vocab.size
      	val lstm = Lstm.LSTM(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
      	lstm.listArguments().foreach {println}
	  	val shapeInfer = Map("_l0_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l0_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),
	  			"_l1_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN),"data"->Shape(BATCH_SIZE,SEQ_LENGTH,n_alphabet),"label"->Shape(BATCH_SIZE,SEQ_LENGTH))
	  	val (a,b,c) = lstm.inferShape(shapeInfer)
//	  	val exe = lstm.simpleBind(Context.defaultCtx,shapeDict=shapeInfer)
    	a.foreach(println)
    	b.foreach {println}
  	}
  	 
  	 
  	 
  
}