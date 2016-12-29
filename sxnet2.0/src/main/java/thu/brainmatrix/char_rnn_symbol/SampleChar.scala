package thu.brainmatrix.char_rnn_symbol

import Config._
import thu.brainmatrix.Base
import thu.brainmatrix.FeedForward
import thu.brainmatrix.Context
import thu.brainmatrix.io.NDArrayLSTMIter
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.Model
import scala.io.Source
import thu.brainmatrix.NDArray
import scala.collection.mutable.ListBuffer
import thu.brainmatrix.util.mathTool
import thu.brainmatrix.Shape
import scala.util.Random
object SampleChar {
	
	def main(args:Array[String]){
		sampleChar_vec_feather
  
	}
	
	
	def sampleChar_vec_feather{
		val ctx = Context.cpu(0)
		val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
		
  		var bacov = for((k,v)<- vocab) yield (v,k)
  		
  		val revertVocab = bacov.updated(bacov.size-1, '?')
  		println(bacov.size)
	    val n_alphabet = vocab.size
		 // load from check-point
      	val (_, argParams, _) = Model.loadCheckpoint("./model/obama", Config.N_EPOCH)
      	val model = new InferCharModel(LSTM_N_LAYER, n_alphabet,DIM_HIDDEN, DIM_EMBED,argParams,ctx,DROPOUT)
      	
		val seqLength = 100
      val inputNdarray = NDArray.zeros(1,n_alphabet)
//      val revertVocab = Utils.makeRevertVocab(vocab)

      // Feel free to change the starter sentence
      var output = "hello"
      val randomSample = true
      var newSentence = true
      val ignoreLength = output.length()

      for (i <- 0 until seqLength) {
        if (i <= ignoreLength - 1) makeInput(output(i), vocab, inputNdarray)
        else makeInput(output.takeRight(1)(0), vocab, inputNdarray)
        val prob = model.forward(inputNdarray, newSentence)
        newSentence = false
        val nextChar = makeOutput(prob, revertVocab, randomSample)
        if (nextChar == Config.UNKNOW_CHAR) newSentence = true
        if (i >= ignoreLength) output = output :+ nextChar
      }

      // Let's see what we can learned from char in Obama's speech.
      println(output)
		
		
		model.dispose()
      	println("*----------------------------------------------*")
	
      	
	
	
	}
	  	
	  // make input from char
  def makeInput(char: Char, vocab: Map[Char, Int], arr: NDArray): Unit = {
    val idx = vocab(char)
    val tmp = NDArray.zeros(arr.shape)
    tmp(0,idx)  = 1
    arr.set(tmp)
  }
	  	
  // we can use random output or fixed output by choosing largest probability
  def makeOutput(prob: Array[Float], vocab: Map[Int, Char],
                 sample: Boolean = false, temperature: Float = 1f): Char = {
    var idx = -1
    val char = if (sample == false) {
      idx = ((-1f, -1) /: prob.zipWithIndex) { (max, elem) =>
        if (max._1 < elem._1) elem else max
      }._2
      if (vocab.contains(idx)) vocab(idx)
      else Config.UNKNOW_CHAR
    } else {
      val fixDict = Array[Char]() ++ (0 until vocab.size).map(i => vocab(i))
      var scaleProb = prob.map(x => if (x < 1e-6) 1e-6 else if (x > 1 - 1e-6) 1 - 1e-6 else x)
      var rescale = scaleProb.map(x => Math.exp(Math.log(x) / temperature).toFloat)
      val sum = rescale.sum.toFloat
      rescale = rescale.map(_ / sum)
      choice(fixDict, rescale)
    }
    char
  }
  
  	// helper function for random sample
  def cdf(weights: Array[Float]): Array[Float] = {
    val total = weights.sum
    var result = Array[Float]()
    var cumsum = 0f
    for (w <- weights) {
      cumsum += w
      result = result :+ (cumsum / total)
    }
    result
  }

  def choice(population: Array[Char], weights: Array[Float]): Char = {
    assert(population.length == weights.length)
    val cdfVals = cdf(weights)
    val x = Random.nextFloat()
    var idx = 0
    var found = false
    for (i <- 0 until cdfVals.length) {
      if (cdfVals(i) >= x && !found) {
        idx = i
        found = true
      }
    }
    population(idx)
  }
	  	
//      val ctx_cpu = Context.cpu(0)
//    	  val map_train = NDArray.zeros(Shape(BATCH_SIZE,SEQ_LENGTH,n_alphabet), ctx_cpu)
//    	  for(i<- 0 until BATCH_SIZE){
//    	 	  for(j<- 0 until SEQ_LENGTH)
//    	 		map_train(i,j,j) = 1	 	  
//    	  }
//      
////	    map_train(0)(0,0) = 0
////	    map_train(0)(0,5) = 1
//	    var text_arr = ListBuffer[NDArray]()
//	    text_arr += NDArray.argmaxChannel(map_train)
////	    text_arr.foreach {println}
////	    for(i<-0 until SEQ_LENGTH*40){
//	    	val dataIter = new NDArrayLSTMIter(data = IndexedSeq(map_train),dataName = "data",IndexedSeq(NDArray.zeros(Shape(BATCH_SIZE))),"label",
//	    			dataBatchSize = BATCH_SIZE, shuffle = false,lastBatchHandle = "pad")//the rest will discard
//	    	
//	    	data.set(dataIter.next().data(0))
//	    	executor.forward
//	    	val probArrays = executor.outputs
////	    	println(probArrays(i%SEQ_LENGTH))
//	    	
//	    	val outputArr = probArrays.map{ x => x.copyTo(ctx_cpu) }
//	    	println("-----------------")
//	    	val outcharInt = mathTool.SampleByPro2D(outputArr(0)).map(_.toFloat)
////	    	val outchar = NDArray.array(outcharInt,Shape(BATCH_SIZE,SEQ_LENGTH))
////	    	text_arr += outchar
////	    	val temp = NDArray.zeros(Shape(BATCH_SIZE,n_alphabet))
//////	    	println(outchar)
////	    	for(j<-0 until BATCH_SIZE){
////	    		temp(j,outchar(j).toInt) = 1
////	    	}
////	    	temp.copyTo(map_train((i+1)%SEQ_LENGTH))
////	    }
//      
//       println("--------------")
//	  	var s = ""
//	    val a = outcharInt.map(x => bacov(x.toInt))
//	    a.foreach { x => s += x }
//	    println(s)
//      
//      
      
      
      
  	
	
	
	

	
	def sampleChar_id_feather{
			val vocab = seq_IO.build_vocabulary(INPUT_FILE_NAME, VOCAB_FILE_NAME)  
  	var bacov = for((k,v)<- vocab) yield (v,k)
  	bacov = bacov.updated(bacov.size-1, '?')
  	println(bacov)
    val n_alphabet = vocab.size
    val lstm = Lstm.lstmGenerator(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, n_alphabet, DROPOUT)
    Base.INPUTSHAPE_AUXILIARY = Map("_l0_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l0_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_h"->Shape(BATCH_SIZE,DIM_HIDDEN),"_l1_init_c"->Shape(BATCH_SIZE,DIM_HIDDEN))
    val modelBase = new FeedForward(lstm, Context.cpu(), numEpoch = N_EPOCH,optimizer = new SGD(learningRate = LEARNING_RATE, momentum = MOMENTUM, wd = WEIGHT_DECAY))
//  	modelBase.loadModelParams(s"./model/charLSTM.params_${N_EPOCH}")
//    lstm.listArguments().foreach {println}
    val source  = Source.fromFile(INPUT_FILE_NAME) 
    val seq_input = source.mkString
    val len_train = math.round(seq_input.length()*DATA_TRAIN_RATIO).toInt
    val text_train = seq_input.take(len_train)
    val inputName = "data"
    val labelName = "label"
    
    val map_train = (0 until SEQ_LENGTH).map(x => (NDArray.ones(Shape(BATCH_SIZE,1))*10))
    
    var text_arr = ListBuffer[NDArray]()
    text_arr += map_train(0)
    for(i<-0 until SEQ_LENGTH-1){
    	val dataIter = new NDArrayLSTMIter(data = map_train,dataName = inputName,IndexedSeq(NDArray.zeros(Shape(BATCH_SIZE,1))),"label",
    			dataBatchSize = BATCH_SIZE, shuffle = false,lastBatchHandle = "pad")//the rest will discard
//    	val traindata = seq_IO.SampleDataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
    	val probArrays = modelBase.predict(data = dataIter)
    	
    	val outcharInt = mathTool.SampleByPro2D(probArrays(i)).map(_.toFloat)
    	val outchar = NDArray.array(outcharInt,Shape(BATCH_SIZE,1))
    	text_arr += outchar
    	outchar.copyTo(map_train(i+1))
    }
  	
  	for(i<-0 until SEQ_LENGTH-1){
    	val dataIter = new NDArrayLSTMIter(data = map_train,dataName = inputName,IndexedSeq(NDArray.zeros(Shape(BATCH_SIZE,1))),"label",
    			dataBatchSize = BATCH_SIZE, shuffle = false,lastBatchHandle = "pad")//the rest will discard
//    	val traindata = seq_IO.SampleDataIter(text = text_train,labelName = "label",vocab = vocab,batch_size = BATCH_SIZE,seq_len = SEQ_LENGTH)
    	val probArrays = modelBase.predict(data = dataIter)
    	
    	val outcharInt = mathTool.SampleByPro2D(probArrays(i)).map(_.toFloat)
    	val outchar = NDArray.array(outcharInt,Shape(BATCH_SIZE,1))
    	text_arr += outchar
    	for(j<- 0 until SEQ_LENGTH-1){
    		map_train(j+1).copyTo(map_train(j))
    	}
    	outchar.copyTo(map_train(SEQ_LENGTH-1))
    }
  	
  	
//  	text_arr.foreach {println}
//  	println("--------------")
  	var texts = for(j<-0 until BATCH_SIZE) yield new StringBuilder
  	
    for(i<-0 until text_arr.length){
    	for((c,s)<-NDAtoChar(bacov,text_arr(i)).zip(texts)){
    			s += c
    	}
    }    
    
  	for((s,idx)<-texts.zipWithIndex){
  		println(s"\ntext $idx th:")
  		println(s)
  	}
    println("\nends...")
    println(text_arr.length)
	}
	
	
	def NDAtoChar(vocab:Map[Int,Char],nda:NDArray):Array[Char] = {
		
	    nda.toArray.map(x => vocab(x.toInt))
		
	}	


}