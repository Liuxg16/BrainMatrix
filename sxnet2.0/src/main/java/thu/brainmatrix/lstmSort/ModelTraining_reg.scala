package thu.brainmatrix.lstmSort


import thu.brainmatrix._
import thu.brainmatrix.optimizer.Adam
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.util.IOHelper
import thu.brainmatrix.rnn.Utils
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._


/**
 * author: liuxianggen
 * data: 20160826
 * brief: this piece of code is almost the same as the ModelTraining.scala
 * The only difference is taht this one is added the regularization part!
 */
object ModelTraining_reg {
	def main(args:Array[String]){
//		mlp
		lstmSort_reg
	}
	
	def lstmSort_reg{
		val path_train = "./data/sort.train.txt"
		val path_test = "./data/sort.valid.txt"
		val saveModelPath = "./model"
	    val batch_size = 100
	    val buckets = List(5)
	    val num_hidden = 300
	    val num_embed = 512
	    val num_lstm_layer = 2
		val seqLen = 5
	    val num_epoch = 1
	    val learningRate = 0.01f
	    val momentum = 0.9
	    
	    val ctx = Context.gpu(0)
	//    # a dict that contains the  word and the index
	    val vocab =  IOHelper.buildVocab("./data/sort.train.txt")
	    println(vocab)

	    val symbol = Lstm.bi_lstmUnroll_reg(num_lstm_layer, seqLen, vocab.size,
                    numHidden = num_hidden, numEmbed = num_embed,
                    numLabel = vocab.size)
	    
        // initalize states for LSTM
      	val initC = for (l <- 0 until num_lstm_layer) yield (s"l${l}_init_c", (batch_size, num_hidden))
	  val initH = for (l <- 0 until num_lstm_layer) yield (s"l${l}_init_h", (batch_size, num_hidden))
	  val initStates = initC ++ initH
	  //regard  '\n' as the separator to train
	  val dataTrain = new ButketIo.BucketSentenceIter(path_train, vocab, buckets,
	                                      batch_size, initStates)
	  val dataTest = new ButketIo.BucketSentenceIter(path_test, vocab, buckets,
	                                      batch_size, initStates)
	  val datasAndLabels = dataTrain.provideData ++ dataTrain.provideLabel
	  val (argShapes, outputShapes, auxShapes) = symbol.inferShape(datasAndLabels)
	
	  val initializer = new Xavier(factorType = "in", magnitude = 2.34f)
	
	  val argNames = symbol.listArguments()
	  val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
	  val auxNames = symbol.listAuxiliaryStates()
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
      val label = argDict("softmax_label")

      val executor = symbol.bind(ctx, argDict, gradDict)

//      println("*---------------------------------------*")
//      println(executor.debugStr)
      
      val opt = new Adam(learningRate = learningRate, wd = 0.0001f)

      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new CustomMetric(accuracy1, "perplexity")
      val batchEndCallback = new Callback.Speedometer(batch_size, 50)
      val epochEndCallback = Utils.doCheckpoint(s"${saveModelPath}/lstmSort")

      for (epoch <- 0 until num_epoch) {
        // Training phase
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false
        // Iterate over training data.
        dataTrain.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && dataTrain.hasNext) {
            val dataBatch = dataTrain.next()

            data.set(dataBatch.data(0))
            label.set(dataBatch.label(0))
            executor.forward(isTrain = true)
//            println("-----------------backward----------------------")
            
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }

            // evaluate at end, so out_cpu_array can lazy copy
            evalMetric.update(dataBatch.label, Array(executor.outputs(1)))
            dataBatch.dispose()
            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            dataTrain.reset()
          }
          // this epoch is done
          epochDone = true
        }
        val (name, value) = evalMetric.get
        println(s"Epoch[$epoch] Train-$name=$value")
        val toc = System.currentTimeMillis
        println(s"Epoch[$epoch] Time cost=${toc - tic}")
        
      
      //VALIDATION
        evalMetric.reset()
        dataTest.reset()
        // TODO: make DataIter implement Iterator
        while (dataTest.hasNext) {
          val evalBatch = dataTest.next()
          data.set(evalBatch.data(0))
          label.set(evalBatch.label(0))
          executor.forward(isTrain = false)
          evalMetric.update(evalBatch.label, Array(executor.outputs(1)))
          evalBatch.dispose()
        }
        val (name_eval, value_eval) = evalMetric.get
        println(s"Epoch[$epoch] Validation-$name_eval=$value_eval")

        epochEndCallback.invoke(epoch, symbol, argDict, auxDict)
      }
      
      executor.dispose()
		
	  println("ends...")
	}
	
	def mlp{
		val saveModelPath = "./model"
	    val batch_size = 100
	    val buckets = List(5)
	    val num_hidden = 300
	    val num_embed = 512
	    val num_lstm_layer = 2
	    val num_epoch = 1
	    val learningRate = 0.1f
	    val momentum = 0.9f
	    
	    val ctx = Context.cpu(0)
	//    # a dict that contains the  word and the index

//	    val symbol = Network.mlp()	    
	    val symbol = Network.lenet
		
	     val dataTrain = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "softmax_label",
      "batch_size" -> batch_size.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0",
      "seed" -> "10"))


	  val datasAndLabels = dataTrain.provideData ++ dataTrain.provideLabel /*++ Map("weight_1"->Shape(128,784))*/
//	  println(datasAndLabels)
	  val (argShapes, outputShapes, auxShapes) = symbol.inferShape(datasAndLabels)
	
//	  argShapes.foreach {println}
//	  outputShapes.foreach {println}
	  
	  val initializer = new Xavier(factorType = "in", magnitude = 2.34f)
	
	  val argNames = symbol.listArguments()
	  val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
	  val auxNames = symbol.listAuxiliaryStates()
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
      val label = argDict("softmax_label")

      val executor = symbol.bind(ctx, argDict, gradDict)

      val opt = new SGD(learningRate = learningRate,momentum = momentum)

      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new Accuracy
      val batchEndCallback = new Callback.Speedometer(batch_size, 50)
      val epochEndCallback = Utils.doCheckpoint(s"${saveModelPath}/lstmSort")

      for (epoch <- 0 until num_epoch) {
        // Training phase
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false
        // Iterate over training data.
        dataTrain.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && dataTrain.hasNext) {
            val dataBatch = dataTrain.next()

            data.set(dataBatch.data(0))
            label.set(dataBatch.label(0))
            executor.forward(isTrain = true)
//           
            println("-------------------------")
            println(executor.outputs(0))
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }

//            // evaluate at end, so out_cpu_array can lazy copy
//            evalMetric.update(dataBatch.label, Array(executor.outputs(1)))
            dataBatch.dispose()
            nBatch += 1
//            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            dataTrain.reset()
          }
          // this epoch is done
          epochDone = true
        }
        val (name, value) = evalMetric.get
        println(s"Epoch[$epoch] Train-$name=$value")
        val toc = System.currentTimeMillis
        println(s"Epoch[$epoch] Time cost=${toc - tic}")
        
      
      //VALIDATION
//        evalMetric.reset()
//        dataTest.reset()
//        // TODO: make DataIter implement Iterator
//        while (dataTest.hasNext) {
//          val evalBatch = dataTest.next()
//          data.set(evalBatch.data(0))
//          label.set(evalBatch.label(0))
//          executor.forward(isTrain = false)
//          println(executor.outputs(0))
////          evalMetric.update(evalBatch.label, executor.outputs)
//          evalBatch.dispose()
//        }
//        val (name_eval, value_eval) = evalMetric.get
//        println(s"Epoch[$epoch] Validation-$name_eval=$value_eval")

//        epochEndCallback.invoke(epoch, symbol, argDict, auxDict)
      }
      
      executor.dispose()
		
	  println("ends...")
	}
	
	// Evaluation
  def perplexity(label: NDArray, pred: NDArray): Float = {
	  
    val shape = label.shape
    val size = shape(0) * shape(1)
    val labelT = {
      val tmp = label.toArray.grouped(shape(1)).toArray
      val result = Array.fill[Float](size)(0f)
      var idx = 0
      for (i <- 0 until shape(1)) {
        for (j <- 0 until shape(0)) {
          result(idx) = tmp(j)(i)
          idx += 1
        }
      }
      result
    }
    var loss = 0f
    val predArray = pred.toArray.grouped(pred.shape(0)).toArray
    for (i <- 0 until pred.shape(1)) {
      loss += -Math.log(Math.max(1e-10, predArray(i)(labelT(i).toInt)).toFloat).toFloat
    }
    loss / size
  }
	
  	// Evaluation
  def accuracy1(label: NDArray, pred: NDArray): Float = {
	  var sumMetric = 0f
	  val shape = label.shape
      val size = shape(0) * shape(1)
	  val labelT = {
      val tmp = label.toArray.grouped(shape(1)).toArray
      val result = Array.fill[Float](size)(0f)
      var idx = 0
      for (i <- 0 until shape(1)) {
        for (j <- 0 until shape(0)) {
          result(idx) = tmp(j)(i)
          idx += 1
        }
      }
      result
    }
	  
	  
	  val predLabel = NDArray.argmaxChannel(pred)
      for ((labelElem, predElem) <- labelT zip predLabel.toArray) {
        if (math.abs(labelElem - predElem)<1e-6) {
//        	println(s"labelElem:$labelElem,predElem:$predElem")
          sumMetric += 1
        }
      }
      predLabel.dispose()
      sumMetric/(label.size)
      
  }
	
}