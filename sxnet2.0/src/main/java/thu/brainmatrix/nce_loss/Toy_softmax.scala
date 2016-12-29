package thu.brainmatrix.nce_loss

import thu.brainmatrix._
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.optimizer.Adam
import scala.collection.Set
/**
 * @author liuxianggen
 * @date 20160811
 * @brief 
 * @return
 * @example
 * @note the performance is so strange!!!
 */
object Toy_softmax {
  
	def main(args:Array[String]){
		training_DIY
	}
	
	def training_DIY{
		val batch_size   = 100
		
		val feature_size = 100
		val num_label    = 6
		val vocab_size   = 10//10000,
		val learningRate = 0.001f//0.01f,
		val numEpoch     = 3
		
		val dataTrain = new DataIter_(10000,batch_size,feature_size,vocab_size)
		val dataTest = new DataIter_(1000,batch_size,feature_size,vocab_size)
//		 val dataTrain = IO.MNISTIter(scala.collection.immutable.Map(
//	      "image" -> "data/train-images-idx3-ubyte",
//	      "label" -> "data/train-labels-idx1-ubyte",
//	      "data_shape" -> "(1, 28, 28)",
//	      "label_name" -> "sm_label",
//	      "batch_size" -> batch_size.toString,
//	      "shuffle" -> "1",
//	      "flat" -> "0",
//	      "silent" -> "0",
//	      "seed" -> "10"))
//
//	    val dataTest = IO.MNISTIter(scala.collection.immutable.Map(
//	      "image" -> "data/t10k-images-idx3-ubyte",
//	      "label" -> "data/t10k-labels-idx1-ubyte",
//	      "data_shape" -> "(1, 28, 28)",
//	      "label_name" -> "sm_label",
//	      "batch_size" -> batch_size.toString,
//	      "shuffle" -> "1",
//	      "flat" -> "0", "silent" -> "0"))
		
		val network = get_net(vocab_size)
		val ctx = Context.cpu(0)
		val datasAndLabels = dataTrain.provideData ++ dataTrain.provideLabel
      	val (argShapes, outputShapes, auxShapes) = network.inferShape(datasAndLabels)
      	val initializer = new Xavier(factorType = "in", magnitude = 2.34f)
      	val argNames = network.listArguments()
      	val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
      	val auxNames = network.listAuxiliaryStates()
      	val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap
      	//a collection that contains the ndarray of grad parameters
      	val gradDict = argNames.zip(argShapes).filter { 
			case (name, shape) =>
				!datasAndLabels.contains(name)
      	}.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap
      	argDict.foreach { case (name, ndArray) =>
        	if (!datasAndLabels.contains(name)) {
          		initializer.initWeight(name, ndArray)
        	}
     	}

      	val data = argDict("data")
      	val label = argDict("label")
      	val executor = network.bind(ctx, argDict, gradDict)
        val opt = new SGD(learningRate = learningRate, momentum=0.9f, wd = 0.0f)
        val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        	(idx, name, grad, opt.createState(idx, argDict(name)))
        }
      	val evalMetric = new Accuracy()
      	val batchEndCallback = new Callback.Speedometer(batch_size, 50)
//      val epochEndCallback = Utils.doCheckpoint(s"${incr.saveModelPath}/obama")

      	for (epoch <- 0 until numEpoch) {
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
	            executor.backward()
	            paramsGrads.foreach { case (idx, name, grad, optimState) =>
	              opt.update(idx, argDict(name), grad, optimState)
	            }
	            
	            // evaluate at end, so out_cpu_array can lazy copy
	            evalMetric.update(dataBatch.label, executor.outputs)
	
	            nBatch += 1
	            batchEndCallback.invoke(epoch, nBatch, evalMetric)
	            dataBatch.dispose()
	          }
	          if (doReset) {
	            dataTrain.reset()
	          }
	          // this epoch is done
	          epochDone = true
	        }
	        var (name, value) = evalMetric.get
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
	          evalMetric.update(evalBatch.label, executor.outputs)
	          evalBatch.dispose()
	        }
	        val (name_eval, value_eval) = evalMetric.get
	        println(s"Epoch[$epoch] Validation-$name_eval=$value_eval")
	        
	//        epochEndCallback.invoke(epoch, symbol, argDict, auxDict)
	    }
      	executor.dispose()
	}
	
	
	def training_model(){
		val batch_size   = 100
		val vocab_size   = 10000
		val feature_size = 100
		val num_label    = 6
		
		val data_train = new DataIter_(100000,batch_size,feature_size,vocab_size)
		val data_test = new DataIter_(1000,batch_size,feature_size,vocab_size)
		
		val network = get_net(vocab_size)
		val devs = Context.gpu(0)
		val models = new FeedForward(symbol = network,ctx = devs,
                                 numEpoch = 8,optimizer = new SGD(learningRate = 0.05f,momentum=0.9f,wd = 0.0001f),
                                 initializer = new Xavier(factorType = "in", magnitude = 2.34f))
		models.fit(trainData = data_train,evalData = data_test,evalMetric = new Accuracy(),
				kvStoreType = "local",epochEndCallback = null, batchEndCallback =  new Callback.Speedometer(batch_size, 50))
	}
	
	
	def get_net(vocab_size:Int):Symbol = {
		val data = Symbol.Variable("data")
		val label = Symbol.Variable("label")
		val embed = Symbol.FullyConnected()(Map("data" -> data, "num_hidden" -> 100))
//		val act1 = Symbol.Activation(name = "relu1")(Map("data" -> embed, "act_type" -> "sigmoid"))
//		val fc2 = Symbol.FullyConnected(name = "fc2")(Map("data" -> act1, "num_hidden" -> 100))
//    	val act2 = Symbol.Activation(name = "relu2")(Map("data" -> fc2, "act_type" -> "sigmoid"))
		val pred = Symbol.FullyConnected()(Map("data" -> embed	, "num_hidden" -> vocab_size))
		val sm  = Symbol.SoftmaxOutput("sm")(Map("data"->pred,"label"->label))
		sm
	}
}


/**
 * @author liuxianggen
 * @date 20150911
 * @brief all the global infomation are listed in there
 * @param count: the number of class
 * @param count: the number of class
 * @return
 * @example
 * @note
 */
class DataIter_(count:Int,batch_size:Int,feature_size:Int,vocab_size: Int) extends DataIter {
	/**
	 * author liuxianggen
	 * brief a generator of a feature and the label,where the feature is a vector,and the label can be learned
	 * return:
	 * data and label
	 */
	def mock_sample :(Array[Float],Float) = {
		val ret = Array.fill[Float](feature_size)(0f)
		var rn = Set[Int]()
		
		while(rn.size<3){
			rn = rn + scala.util.Random.nextInt(feature_size-1) 
		}
		var s = 0
		rn.foreach { x => {
			ret(x)= 1.0f
			s *= feature_size	
			s += x
		}}
		(ret, (s % vocab_size).toFloat)
	}
	
	
	private var idx = 0
	
	override def batchSize: Int = batch_size
	/**
     * the index of current batch
     * @return
     */
    override def getIndex(): IndexedSeq[Long] = IndexedSeq[Long]()

    // The name and shape of label provided by this iterator
    override def provideData: Map[String, Shape] = Map("data"->Shape(batch_size,feature_size))
      /**
     * get the number of padding examples
     * in current batch
     * @return number of padding examples in current batch
     */
    override def getPad(): Int = 0

    // The name and shape of data provided by this iterator
    override def provideLabel: Map[String, Shape] =  Map("label"->Shape(batch_size))
    
    val datas = (0 until (count/batch_size)).map(x =>{
    	val mock_samples =  (0 until batch_size).map(i =>{
    		mock_sample
    	}).toArray
    	val data_arr = mock_samples.map(_._1).foldLeft(Array[Float]())(_ ++ _)
    	val label = NDArray.array(mock_samples.map(_._2),Shape(batch_size))
    	val data =NDArray.array(data_arr,Shape(batch_size,feature_size))
    	(data,label)
    }).toArray
    
    println(s"DataIter_ batches:${datas.length}")
    /**
     * wrong template
     */
//    override def next(): DataBatch = {
//    	val tempidx = idx
//    	idx += 1
//    	datas(tempidx)
//    }
    
    override def next(): DataBatch = {
    	val tempidx = idx
    	idx += 1
    	val (data,label) = datas(tempidx)
//    	new DataBatch(IndexedSeq(data),IndexedSeq(label),getIndex(),getPad())//error expression
    	new DataBatch(IndexedSeq(data.copy()),IndexedSeq(label.copy()),getIndex(),getPad())
    }
    
    
     override def reset(): Unit = {
    	 idx = 0
     }
     
     override def hasNext: Boolean = {
      if (idx < datas.length) true else false
    }
     
     /**
     * get data of current batch
     * @return the data of current batch
     */
    override def getData(): IndexedSeq[NDArray] = IndexedSeq(datas(idx)._1)

    /**
     * Get label of current batch
     * @return the label of current batch
     */
    override def getLabel(): IndexedSeq[NDArray] = IndexedSeq(datas(idx)._2)
	
}