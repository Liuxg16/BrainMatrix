package thu.brainmatrix.nce_loss

import thu.brainmatrix._
import thu.brainmatrix.optimizer.SGD
import scala.collection.Set
import com.sun.org.apache.xalan.internal.xsltc.compiler.Number

/**
 * @author liuxianggen
 * @date 20160811
 * @brief 
 * @return
 * @example
 * @note the performance is so strange!!!
 */
object Toy_nce {
  
	def main(args:Array[String]){
		training_DIY
	}
	
	def training_DIY{
		val batch_size   = 128
		val vocab_size   = 10000
		val feature_size = 100
		val num_label    = 6
		val learningRate = 8f//8f=> 95.53%
		val numEpoch     = 2
		
		val dataTrain = new DataIter_nce(100000,batch_size,feature_size,vocab_size,num_label)
		val dataTest = new DataIter_nce(1000,batch_size,feature_size,vocab_size,num_label)
		
		val network = get_net(vocab_size,num_label)
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
      	val evalMetric = new NceAccuracy()
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

	def training_model{
		val batch_size   = 128
		val vocab_size   = 1000
		val feature_size = 100
		val num_label    = 6
		
		val data_train = new DataIter_nce(10000,batch_size,feature_size,vocab_size,num_label)
		val data_test = new DataIter_nce(1000,batch_size,feature_size,vocab_size,num_label)
		
		val network = get_net(vocab_size,num_label)
		val devs = Context.gpu(0)
		val models = new FeedForward(symbol = network,ctx = devs,
                                 numEpoch = 8,optimizer = new SGD(learningRate = 0.05f,momentum=0.9f,wd = 0.0001f),
                                 initializer = new Xavier(factorType = "in", magnitude = 2.34f))
		models.fit(trainData = data_train,evalData = data_test,evalMetric = new Accuracy(),
				kvStoreType = "local",epochEndCallback = null, batchEndCallback =  new Callback.Speedometer(batch_size, 50))
	}
	
	def get_net(vocab_size:Int,num_label:Int):Symbol = {
		val data = Symbol.Variable("data")
		val label = Symbol.Variable("label")
		val label_weight = Symbol.Variable("label_weight")
		val embed_weight = Symbol.Variable("embed_weight")
		var pred = Symbol.FullyConnected()(Map("data" -> data, "num_hidden" -> 100))
//		pred = Symbol.FullyConnected()(Map("data" -> pred, "num_hidden" -> vocab_size))
		nce_loss(pred,label,label_weight,embed_weight,vocab_size,100,num_label)
	}
	
	def nce_loss(data:Symbol,label:Symbol,label_weight:Symbol,embed_weight:Symbol,vocab_size:Int,num_hidden:Int,num_label:Int) :Symbol = {
		val label_embed = Symbol.Embedding("label_embed")(Map("data" -> label, "input_dim" -> vocab_size,
                                           "weight" -> embed_weight, "output_dim" -> num_hidden))
        val hidden = Symbol.Reshape()(Map("data"->data, "shape" -> s"(-1,1,$num_hidden)"))
        val pred = Symbol.broadcast_mul(hidden,label_embed)
		val pred1 = Symbol.Sum("sum")(Map("data"->pred,"axis"->2))
		Symbol.LogisticRegressionOutput("lro")(Map("data"->pred1,"label"->label_weight))
		
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
class DataIter_nce(count:Int,batch_size:Int,feature_size:Int,vocab_size: Int,num_label:Int) extends DataIter {
	/**
	 * author liuxianggen
	 * brief a generator of a feature and the label,where the feature is a vector,and the label can be learned
	 * return:
	 * data and label
	 */
	def mock_sample :(Array[Float],Array[Float],Array[Float]) = {
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
		
		val label = (s % vocab_size).toFloat +: (0 until num_label-1).map(_ => scala.util.Random.nextInt(vocab_size -1).toFloat)
		val label_weight = 1f +: Array.fill[Float](num_label-1)(0f)
		(ret, label.toArray,label_weight)
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
    override def provideLabel: Map[String, Shape] =  Map("label"->Shape(batch_size,num_label),"label_weight"->Shape(batch_size,num_label))
    
    val datas = (0 until (count/batch_size)).map(x =>{
    	val mock_samples =  (0 until batch_size).map(i =>{
    		mock_sample
    	}).toArray
    	val data_arr = mock_samples.map(_._1).foldLeft(Array[Float]())(_ ++ _)
    	val label_arr = mock_samples.map(_._2).foldLeft(Array[Float]())(_ ++ _)
    	val label_weight_arr = mock_samples.map(_._3).foldLeft(Array[Float]())(_ ++ _)
    	val data =NDArray.array(data_arr,Shape(batch_size,feature_size))
    	val label = NDArray.array(label_arr,Shape(batch_size,num_label))
    	val label_weight = NDArray.array(label_weight_arr,Shape(batch_size,num_label))
    	(data,label,label_weight)
    }).toArray
    
    // println(s"DataIter_ batches:${datas.length}")
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
    	val (data,label,label_weight) = datas(tempidx)
//    	new DataBatch(IndexedSeq(data),IndexedSeq(label),getIndex(),getPad())//error expression
    	new DataBatch(IndexedSeq(data.copy()),IndexedSeq(label.copy(),label_weight.copy()),getIndex(),getPad())
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
    override def getLabel(): IndexedSeq[NDArray] = IndexedSeq(datas(idx)._2, datas(idx)._3)
	
}




