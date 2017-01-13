package thu.brainmatrix

import thu.brainmatrix.optimizer.SGD
import scala.collection.mutable.ListBuffer
object Main {
  def main(args:Array[String]){
	 val batchSize = 100
	    val data = Symbol.CreateVariable("data")
// 		val flatten = Symbol.Flatten(Map("data" -> data, "name" -> "flatten"))
 		val fc1 = Symbol.FullyConnected()(Map("data" -> data, "name" -> "fc1", "num_hidden" -> 128))
 		
	    val act1 = Symbol.Activation()(Map("data" -> fc1, "name" -> "relu1", "act_type" -> "relu"))
	    
	    val fc2 = Symbol.FullyConnected()(Map("data" -> act1, "name" -> "fc2", "num_hidden" -> 64))
 		val act2 = Symbol.Activation()(Map("data" -> fc2, "name" -> "relu2", "act_type" -> "relu"))
 		val fc3 = Symbol.FullyConnected()(Map("data" -> act2, "name" -> "fc3", "num_hidden" -> 10))
 		val sm = Symbol.SoftmaxOutput("sm")(Map("data" -> fc3))
 		

	    val numEpoch = 5
	    val model = new FeedForward(sm, Context.cpu(), numEpoch = numEpoch,
	      optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
	
	    // get data
	   // "./scripts/get_mnist_data.sh" !
	    val trainDataIter = IO.MNISTIter(scala.collection.immutable.Map(
	      "image" -> "data/train-images-idx3-ubyte",
	      "label" -> "data/train-labels-idx1-ubyte",
	      "data_shape" -> "(784)",
	      "label_name" -> "sm_label",
	      "batch_size" -> batchSize.toString,
	      "shuffle" -> "1",
	      "flat" -> "1",
	      "silent" -> "0",
	      "seed" -> "10"))
	    println(trainDataIter.provideLabel)
	    
	    val valDataIter = IO.MNISTIter(scala.collection.immutable.Map(
	      "image" -> "data/t10k-images-idx3-ubyte",
	      "label" -> "data/t10k-labels-idx1-ubyte",
	      "data_shape" -> "(784)",
	      "label_name" -> "sm_label",
	      "batch_size" -> batchSize.toString,
	      "shuffle" -> "1",
	      "flat" -> "1", "silent" -> "0"))
	    model.fit(trainDataIter, valDataIter)
	    println("Finish fit ...")
	    val probArrays = model.predict(valDataIter)
	    val prob = probArrays(0)
	    println("Finish predict ...")
	
	    valDataIter.reset()
	    val labels = ListBuffer.empty[NDArray]
	    
	    while (valDataIter.hasNext) {
	      var evalData = valDataIter.next()
	      labels += evalData.label(0).copy()
	    }
	    val y = NDArray.concatenate(labels)
	    val py = NDArray.argmaxChannel(prob)
	    var numCorrect = 0
	    var numInst = 0
	    for ((labelElem, predElem) <- y.toArray zip py.toArray) {
	      if (labelElem == predElem) {
	        numCorrect += 1
	      }
	      numInst += 1
	    }
	    val acc = numCorrect.toFloat / numInst
	    println("Final accuracy = ")
	    println(acc)
  }
}