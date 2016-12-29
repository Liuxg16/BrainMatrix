

//import thu.brainmatrix.optimizer.SGD

package thu.brainmatrix.cnn

import scala.collection.mutable.ListBuffer
import thu.brainmatrix.Context
import thu.brainmatrix.NDArray
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.IO
import thu.brainmatrix.Context.ctx2Array
import thu.brainmatrix.Symbol
import thu.brainmatrix.FeedForward
/**
 * by liuxiangen
 * 2016-4-5
 */
object TestTraininglxg {
  
  def main(args:Array[String]){
  	/**
  	 * for validation
  	 */
//  	val lrs = Array(0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.02,0.03,
//  			0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,100)//0.1
//  	val momentums = Array(0.01,0.5,0.1,0.5,0.7,0.8,0.83,0.86,0.88,0.9,0.93,0.96,0.99,1)//0.9
//  	val wds = Array(1e-6,1e-5,1e-4,1e-3,1e-2,1e-1)
//  	Array.range(0, 26).map(i  => {
//  			(0 to 13).map(j =>{
//  				(0 to 5).map(k =>
//  					train_lenet(lrs(i).toFloat,momentums(j).toFloat,wds(k).toFloat,1)	
//  					)
//  		})
//  			
//  	})
  	
//  	train_lenet(0.1f,0.9f,0.0001f,1)
  	
  	Training_mlp
  	
  }
  
  
   def train_lenet(lr:Float,mom:Float,wdd:Float,epochs:Int){
   	
  	println("----------------validation--------------------") 
  	println("lr: " + lr +"mom: " + mom +"wdd: " + wdd )
  	val batchSize = 100

    val data = Symbol.CreateVariable("data")
    val conv1 = Symbol.Convolution()(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 20, "kernel" -> (5, 5)/*, "stride" -> (2, 2)*/))

    val act1 = Symbol.Activation()(Map("data" -> conv1, "name" -> "tanh1", "act_type" -> "tanh"))                       
    val mp1 = Symbol.Pooling()(Map("data" -> act1, "name" -> "mp1",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
                                 
    //second conv
    val conv2 = Symbol.Convolution()(Map("data" -> mp1, "name" -> "conv2", "num_filter" -> 50,
                                       "kernel" -> (5, 5), "stride" -> (2, 2)))
    val act2 = Symbol.Activation()(Map("data" -> conv2, "name" -> "tanh2", "act_type" -> "tanh"))
    val mp2 = Symbol.Pooling()(Map("data" -> act2, "name" -> "mp2",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
                              
    //first fullc
    val fl = Symbol.Flatten()(Map("data" -> mp2, "name" -> "flatten"))
    val fc1 = Symbol.FullyConnected()(Map("data" -> fl, "name" -> "fc1", "num_hidden" -> 500))
    val act3 = Symbol.Activation()(Map("data" -> fc1, "name" -> "tanh3", "act_type" -> "tanh"))
    
    //second fullc
    val fc2 = Symbol.FullyConnected()(Map("data" -> act3, "name" -> "fc2", "num_hidden" -> 10))
    
    //loss
    val softmax = Symbol.SoftmaxOutput()(Map("data" -> fc2, "name" -> "sm"))
 
    val numEpoch = epochs
  
    val modelBase = new FeedForward(softmax,Context.cpu(), numEpoch = numEpoch,
      optimizer = new SGD(learningRate = lr, momentum = mom, wd = wdd))
    
    val trainDataIter = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0",
      "seed" -> "10"))

    val valDataIter = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/t10k-images-idx3-ubyte",
      "label" -> "data/t10k-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0", "silent" -> "0"))
    

    
    modelBase.fit(trainData = trainDataIter,evalData = valDataIter)
    println("Finish fit ...")

    val probArrays = modelBase.predict(data = valDataIter)
  
    val prob = probArrays(0)
    println("Finish predict ...")

    valDataIter.reset()
    val labels = ListBuffer.empty[NDArray]
    var evalData = valDataIter.next()
    while (evalData != null) {
      labels += evalData.label(0).copy()
      evalData = valDataIter.next()
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
  
//    def Alex_mnist{
//        val batchSize = 100
//        val input_data = mx.symbol.Variable(name="data")
////     stage 1
//        val conv1 = mx.symbol.Convolution(data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
//        val relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
//        val pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
//        val lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
////    # stage 2
//        val conv2 = mx.symbol.Convolution(data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
//        val relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
//        val pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
//        val lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
////    # stage 3
//        val conv3 = mx.symbol.Convolution(data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
//        val relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
//        val conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
//        val relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
//        val conv5 = mx.symbol.Convolution(data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
//        val relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
//        val pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
////    # stage 4
//        val flatten = mx.symbol.Flatten(data=pool3)
//        val fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
//        val relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
//        val dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
////    # stage 5
//        val fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=4096)
//        val relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
//        val dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
////    # stage 6
//        val fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=num_classes)
//        val softmax = mx.symbol.SoftmaxOutput(data=fc3, name="softmax“)
//      }
//   
   
    def Training_mlp{
    	
	  	val batchSize = 100
	    val data = Symbol.CreateVariable("data")
// 		val flatten = Symbol.Flatten(Map("data" -> data, "name" -> "flatten"))
 		val fc1 = Symbol.FullyConnected()(Map("data" -> data, "name" -> "fc1", "num_hidden" -> 128))
 		
	    val act1 = Symbol.Activation()(Map("data" -> fc1, "name" -> "relu1", "act_type" -> "relu"))
	    
	    val fc2 = Symbol.FullyConnected()(Map("data" -> act1, "name" -> "fc2", "num_hidden" -> 64))
 		val act2 = Symbol.Activation()(Map("data" -> fc2, "name" -> "relu2", "act_type" -> "relu"))
 		val fc3 = Symbol.FullyConnected()(Map("data" -> act2, "name" -> "fc3", "num_hidden" -> 10))
 		val sm = Symbol.SoftmaxOutput("sm")(Map("data" -> fc3))
 		

	    val numEpoch = 10
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
   
   
   
   
  def testCNN{

  	 val batchSize = 100

    val data = Symbol.CreateVariable("data")
    val conv1 = Symbol.Convolution()(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn1 = Symbol.BatchNorm()(Map("data" -> conv1, "name" -> "bn1"))
    val act1 = Symbol.Activation()(Map("data" -> bn1, "name" -> "relu1", "act_type" -> "relu"))                       
    val mp1 = Symbol.Pooling()(Map("data" -> act1, "name" -> "mp1",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
    val conv2 = Symbol.Convolution()(Map("data" -> mp1, "name" -> "conv2", "num_filter" -> 32,
                                       "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn2 = Symbol.BatchNorm()(Map("data" -> conv2, "name" -> "bn2"))
    val act2 = Symbol.Activation()(Map("data" -> bn2, "name" -> "relu2", "act_type" -> "relu"))
    val mp2 = Symbol.Pooling()(Map("data" -> act2, "name" -> "mp2",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
    val fl = Symbol.Flatten()(Map("data" -> mp2, "name" -> "flatten"))
    val fc2 = Symbol.FullyConnected()(Map("data" -> fl, "name" -> "fc2", "num_hidden" -> 10))
    val softmax = Symbol.SoftmaxOutput()(Map("data" -> fc2, "name" -> "sm"))
 
    val numEpoch = 1
  
    val modelBase = new FeedForward(softmax, Context.cpu(), numEpoch = numEpoch,
      optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
    
    val trainDataIter = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0",
      "seed" -> "10"))

    val valDataIter = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/t10k-images-idx3-ubyte",
      "label" -> "data/t10k-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0", "silent" -> "0"))
      
    
    modelBase.fit(trainData = trainDataIter,evalData = valDataIter)
//    println("Finish fit ...")
//
//    val probArrays = modelBase.predict(data = valDataIter)
//  
//    val prob = probArrays(0)
//    println("Finish predict ...")
//
//    valDataIter.reset()
//    val labels = ListBuffer.empty[NDArray]
//    var evalData = valDataIter.next()
//    while (evalData != null) {
//      labels += evalData.label(0).copy()
//      evalData = valDataIter.next()
//    }
//    val y = NDArray.concatenate(labels)
//
//    val py = NDArray.argmaxChannel(prob)
//  
//    var numCorrect = 0
//    var numInst = 0
//    for ((labelElem, predElem) <- y.toArray zip py.toArray) {
//      if (labelElem == predElem) {
//        numCorrect += 1
//      }
//      numInst += 1
//    }
//    val acc = numCorrect.toFloat / numInst
//    println("Final accuracy = ")
//    println(acc)

  }
  
  
  def testCNN1{
  	     // symbol net
    val batchSize = 100

    val data = Symbol.CreateVariable("data")
    val conv1 = Symbol.Convolution()(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn1 = Symbol.BatchNorm()(Map("data" -> conv1, "name" -> "bn1"))
    val act1 = Symbol.Activation()(Map("data" -> bn1, "name" -> "relu1", "act_type" -> "relu"))                       
    val mp1 = Symbol.Pooling()(Map("data" -> act1, "name" -> "mp1",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
    val conv2 = Symbol.Convolution()(Map("data" -> mp1, "name" -> "conv2", "num_filter" -> 32,
                                       "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn2 = Symbol.BatchNorm()(Map("data" -> conv2, "name" -> "bn2"))
    val act2 = Symbol.Activation()(Map("data" -> bn2, "name" -> "relu2", "act_type" -> "relu"))
    val mp2 = Symbol.Pooling()(Map("data" -> act2, "name" -> "mp2",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
    val fl = Symbol.Flatten()(Map("data" -> mp2, "name" -> "flatten"))
    val fc2 = Symbol.FullyConnected()(Map("data" -> fl, "name" -> "fc2", "num_hidden" -> 10))
    val softmax = Symbol.SoftmaxOutput()(Map("data" -> fc2, "name" -> "sm"))
                                 
  
//    val (a,b,c) = softmax.inferShape(Map("data"->Vector(32,1,48,48)))
//    a.foreach(println)
//    println("------------------------------------------------------------")
//    b.foreach {println}
//------------------------------------------------------
//Vector(100, 1, 48, 48)
//Vector(32, 1, 3, 3)
//Vector(32)
//Vector(32)
//Vector(32)
//Vector(32, 32, 3, 3)
//Vector(32)
//Vector(32)
//Vector(32)
//Vector(10, 288)
//Vector(10)
//Vector(100)
//------------------------------------------------------------
//Vector(100, 10)
    val numEpoch = 2
  
    val modelBase = new FeedForward(softmax, Context.cpu(), numEpoch = numEpoch,
      optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
    
    val trainDataIter = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0",
      "seed" -> "10"))

    val valDataIter = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/t10k-images-idx3-ubyte",
      "label" -> "data/t10k-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0", "silent" -> "0"))

    modelBase.fit(trainData = trainDataIter,evalData = valDataIter)
    println("Finish fit ...")

  
//    val prob = probArrays(0)
//    println("Finish predict ...")
//
//    valDataIter.reset()
//    val labels = ListBuffer.empty[NDArray]
//    var evalData = valDataIter.next()
//    while (evalData != null) {
//      labels += evalData.label(0).copy()
//      evalData = valDataIter.next()
//    }
//    val y = NDArray.concatenate(labels)
//
//    val py = NDArray.argmaxChannel(prob)
//    
////    println(y.shape)
////    println(py.shape)
//
//    var numCorrect = 0
//    var numInst = 0
//    for ((labelElem, predElem) <- y.toArray zip py.toArray) {
//      if (labelElem == predElem) {
//        numCorrect += 1
//      }
//      numInst += 1
//    }
//    val acc = numCorrect.toFloat / numInst
//    println("Final accuracy = ")
//    println(acc)

  }
  
 
}
