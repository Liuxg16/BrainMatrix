//
//import ml.dmlc.mxnet.Symbol
//import ml.dmlc.mxnet.Base._
//import scala.collection.mutable.{ArrayBuffer,ListBuffer}
//import ml.dmlc.mxnet.Context
//import ml.dmlc.mxnet.lxg.ScalaSymFunction
//import ml.dmlc.mxnet.NDArray
//import ml.dmlc.mxnet.Random
//import ml.dmlc.mxnet.Executor
//
//
//object TestSymbol{
//    
//    
//	def main(args:Array[String]){
//			inferShapeTest_mxnet
////			ListAtomicFuncTest
////			simpleBind
////			auxTest
////		addTest
////	  printVectorTest
////		symbolOperatorTest
////		addTest
////		simpleBind
////	  ElementWiseSumTest
////		convTest
//		println("---------------------------------------")
//	}
//
//  def inferShapeTest_mxnet{
//  	 val data = Symbol.Variable("data")
//     val fc1 = Symbol.FullyConnected(Map("data" -> data, "name" -> "fc2", "num_hidden" -> 12))
//  	
//  	 val kwargs_shape = Map("data"->Vector(200,15))
//  	 val keys = ArrayBuffer.empty[String]
//     val indPtr = ArrayBuffer(0)
//     val sdata = ArrayBuffer.empty[Int]
//     kwargs_shape.foreach { case (key, shape) =>
//       keys += key
//       sdata ++= shape
//       indPtr += sdata.size
//     }
//  	println("keys:")
//  	keys.foreach {println}
//  	println("\nsdata:")
//  	sdata.foreach(println)
//  	println("\nindPtr:"+indPtr)
//
//  	println("\n---------------------------------------------------")
//     val (argShapes, _ , auxShapes) = fc1.inferShape(keys.toArray, indPtr.toArray, sdata.toArray)
//     argShapes.foreach { x => {
//    	 x.foreach{y => print(" "+ y )} 
//    	 println
//    	 }
//     }
////     fc2.listArguments().foreach { println }
//  }
//  
//  
//  	def symbolOperatorTest(){
//  		val data1 = Symbol.Variable("data1")
//    	val data2 = Symbol.Variable("data2")
////    	
//    	
//    	val sum = data1 + data2
//    	sum.listArguments().foreach(println)
//  	}
//  
//    def printVectorTest(){
//      _LIB.mxSymbolPrintVector(3,Array(1,2,3))
//    }
//  
//  def ListAtomicFuncTest{
//  	val symbolList = ListBuffer.empty[SymbolHandle]
//    checkCall(_LIB.mxSymbolListAtomicSymbolCreators(symbolList))
//    symbolList.map(makeAtomicSymbolFunction).toMap
//    
//    def makeAtomicSymbolFunction(handle: SymbolHandle): (String,ScalaSymFunction) = {
//        val name = new RefString
//        val desc = new RefString
//        val keyVarNumArgs = new RefString
//        val numArgs = new MXUintRef
//        val argNames = ListBuffer.empty[String]
//        val argTypes = ListBuffer.empty[String]
//        val argDescs = ListBuffer.empty[String]
//    
//        checkCall(_LIB.mxSymbolGetAtomicSymbolInfo(
//          handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs))
//        val paramStr = ctypes2docstring(argNames, argTypes, argDescs)
//        val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
//        println("Atomic Symbol function defination:\n{}", docStr)
//        (name.value, new ScalaSymFunction(handle, keyVarNumArgs.value))
//      }
//
//  }
//  
//  
//  def simpleBind{
//  	import ml.dmlc.mxnet.Context
//  	val batchSize = 100
//
//    val data = Symbol.Variable("data")
//    val conv1 = Symbol.Convolution(Map("data" -> data, "name" -> "conv1",
//                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))
//    val bn1 = Symbol.BatchNorm(Map("data" -> conv1, "name" -> "bn1"))
//    val act1 = Symbol.Activation(Map("data" -> bn1, "name" -> "relu1", "act_type" -> "relu"))
//    val mp1 = Symbol.Pooling(Map("data" -> act1, "name" -> "mp1",
//                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
//
//    val conv2 = Symbol.Convolution(Map("data" -> mp1, "name" -> "conv2", "num_filter" -> 32,
//                                       "kernel" -> (3, 3), "stride" -> (2, 2)))
//    val bn2 = Symbol.BatchNorm(Map("data" -> conv2, "name" -> "bn2"))
//    val act2 = Symbol.Activation(Map("data" -> bn2, "name" -> "relu2", "act_type" -> "relu"))
//    val mp2 = Symbol.Pooling(Map("data" -> act2, "name" -> "mp2",
//                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
//
//    val fl = Symbol.Flatten(Map("data" -> conv1, "name" -> "flatten"))
//    val fc2 = Symbol.FullyConnected(Map("data" -> fl, "name" -> "fc2", "num_hidden" -> 10))
//    val softmax = Symbol.SoftmaxOutput(Map("data" -> fc2, "name" -> "sm"))
//    
//    val  dataShapes = Map("data" -> Vector(20,1,28, 28))
//  	println("*****************************************")
////  	 val dataShapes_ =collection.immutable.Map(dataShapes.toList: _*) 
//    val exe = softmax.simpleBind(Context.cpu(), "write", shapeDict = dataShapes)
//  //  val dataArr = Random.normal(0, 1,Vector(100,1,28,28))
//    println("*****************************************")
//         println("----------------------------")
//            println(softmax.debugStr)
//            println(exe.debugStr)
////    for(i<-0 until 10){
//	    exe.forward(isTrain = true)
//	    exe.backward()
//	    
////    }
////  	println(exe.outputs(0))
//  	 
//  }
//  
//  
//  def auxTest{
//  	val data = Symbol.Variable("data")
//    val conv1 = Symbol.Convolution(Map("data" -> data, "name" -> "conv1",
//                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))
//    conv1.listAuxiliaryStates().foreach(println)
//    
//  }
//  
//  
//  def addTest{
//	  
//	  val a = Symbol.Variable("a")
//	  val b = Symbol.Variable("b")
//	  val c = a + 2
//	  val args = c.listArguments()
//	  args.foreach(println)
//	  println("-----------------------------------")
//	
//  }
//  
//  
//  def convTest{
//	val shape = Vector(20,1,28, 28)
//    val lhs = Symbol.Variable("lhs")
//    val rhs = Symbol.Variable("rhs")
//    val sum = lhs + rhs
// 
//    println("++++++++++++++++++++++++++++++++++++++++++++++")
//	val conv1 = Symbol.Convolution(Map("data" -> sum, "name" -> "conv1",
//                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))
//    
//    val fc = Symbol.FullyConnected(Map("data" -> sum, "name" -> "fc3", "num_hidden" -> 10))
//
//    val softmax = Symbol.SoftmaxOutput(Map("data" -> fc, "name" -> "sm"))
//    println(softmax.listArguments())
//
//    val lhsArr = Random.uniform(-10f, 10f, shape)
//    val rhsArr = Random.uniform(-10f, 10f, shape)
//    val lhsGrad = NDArray.empty(shape)
//    val rhsGrad = NDArray.empty(shape)
//    
//    val ctxMapKeys = ArrayBuffer.empty[String]
//    val ctxMapDevTypes = ArrayBuffer.empty[Int]
//    val ctxMapDevIDs = ArrayBuffer.empty[Int]
//		
//    val args = Array(lhsArr, rhsArr)
//    val argsGrad = Array(lhsGrad, rhsGrad)
//    
//    val execHandle = new ExecutorHandleRef
//    println("++++++++++++++++++++++++++++++++++++++++++++++")
// 	checkCall(_LIB.mxExecutorBindX(sum.handle,
//                                   1,//1
//                                   0,//0
//                                   ctxMapKeys.size,//0
//                                   ctxMapKeys.toArray,//null
//                                   ctxMapDevTypes.toArray,//null
//                                   ctxMapDevIDs.toArray,//null
//                                   args.size,
//                                   args.map(_.handle),
//                                   argsGrad.map(_.handle),
//                                   Array(1,1),
//                                   new Array[NDArrayHandle](0),
//                                   execHandle))
//
//	val executor = new Executor(execHandle.value,sum)
//		
////    val exec3 = ret.bind(Context.cpu(), args = Seq(lhsArr, rhsArr))
////    val exec4 = ret.bind(Context.cpu(), args = Map("rhs" -> rhsArr, "lhs" -> lhsArr),
////                         argsGrad = Map("lhs" -> lhsGrad, "rhs" -> rhsGrad))
//    val exec5 = softmax.simpleBind(Context.cpu(), "write", shapeDict = Map("rhs" ->shape,"rhs" -> shape))
//
//    println("++++++++++++++++++++++++++++++++++++++++++++++")
//    executor.forward()
//    exec5.forward(true)
//
//    println("++++++++++++++++++++++++++++++++++++++++++++++")
//    println(executor.outputs(0))
//    exec5.outputs.foreach { println }
////    val outGrad = Random.uniform(-10f, 10f, shape)
////    executor.backward(Array(outGrad))
//    exec5.backward()
//    println("++++++++++++++++++++++++++++++++++++++++++++++")
//
////    println(outGrad)
//    
//	}
//  
//  
//  def ElementWiseSumTest{
//    val data = Symbol.Variable("data")
//    val data1 = Symbol.Variable("data1")
//    val lat = Symbol.ElementWiseSum(Array(data,data1),"lateralCon")
//    println(lat.debugStr)
//  }
//  
//}