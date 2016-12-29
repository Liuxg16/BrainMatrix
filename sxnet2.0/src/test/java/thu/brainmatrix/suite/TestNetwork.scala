package thu.brainmatrix.suite

import thu.brainmatrix.Base._
import thu.brainmatrix.StaticGraph
import thu.brainmatrix.Symbol
import thu.brainmatrix.NDArray
import thu.brainmatrix.Executor
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import thu.brainmatrix.FeedForward
import thu.brainmatrix.Symbol
import thu.brainmatrix.Shape
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.Context
import thu.brainmatrix.IO
import thu.brainmatrix.Random
import thu.brainmatrix.Context.ctx2Array

/**
 * 2016-4-1
 */
object TestNetwork {
  
	def main(args:Array[String]){
//		simpleNNTest
//		simpleNN_model
//		simpleNNTest_mxnet
//		simpleNNBackwardTest
//		simpleNNBackwardTest_2
//		simpleNNTrainingTest
//		simpleBindingTest
		
//		mlp_test
		bindTest
	}
	
	
	
	
	def simpleNNForwardTest{
//		val dataS = Symbol.CreateVariable("data")
//		
//  	val kwargs_type = Map("name" -> "fc1", "num_hidden" -> "12")
//    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
//  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
//  	sb.Compose(kwargs_symbol, "fc1")
//	
////  	val sm= Symbol.Create("softmaxOutput", kwargs)
//  	
////  	var  out_graph= new StaticGraph()
//  	sb.ToStaticGraph()
// 		println(sb.staticGraph.debug)
//  	println("\n---------------------------------------------------")
////  	checkCall(out_graph.ToStaticGraph)
//		val kwargs_shape = Map("data"->Shape(200,15))
//  	
//    val (argShapes, outShapes, auxShapes) = sb.inferShape1(sb.staticGraph,kwargs_shape)
//    argShapes.foreach {println}
//  	outShapes.foreach {println}
//  	
//  	
//  	val data = NDArray.ones(Shape(200,15))
//  	val weight = NDArray.ones(Shape(12,15))//according to inferShape function
//  	val bias = NDArray.ones(Shape(12))//according to inferShape function
////  	val label = NDArray.ones(Shape(200,12))
//  	
//  	val data_grad = NDArray.ones(Shape(200,15))
//  	val weight_grad = NDArray.ones(Shape(12,15))//according to inferShape function
//  	val bias_grad = NDArray.ones(Shape(12))//according to inferShape function
//  	
//  	val in_args: Array[NDArray] = Array(data, weight, bias)
//    val arg_grad_store: Array[NDArray] = Array(data_grad, weight_grad, bias_grad)
//  	val grad_req_type: Array[Int] = Array(1,1,1)
//  	
//  	
//  	val ctxMapKeys = ArrayBuffer.empty[String]
//    val ctxMapDevTypes = ArrayBuffer.empty[Int]
//    val ctxMapDevIDs = ArrayBuffer.empty[Int]
//
//  	val execHandle = new ExecutorHandleRef
//  	
//  	println("---------------------binding-----------------------")
//    checkCall(_LIB.mxScalaExecutorBindX(sb.staticGraph.handle,
//                                   1,//1
//                                   0,//0
//                                   ctxMapKeys.size,//0
//                                   ctxMapKeys.toArray,//null
//                                   ctxMapDevTypes.toArray,//null
//                                   ctxMapDevIDs.toArray,//null
//                                   in_args.size,
//                                   in_args.map(_.handle),
//                                   arg_grad_store.map(_.handle),
//                                   grad_req_type,
//                                   new Array[NDArrayHandle](0),
//                                   execHandle))
//                                   
//    println("---------------------executor-----------------------")
//    val executor = new Executor(execHandle.value, null)
//  	println("---------------------froward-----------------------")
//  	executor.forward()
//  	println("---------------------output-----------------------")
//  	executor.outputs.foreach {println}
//  	
	}
	
//	succeed!
//	def simpleNNBackwardTest{
//		val dataS = Symbol.CreateVariable("data")
//		
//  	val kwargs_type = Map("name" -> "fc1", "num_hidden" -> "6")
//    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
//  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
//  	sb.Compose(kwargs_symbol, "fc1")
//	
////  	val sm= Symbol.Create("softmaxOutput", kwargs)
//  	
//  	var  out_graph= new StaticGraph()
//  	sb.ToStaticGraph()
// 		println(sb.staticGraph.debug)
//  	println("\n---------------------------------------------------")
////  	checkCall(out_graph.ToStaticGraph)
//		val kwargs_shape = Map("data"->Shape(15,10))
//  	
//    val (argShapes, outShapes, auxShapes) = sb.inferShape1(sb.staticGraph,kwargs_shape)
//    argShapes.foreach {println}
//  	outShapes.foreach {println}
//  	
//  	
//  	val data = NDArray.ones(Shape(15,10))
//  	val weight = NDArray.ones(Shape(6,10))//according to inferShape function
//  	val bias = NDArray.ones(Shape(6))//according to inferShape function
////  	val label = NDArray.ones(Shape(200,12))
//  	
//  	val data_grad = NDArray.ones(Shape(15,10))
//  	val weight_grad = NDArray.ones(Shape(6,10))//according to inferShape function
//  	val bias_grad = NDArray.ones(Shape(6))//according to inferShape function
//  	
//  	val in_args: Array[NDArray] = Array(data, weight, bias)
//    val arg_grad_store: Array[NDArray] = Array(data_grad, weight_grad, bias_grad)
////  	 val arg_grad_store: Array[NDArray] = Array(new NDArray(0), weight_grad, bias_grad)
//  	val grad_req_type: Array[Int] = Array(0,1,1)
//  	
//  	
//  	val ctxMapKeys = ArrayBuffer.empty[String]
//    val ctxMapDevTypes = ArrayBuffer.empty[Int]
//    val ctxMapDevIDs = ArrayBuffer.empty[Int]
//
//  	val execHandle = new ExecutorHandleRef
//  	
//  	println("---------------------binding-----------------------")
//    checkCall(_LIB.mxScalaExecutorBindX(out_graph.handle,
//                                   1,//1
//                                   0,//0
//                                   ctxMapKeys.size,//0
//                                   ctxMapKeys.toArray,//null
//                                   ctxMapDevTypes.toArray,//null
//                                   ctxMapDevIDs.toArray,//null
//                                   in_args.size,
//                                   in_args.map(_.handle),
//                                   arg_grad_store.map(_.handle),
//                                   grad_req_type,
//                                   new Array[NDArrayHandle](0),
//                                   execHandle))
//                                   
//    println("---------------------executor-----------------------")
//    val executor = new Executor(execHandle.value, null)
//  	println("---------------------froward-----------------------")
//  	executor.forward()
//  	println("---------------------output-----------------------")
////  	executor.outputs.foreach {println}
//  	println(executor.outputs(0))
//  	
//  	println("---------------------backward-----------------------")
//  	val outGrad = Random.uniform(-10f, 10f, Shape(15,5))
//  	executor.backward(Array(outGrad))
//  	println(outGrad)
//  	println(data_grad)
////  	
//	}
	
	
	//	succeed!
	def simpleNNTrainingTest{
		
		val num_instance = 10
    val input_dim = 15
		val dataS = Symbol.CreateVariable("data")
		
		val hidden_1 = 6
  	val kwargs_type = Map("name" -> "fc1", "num_hidden" -> (""+hidden_1))
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "fc1")
	
//  	val sm= Symbol.Create("softmaxOutput", kwargs)
 
  	
  	
  	sb.ToStaticGraph()
 		println(sb.staticGraph.debug)
  	sb.staticGraph.ToStaticGraph
  	
//  	checkCall(out_graph.ToStaticGraph)
		val kwargs_shape = Map("data"->Shape(num_instance,input_dim))
  	
    val (argShapes, outShapes, auxShapes) = sb.inferShape(kwargs_shape)
    argShapes.foreach {println}
  	outShapes.foreach {println}
  	
  	
    
  	val data = NDArray.ones(Shape(num_instance,input_dim))
  	val weight = NDArray.ones(Shape(hidden_1,input_dim))//according to inferShape function
  	val bias = NDArray.ones(Shape(hidden_1))//according to inferShape function

  	
    
   

  	println("\n---------------------------------------------------")
 
  	val data_grad = NDArray.ones(Shape(num_instance,input_dim))

  	val weight_grad = NDArray.ones(Shape(hidden_1,input_dim))//according to inferShape function
  		
  	val bias_grad = NDArray.ones(Shape(hidden_1))//according to inferShape function
  
  	val in_args: Array[NDArray] = Array(data, weight, bias)
    val arg_grad_store: Array[NDArray] = Array(data_grad, weight_grad, bias_grad)
//  	 val arg_grad_store: Array[NDArray] = Array(new NDArray(0), weight_grad, bias_grad)
  	val grad_req_type: Array[Int] = Array(0,1,1)
  	
  	
  	val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]

  	val execHandle = new ExecutorHandleRef
  	
  	println("---------------------binding-----------------------")
    checkCall(_LIB.mxScalaExecutorBindX(sb.staticGraph.handle,
                                   1,//1
                                   0,//0
                                   ctxMapKeys.size,//0
                                   ctxMapKeys.toArray,//null
                                   ctxMapDevTypes.toArray,//null
                                   ctxMapDevIDs.toArray,//null
                                   in_args.size,
                                   in_args.map(_.handle),
                                   arg_grad_store.map(_.handle),
                                   grad_req_type,
                                   new Array[NDArrayHandle](0),
                                   execHandle))
                                   
    println("---------------------executor-----------------------")
    val executor = new Executor(execHandle.value, null)
  	println("---------------------froward-----------------------")
  	executor.forward()
  	println("---------------------output-----------------------")
//  	executor.outputs.foreach {println}
  	println(executor.outputs(0))
  	
  	println("---------------------backward-----------------------")
  	val outGrad = Random.uniform(-10f, 10f, Shape(num_instance,hidden_1))
  	executor.backward(Array(outGrad))
  	println(outGrad)
  	println(data_grad)
  	println(weight_grad)
//  	
	}
	
	
	
	
	//	succeed!
	def simpleBindingTest{
		
		val num_instance = 10
    val input_dim = 15
		val dataS = Symbol.CreateVariable("data")
		
	
		
		
		val hidden_1 = 6
  	val kwargs_type = Map("name" -> "fc1", "num_hidden" -> (""+hidden_1))
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "fc1")
	
  	 
  	
//  	val sm= Symbol.Create("softmaxOutput", kwargs)
  	
  	sb.ToStaticGraph()
 		println(sb.staticGraph.debug)
  	sb.staticGraph.ToStaticGraph
  	
  	
//  	val argNDArrays = (argShapes) map { case shape =>
//      // TODO: NDArray dtype
//      NDArray.zeros(shape, ctx)
//    }
//  	checkCall(out_graph.ToStaticGraph)
		val kwargs_shape = Map("data"->Shape(num_instance,input_dim))
  	
		val (argShapes, _, auxShapes) = sb.inferShape(kwargs_shape)
    
    argShapes.foreach {println}
    require(argShapes != null, "Input node is not complete")
    // alloc space
    val argNDArrays = (argShapes) map { case shape =>
      // TODO: NDArray dtype
      NDArray.ones(shape)
    }
     val gradNDArrays =(argShapes zipWithIndex) map { case (shape,idx) =>
      // TODO: NDArray dtype
    	if(idx!=0 &&idx !=argShapes.size-1 ){
    			NDArray.ones(shape)	
    	}else{
    			new NDArray(0)
    	}
    }
		
 

  	
  	
    
  	val data = NDArray.ones(Shape(num_instance,input_dim))
  	val weight = NDArray.ones(Shape(hidden_1,input_dim))//according to inferShape function
  	val bias = NDArray.ones(Shape(hidden_1))//according to inferShape function

  	
    
   

  	println("\n---------------------------------------------------")
 
  	val data_grad = NDArray.ones(Shape(num_instance,input_dim))

  	val weight_grad = NDArray.ones(Shape(hidden_1,input_dim))//according to inferShape function
  		
  	val bias_grad = NDArray.ones(Shape(hidden_1))//according to inferShape function
  
  	val in_args: Array[NDArray] = Array(data, weight, bias)
    val arg_grad_store: Array[NDArray] = Array(data_grad, weight_grad, bias_grad)
//  	 val arg_grad_store: Array[NDArray] = Array(new NDArray(0), weight_grad, bias_grad)
  	val grad_req_type: Array[Int] = Array(1,1,1)
  	
  	
  	val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]

  	val execHandle = new ExecutorHandleRef
  	
  	println("---------------------binding-----------------------")
    checkCall(_LIB.mxScalaExecutorBindX(sb.staticGraph.handle,
                                   1,//1
                                   0,//0
                                   ctxMapKeys.size,//0
                                   ctxMapKeys.toArray,//null
                                   ctxMapDevTypes.toArray,//null
                                   ctxMapDevIDs.toArray,//null
//                                   in_args.size,
//                                   in_args.map(_.handle),
//                                   arg_grad_store.map(_.handle),
                                   argNDArrays.size,
                                   argNDArrays.map(_.handle).toArray,
                                   gradNDArrays.map{_.handle}.toArray,
                                   grad_req_type,
                                   new Array[NDArrayHandle](0),
                                   execHandle))
                                   
    println("---------------------executor-----------------------")
    val executor = new Executor(execHandle.value, null)
  	println("---------------------froward-----------------------")
  	executor.forward()
  	println("---------------------output-----------------------")
//  	executor.outputs.foreach {println}
  	println(executor.outputs(0))
  	
  	println("---------------------backward-----------------------")
  	val outGrad = Random.uniform(-10f, 10f, Shape(num_instance,hidden_1))
  	executor.backward(Array(outGrad))
  	println(outGrad)
  	println(data_grad)
  	println(weight_grad)
//  	
	}
	
	
	
	
	
	/**
	 * 
	 *by liuxianggen
	 * 2016-4-5
	 * 
	 */
	def simpleNNBackwardTest_2{
		val dataS = Symbol.CreateVariable("data")
		
    val sb:Symbol = Symbol.Create("FullyConnected",Map("name" -> "fc1", "num_hidden" -> "5"))
  	sb.Compose(Map("data"->dataS) , "fc1")
  	
  	val sb1:Symbol = Symbol.Create("FullyConnected",Map("name" -> "fc2", "num_hidden" -> "5"))
  	sb1.Compose(Map("data"->sb) , "fc2")
	
//  	val act1:Symbol = Symbol.Create("Activation",Map("name" -> "relu2", "act_type" -> "relu"))
//  	act1.Compose(Map("data"->sb1) , "relu2")
//  	
  	val kwargs_type_sm = Map("name" -> "sm")
  	val sm= Symbol.Create("SoftmaxOutput",Map("grad_scale"->"1"))
  	sm.Compose(Map("data" -> sb1), "sm")
//  	
////  	val act = Symbol.Create("Activation",Map("name" -> "act", "act_type" -> "relu"))
////  	act.Compose(Map("data"->sb),"act")
//  	
//  	
//  	sm.ToStaticGraph()
// 		println(sb.staticGraph.debug)
// 		sb.staticGraph.ToStaticGraph
//  	println("\n---------------------------------------------------")
//		val kwargs_shape = Map("data"->Shape(15,10))
//    val (argShapes, outShapes, auxShapes) = sm.inferShape1(out_graph,kwargs_shape)
//    argShapes.foreach {println}
//  	outShapes.foreach {println}
  	
//  	val num=15
//  	val label = NDArray.zeros(Shape(num))
//  	for(i <- 0 until num){
//  		val temp = (i/3).floor
//  		println(temp)
//  		label(i) = temp
//  		println(label)
//  	}
//  
//  	
  	val num_instance = 15
  	val input_dim = 10
  	val data = NDArray.rangeRows(0, num_instance, input_dim)//num_instance,10
//  	val data = Random.uniform(-10f, 10f, Shape(num_instance,input_dim))
  	val label = NDArray.range(0,5,3)
//  	val data =NDArray.ones(Shape(num_instance,10))
//  	val label = NDArray.range(num_instance)
//  	
//  	for (i <- 0 until num_instance) {
//      for (j <- 0 until input_dim) {
//        data(i, j) = i * 1.0f + (scala.util.Random.nextFloat - 0.5f)
//      }
//    
//    }
		println(data)
  	println(label)
  	
  	
  	val weight = NDArray.ones(Shape(5,10))//according to inferShape function
  	val bias = NDArray.ones(Shape(5))//according to inferShape function
  	
  	val weight1 = NDArray.ones(Shape(5,5))//according to inferShape function
  	val bias1 = NDArray.ones(Shape(5))//according to inferShape function
  	
  	
  	var data_grad = NDArray.ones(Shape(num_instance,10))
  	var weight_grad = NDArray.ones(Shape(5,10))//according to inferShape function
  	var bias_grad = NDArray.ones(Shape(5))//according to inferShape function
  	
  	var weight_grad1 = NDArray.ones(Shape(5,5))//according to inferShape function
  	var bias_grad1 = NDArray.ones(Shape(5))//according to inferShape function
  	
  	var label_grad = NDArray.ones(Shape(num_instance))
  	
  	val in_args: Array[NDArray] = Array(data, weight, bias,weight1, bias1,label)
//    var arg_grad_store: Array[NDArray] = Array(data_grad, weight_grad, bias_grad,label_grad)
//  	 	val in_args: Array[NDArray] = Array(data, weight, bias)
//    	val arg_grad_store: Array[NDArray] = Array(data_grad, weight_grad, bias_grad)
  	val arg_grad_store: Array[NDArray] = Array(new NDArray(0), weight_grad, bias_grad,weight_grad1, bias_grad1,new NDArray(0))
  	val grad_req_type: Array[Int] = Array(0,1,1,1,1,0)
  	
  	
//  	val executor = sm.bindHelper(in_args, arg_grad_store, grad_req_type)
//
//  	println("---------------------froward-----------------------")
////  	executor.forward()
//  	println("---------------------output-----------------------")
////  	executor.outputs.foreach {println}
////  	println(executor.outputs(0))
//  	println("---------------------backward-----------------------")
//  	val outGrad = Random.uniform(-10f, 10f, Shape(15,6))
//  	executor.backward()
//  	checkCall(_LIB.mxExecutorBackward(executor.handle, Array(outGrad.handle)))
//  	executor.backward()
//  	println(data)
//  	println(label)
  	
//  	for(i<- 0 until 10){
//  		 println("epoch:"+i)
//			 	executor.forward()
//			 	executor.backward()
//			 	println(executor.outputs(0))
//			 	val acc: Float = output_accuracy(executor.outputs(0), label)
//			  Console.println("Accuracy: " + acc)
//			  println(arg_grad_store(2))
////			  println(in_args(2))
//			  for (j <- 1 to 4) {
//        arg_grad_store(j) *= 5*1e-3f
//        in_args(j) -= arg_grad_store(j)
//        
//      }
//  	 }
////  	executor.forward()
////  	executor.backward()
////  	println(outGrad)
////  	println(data_grad)
////  	println(weight_grad)
//
//		executor.dispose()
	}
	
	
	
	
	
	
	def simpleNNTest_mxnet{
		
		val datas = Symbol.CreateVariable("data")
		val fc1 = Symbol.FullyConnected()(Map("data" -> datas, "name" -> "fc1", "num_hidden" -> 10))
  
  	
//val kwargs_shape = scala.collection.immutable.Map("data"->Shape(200,15))
//val (arg,out,aux) = sm.inferShape(kwargs_shape)
//  	println(sm.listArguments())
//  	ArrayBuffer(data, fc1_weight, fc1_bias, sm_label)
//  	arg.foreach { println}
//		Shape(200, 15)
//Shape(10, 15)
//Shape(10)
//Shape(200)
  	
  	
  	val data = NDArray.ones(Shape(200,15))
  	val weight = NDArray.ones(Shape(12,15))//according to inferShape function
  	val bias = NDArray.ones(Shape(12))//according to inferShape function
//  	val label = NDArray.ones(Shape(200))
  	
  	val data_grad = NDArray.ones(Shape(200,15))
  	val weight_grad = NDArray.ones(Shape(12,15))//according to inferShape function
  	val bias_grad = NDArray.ones(Shape(12))//according to inferShape function
  	
  	val in_args: Array[NDArray] = Array(data, weight, bias)
//  		val in_args: Array[NDArray] = Array(data, weight, bias)
    val arg_grad_store: Array[NDArray] = Array(data_grad,weight_grad, bias_grad)
  	val grad_req_type: Array[Int] = Array(1,1,1)
  	
  	
  	val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]
//
//  	
//  	
////  	println(bias.toString())
  	val execHandle = new ExecutorHandleRef
//  	
//  	println("---------------------binding-----------------------")
////  	LIB.mxExecutorBind(out_graph.handle,1, 0, in_args.length, in_args.map(_.handle), arg_grad_store.map(_.handle),
////      grad_req_type, 0, new Array[NDArrayHandle](0), out)
    checkCall(_LIB.mxExecutorBindX(fc1.handle,
                                   1,//1
                                   0,//0
                                   ctxMapKeys.size,//0
                                   ctxMapKeys.toArray,//null
                                   ctxMapDevTypes.toArray,//null
                                   ctxMapDevIDs.toArray,//null
                                   in_args.size,
                                   in_args.map(_.handle),
                                   arg_grad_store.map(_.handle),
                                   grad_req_type,
                                   new Array[NDArrayHandle](0),
                                   execHandle))
//		
////	
//                                   
//    println("---------------------executor-----------------------")
//    val executor = new Executor(execHandle.value, fc1)
//		  	println("---------------------froward-----------------------")
//  	executor.forward()
//  	println("---------------------output-----------------------")
//  	executor.outputs.foreach {println}
//		
//  	
	}
	
	
	 def mlp_test(): Unit = {
    val nh1:Int = 20
    val nh2:Int = 10
    val input_dim = 28
    val num_instance = 20
    
    val input = Symbol.CreateVariable("input")
    val fc1 = Symbol.FullyConnected()(Map("data" -> input, "name" -> "fc1", "num_hidden" -> nh1))
    val relu1 = Symbol.Activation()(Map("data" -> fc1, "act_type" -> "relu"))
    // relu1.listArguments().foreach {println}
    val fc2 = Symbol.FullyConnected()(Map("data" -> relu1, "name" -> "fc2", "num_hidden" -> nh2))
    // fc2.listArguments().foreach(println)
    val output = Symbol.SoftmaxOutput()(Map("data" -> fc2, "name" -> "out"))
//    output.listArguments().foreach(println)
    
    val (arg,out,aux) = output.inferShape(scala.collection.immutable.Map("input"->Shape(num_instance, input_dim)))
//  	
  	arg.foreach { println}
    println("---------------------------------------------------------")
    out.foreach { println}

 
    val arr_x = NDArray.zeros(num_instance, input_dim)//128,28
    val arr_y = NDArray.zeros(num_instance)//128

    for (i <- 0 until num_instance) {
      for (j <- 0 until input_dim) {
        arr_x(i, j) = i % 10 * 1.0f + (scala.util.Random.nextFloat - 0.5f)
      }
      arr_y(i) = i % 10
    }

    // Console.println(arr_x)

    val arr_W1 = Random.normal(0f, 1f, Shape(nh1, input_dim))//
    val arr_b1 = NDArray.zeros(nh1)
    val arr_W2 = Random.normal(0f, 1f, Shape(nh2, nh1))
    val arr_b2 = NDArray.zeros(nh2)
    //
    val arr_W1_g = NDArray.zeros(nh1, input_dim)
    val arr_b1_g = NDArray.zeros(nh1)
    val arr_W2_g = NDArray.zeros(nh2, nh1)
    val arr_b2_g = NDArray.zeros(nh2)
    //
    val in_args: Array[NDArray] = Array(arr_x, arr_W1, arr_b1, arr_W2, arr_b2, arr_y)
    val arg_grad_store: Array[NDArray] = Array(NDArray.zeros(1), arr_W1_g, arr_b1_g, arr_W2_g, arr_b2_g, NDArray.zeros(1))
    val grad_req_type: Array[Int] = Array(0, 1, 1, 1, 1, 0)

//    val executor = output.bind(in_args, arg_grad_store, grad_req_type)
		val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]
//
//  	
//  	
////  	println(bias.toString())
  	val execHandle = new ExecutorHandleRef
    checkCall(_LIB.mxExecutorBindX(output.handle,
                                   1,//1
                                   0,//0
                                   ctxMapKeys.size,//0
                                   ctxMapKeys.toArray,//null
                                   ctxMapDevTypes.toArray,//null
                                   ctxMapDevIDs.toArray,//null
                                   in_args.size,
                                   in_args.map(_.handle),
                                   arg_grad_store.map(_.handle),
                                   grad_req_type,
                                   new Array[NDArrayHandle](0),
                                   execHandle))
     val executor = new Executor(execHandle.value, fc1)
    
    Console.println("Training ...")

    // val max_iters = 12001
    // val learning_rate = 0.00001f
    val max_iters = 20
    val learning_rate = 0.0001f
    	
    val grad = NDArray.ones(Shape(3,5))
    
    for (iter <- 0 until max_iters+1) {
      executor.forward(true)
      if (iter % 1 == 0) {
        Console.println("epoch " + iter)
        val acc: Float = output_accuracy(executor.outputs(0), arr_y)
        Console.println("Accuracy: " + acc)
      }
      executor.backward(grad)
      for (i <- 1 to 4) {
        arg_grad_store(i) *= learning_rate
        in_args(i) -= arg_grad_store(i)
      }
    }
  }

  def output_accuracy(pred: NDArray, target: NDArray): Float = {
    val num_instance = pred.shape(0)
    val eps = 1e-6
    var right = 0
    for (i <- 0 until num_instance) {
      var mx_p = pred(i, 0)
      var p_y: Float = 0
      for(j <- 0 until 5){
        if(pred(i,j) > mx_p){
          mx_p = pred(i,j)
          p_y = j
        }
      }
      if(scala.math.abs(p_y - target(i)) < eps) right += 1
    }
    right * 1.0f / num_instance
  }
	
	def bindTest{
		val shape = Shape(10, 5)
    val lhs = Symbol.CreateVariable("lhs")
    val rhs = Symbol.CreateVariable("rhs")
    val ret = lhs + rhs
    println(ret.listArguments())
//    require(ret.listArguments().toArray == Array("lhs", "rhs"))

    val lhsArr = Random.uniform(-10f, 10f, shape)
    val rhsArr = Random.uniform(-10f, 10f, shape)
    val lhsGrad = NDArray.empty(shape)
    val rhsGrad = NDArray.empty(shape)
    
    val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]
    val args = Array(lhsArr, rhsArr)
    val argsGrad = Array(lhsGrad, rhsGrad)
    val execHandle = new ExecutorHandleRef
 		checkCall(_LIB.mxExecutorBindX(ret.handle,
                                   1,//1
                                   0,//0
                                   ctxMapKeys.size,//0
                                   ctxMapKeys.toArray,//null
                                   ctxMapDevTypes.toArray,//null
                                   ctxMapDevIDs.toArray,//null
                                   args.size,
                                   args.map(_.handle),
                                   argsGrad.map(_.handle),
                                   Array(1,1),
                                   new Array[NDArrayHandle](0),
                                   execHandle))
		val executor = new Executor(execHandle.value, ret)

		val exec3 = ret.bind(Context.cpu(), args = Seq(lhsArr, rhsArr))
    executor.forward()
    exec3.forward()

    val out1 = lhsArr + rhsArr
    val out2 = executor.outputs(0)

    // test gradient
    val outGrad = NDArray.ones(shape)
    executor.backward(Array(outGrad))
  	println(lhsGrad-outGrad)
	}
}