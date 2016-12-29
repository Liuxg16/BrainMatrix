package thu.brainmatrix.suite

import thu.brainmatrix.Base._
import thu.brainmatrix._
import thu.brainmatrix.optimizer.SGD
import scala.collection.mutable.Stack
import scala.collection.mutable.ArrayBuffer
import scala.Vector

import org.scalatest.{BeforeAndAfterAll, FunSuite}
/**
 * 2016-3-22
 * by liuxianggen
 */
class SymbolSuite extends FunSuite with BeforeAndAfterAll{
  	
	
	
	
	
	
	/**
	 * test gradient
	 */
	
	test("sig gradient"){
		val rhs = Symbol.CreateVariable("rhs")
      	val lhs = Symbol.CreateVariable("lhs")
      	val dot = Symbol.FullyConnected("dot")(Map("data"->rhs,"weight"->lhs,"no_bias"->true,"num_hidden"->4))
      	val res = Symbol.Activation("sig")(Map("data"->dot,"act_type" -> "sigmoid"))
      	val lshape = Shape(4,3)
      	val rshape = Shape(2,3)
//      	res.listArguments().foreach(println)
      	val (a,b,c) = res.inferShape(Map("rhs"->rshape))
      	a.foreach {x => println(x)}
      	val rhsArr = NDArray.array(Array(1,0,1,-2,1,0),rshape)
      	val lhsArr = NDArray.array(Array(1,2,3,4,5,6,1,2,3,4,5,6),lshape)	
      	println("rhsArr:"+rhsArr)
      	println("lhsArr:"+lhsArr)
//      	println("sigmoid rhsArr:"+NDArray.sigmod(rhsArr))
//      	println("sigmoid lhsArr:"+NDArray.sigmod(lhsArr))
      	val rhsArr_g = NDArray.zeros(rshape)
      	val lhsArr_g = NDArray.zeros(lshape)	

      	val executor = res.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr),argsGrad = Map("lhs"->lhsArr_g,"rhs"->rhsArr_g))
      	executor.forward(isTrain=true)
      	val out2 = executor.outputs(0)
      	println(out2)
      	val error = NDArray.array(Array(2,0,0,0,0,-1,0,0),Shape(2,4))
//      	val error = NDArray.ones(Shape(2,4))
      	
      	
      	
      	println("errro:"+error)
      	executor.backward(error)
      	
      	val resarr = NDArray.dot(rhsArr,NDArray.transpose(lhsArr))
      	val temp = error*(NDArray.sigmod(resarr)*(NDArray.sigmod(resarr)*(-1)+1))
      	println("sigmoid gradient:" + NDArray.dot(temp, lhsArr))
      	println("-------------------------------")
      	executor.gradArrays.foreach {println}
	}
	
	
	test("mul gradient"){
		val rhs = Symbol.CreateVariable("rhs")
      	val lhs = Symbol.CreateVariable("lhs")
//      	val res = Symbol.FullyConnected("dot")(Map("data"->rhs,"weight"->lhs,"no_bias"->true,"num_hidden"->4))
      	val res = rhs * lhs 
      	 
      	val lshape = Shape(2,3)
//      	res.listArguments().foreach(println)
//      	val (a,b,c) = res.inferShape(Map("rhs"->lshape))
//      	a.foreach {x => println(x)}
      	val rhsArr = NDArray.array(Array(10,0,1,-2,1,0),lshape)
      	val lhsArr = NDArray.array(Array(1,2,3,4,5,6),lshape)	
      	println("rhsArr:"+rhsArr)
      	println("lhsArr:"+lhsArr)
      	val rhsArr_g = NDArray.zeros(lshape)
      	val lhsArr_g = NDArray.zeros(lshape)	

      	val executor = res.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr),argsGrad = Map("lhs"->lhsArr_g,"rhs"->rhsArr_g))
      	executor.forward(isTrain=true)
      	val out2 = executor.outputs(0)
      	println(out2)
      	val error = NDArray.array(Array(2,0,0,0,-1,0),Shape(2,3))
//      	val error = NDArray.ones(Shape(2,4))
      	
      	println("errro:"+error)
      	executor.backward(error)
      	println("-------------------------------")
      	executor.gradArrays.foreach {println}
	}
	
	
	
	test("add gradient"){
		val rhs = Symbol.CreateVariable("rhs")
      	val lhs = Symbol.CreateVariable("lhs")
//      	val res = Symbol.FullyConnected("dot")(Map("data"->rhs,"weight"->lhs,"no_bias"->true,"num_hidden"->4))
      	val res = rhs+ lhs * 2
      	 
      	val lshape = Shape(2,3)
//      	res.listArguments().foreach(println)
//      	val (a,b,c) = res.inferShape(Map("rhs"->lshape))
//      	a.foreach {x => println(x)}
      	val rhsArr = NDArray.array(Array(1,0,1,-2,1,0),lshape)
      	val lhsArr = NDArray.array(Array(1,2,3,4,5,6),lshape)	
      	println("rhsArr:"+rhsArr)
      	println("lhsArr:"+lhsArr)
      	val rhsArr_g = NDArray.zeros(lshape)
      	val lhsArr_g = NDArray.zeros(lshape)	

      	val executor = res.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr),argsGrad = Map("lhs"->lhsArr_g,"rhs"->rhsArr_g))
      	executor.forward(isTrain=true)
      	val out2 = executor.outputs(0)
      	println(out2)
      	val error = NDArray.diag(Shape(2,3))
//      	val error = NDArray.ones(Shape(2,4))
      	
      	println("errro:"+error)
      	executor.backward(error)
      	println("-------------------------------")
      	executor.gradArrays.foreach {println}
	}
	
	
	test("dot gradient"){
		val rhs = Symbol.CreateVariable("rhs")
      	val lhs = Symbol.CreateVariable("lhs")
      	val res = Symbol.FullyConnected("dot")(Map("data"->rhs,"weight"->Symbol.transpose(lhs),"no_bias"->true,"num_hidden"->4))
      	 
//      	val lshape = Shape(4,3)
      	val lshape = Shape(3,4)
      	val rshape = Shape(2,3)
//      	res.listArguments().foreach(println)
//      	val (a,b,c) = res.inferShape(Map("rhs"->rshape))
//      	a.foreach {x => println(x)}
      	val rhsArr = NDArray.array(Array(1,0,1,-2,1,0),rshape)
      	val lhsArr = NDArray.array(Array(1,2,3,4,5,6,1,2,3,4,5,6),lshape)	
      	println("rhsArr:"+rhsArr)
      	println("lhsArr:"+lhsArr)
      	val rhsArr_g = NDArray.zeros(rshape)
      	val lhsArr_g = NDArray.zeros(lshape)	
      	
      	println("ddd")
      	
      	
      	val executor = res.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr),argsGrad = Map("lhs"->lhsArr_g,"rhs"->rhsArr_g))
      	executor.forward(isTrain=true)
      	val out2 = executor.outputs(0)
      	println(out2)
      	val error = NDArray.array(Array(2,0,0,0,0,-1,0,0),Shape(2,4))
//      	val error = NDArray.ones(Shape(2,4))
      	
      	println("errro:"+error)
      	executor.backward(error)
      	println("-------------------------------")
      	executor.gradArrays.foreach {println}
	}
	
	
	/**
   	* operation
   	*/
	
	/**
	 * square
	 */
	test("square"){
		val shape = Shape(3, 4)
	    val lhs = Symbol.CreateVariable("lhs")
	    val res = Symbol.square(lhs)
	    val lhsArr = NDArray.ones(shape)*2
	    lhsArr(1,1) = 4
	    val executor = res.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr))
	    executor.forward()
	    val out2 = executor.outputs(0)
	    println(out2)
	}
	
	
	
    /**
     * by liuxianggen
     * 20160825
     * there are to steps:
     * 1.softmax
     * 2.sum{log(p{label(i)})}
     * Calculate cross_entropy(lhs, one_hot(rhs))
		Parameters
		----------
		lhs : Symbol
		Left symbolic input to the function
		rhs : Symbol
		Right symbolic input to the function
    */
	
	test("Softmax_cross_entropy"){
		val shape = Shape(4,2)
	    val lhs = Symbol.CreateVariable("lhs")
	    val weight = Symbol.CreateVariable("weight")
	    val fully = Symbol.FullyConnected("f")(Map("data"->lhs,"weight"-> weight,"no_bias"-> true,"num_hidden"->4))
	    val rhs = Symbol.CreateVariable("rhs")
	    
	    val sum = Symbol.Softmax_cross_entropy(fully,rhs)
	    
//	    val sum = Symbol.sum(fully)
	    val weightArr = Random.normal(0f,1f,Shape(4,2))
	    val lhsArr = NDArray.array(Array(1f,2f,3f,1f),Shape(2,2),Context.defaultCtx)
	    val rhsArr = NDArray.ones(Shape(2),Context.defaultCtx)
	    
	    val weightArr_g = NDArray.zeros(Shape(4,2))
	    val gradDict = Map("weight"->weightArr_g)
	    val executor = sum.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr,"weight"->weightArr),argsGrad = gradDict)
	    println("num:"+executor.outputs.length)
	    var out2 = executor.outputs(0)
	    println(out2)	
	    val gradarr = NDArray.array(Array(1f),Shape(1),Context.defaultCtx) 
	    executor.backward(executor.outputs(0))
	    println("ddd")
	    executor.gradArrays.foreach {println}	
	}
	
	
	/**
	 * network backward
	 */
	
	test("for-back-network"){
		val shape = Shape(4,2)
	    val lhs = Symbol.CreateVariable("lhs")
	    val weight = Symbol.CreateVariable("weight")
	    val fully = Symbol.FullyConnected("f")(Map("data"->lhs,"weight"-> weight,"no_bias"-> true,"num_hidden"->10))
	    val act1 = Symbol.Activation()(Map("data" -> fully, "name" -> "relu1", "act_type" -> "relu"))
	    val fc2 = Symbol.FullyConnected()(Map("data" -> act1, "name" -> "fc2", "num_hidden" -> 64))
//	    val rhs = Symbol.CreateVariable("rhs")
	    
//	    val sum = Symbol.Softmax_cross_entropy(lhs,rhs)
	    val sum = Symbol.sum(fully)
	    val lhsArr = NDArray.array(Array(1f,2f,3f,10f),Shape(2,2))
	    val weightArr = NDArray.zeros(Shape(2,2))
	    val executor = sum.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"weight"->weightArr))
	    println(sum.staticGraph.debug)
	    executor.forward(isTrain = true)
	    var out2 = executor.outputs(0).copy()
	    println(out2)	
	    val gradarr = NDArray.array(Array(1f),Shape(1)) 
	    executor.backward(gradarr)
	    
	    executor.gradArrays.foreach {println}	
	}
	
	
    test("Sum") {
	    val shape = Shape(10, 3, 4)
	    val lhs = Symbol.CreateVariable("lhs")
	    val sum = Symbol.Sum("sum")(Map("data"->lhs))
	    val lhsArr = NDArray.ones(shape)
	    lhsArr(1,1) = 4
	    val executor = sum.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr))
	    executor.forward()
	    val out2 = executor.outputs(0)
	    println(out2.shape)
   }
    
    
    /*
     * symbol assignment
     */
    test("symbol assigment") {
	    val shape = Shape(3, 4)
	    
	    val lhs = Symbol.CreateVariable("lhs")
	    var data = lhs+1
	    var data1 = data
	    data1 += lhs
	    val res = Symbol.Group(data,data1)
	    
	    val lhsArr = NDArray.ones(shape)
	    lhsArr(1,1) = 4
	    val executor = res.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr))
	    executor.forward()
	    val out2 = executor.outputs(0)
	    println(out2)
	    println(executor.outputs(1))
   }
    
    
	
	test("broadcast_plus"){
		val lhs = Symbol.CreateVariable("lhs")
    	val rhs = Symbol.CreateVariable("rhs")
    	val ret = Symbol.broadcast_minus(lhs,rhs)
    	val lhsArr = NDArray.ones(Shape(4,2))*2
    	val rhsArr = NDArray.ones(Shape(1,2))
    	val executor = ret.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr))
    	executor.forward()
    	val out2 = executor.outputs(0)
    	println(out2)
	}
	
	
	
	test("reshape:(4,3)"){
  		val label = Symbol.CreateVariable("label")
  		val inputs = Symbol.Reshape()(Map("data" -> label, "shape" -> "(-1,-1,6)"))
  		val shape = Shape(3, 4)
  		val lhsArr = NDArray.ones(shape)
		val executor = inputs.easy_bind(ctx = Context.cpu(), args = Map("label"->lhsArr))
		executor.forward()
		val out2 = executor.outputs(0)
  		println(out2.shape)
  	}
	
	test("reshape"){
  		val label = Symbol.CreateVariable("label")
  		val inputs = Symbol.Reshape()(Map("data" -> label, "target_shape" -> "(0,)"))
  		val shape = Shape(10, 4)
  		val lhsArr = NDArray.ones(shape)
		val executor = inputs.easy_bind(ctx = Context.cpu(), args = Map("label"->lhsArr))
		executor.forward()
		val out2 = executor.outputs(0)
  		println(out2.shape)
  	}

	
	
	test("SliceChannel"){
		val shape = Shape(10, 4, 3)
		val data = Symbol.CreateVariable("data")
		val inputs = Symbol.SliceChannel()(Array(data),Map("num_outputs" -> 4, "squeeze_axis" -> true))
		val lhsArr = NDArray.ones(shape)
		val executor = inputs.easy_bind(ctx = Context.cpu(), args = Map("data"->lhsArr))
		executor.forward()
		val out2 = executor.outputs(0)
    	println(out2.shape)
    	
	}
	
	
  test("abs") {
    val shape = Shape(10, 3)
    val lhs = Symbol.CreateVariable("lhs")
    val lhs_abs = Symbol.abs(lhs)
    val ret =lhs_abs-lhs 
    assert(ret.listArguments().toArray === Array("lhs"))
    val lhsArr = NDArray.zeros(shape)-NDArray.ones(shape)
    val executor = ret.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr))
    executor.forward()
    val out1 = lhsArr*2
    val out2 = executor.outputs(0)
    println(out2)
    
  }
  
  test("Activation"){
      val lhs = Symbol.CreateVariable("lhs")
      val s = Symbol.Activation("ss")(Map("data"->lhs,"act_type"->"tanh"))
  }

  
  test("concat") {
    val shape = Shape(10, 3)
    val lhs = Symbol.CreateVariable("lhs")
    
    val concat0=Symbol.Concat("concat0")(Array(lhs))
    assert(concat0.listArguments().toArray === Array("lhs"))
    val lhsArr = NDArray.ones(shape)
    val executor = concat0.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr))
    executor.forward()
    val out2 = executor.outputs(0)
    println(out2)
  }
  
  /**
   * 
   * 
   */
  test("ElementWiseSum"){
    val lhs = Symbol.CreateVariable("lhs")
    val rhs = Symbol.CreateVariable("rhs")
    val ret = Symbol.ElementWiseSum("ElementWiseSum1")(Array(lhs,rhs))
    val shape = Shape(10, 3)
    val lhsArr = NDArray.ones(shape)
    val rhsArr = NDArray.ones(shape)*2
    val executor = ret.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr))
    executor.forward()
    val out2 = executor.outputs(0)
    println(out2)
  }
  
  /**
   * @author liuxianggen
   * @date 20160726
   * @brief here, you can test the symbol softmaxOutput operation, and know its loss output and gradient
   * 		more information please refer to the definition of softmaxOutput
   * @note
   */
  test("softmax Operation"){
      val data = Symbol.CreateVariable("data")
      val label = Symbol.CreateVariable("label")
      
      val batch_size = 10
      val num_input = 6
      val hidden = 100
      val shape = Shape(batch_size, num_input)
      
      val fully = Symbol.FullyConnected("fc1")(Map("data"->data,"num_hidden"->hidden))
      val ret =  Symbol.SoftmaxOutput("softmax")(Map("data" -> fully,"label"->label))  
//      ret.listArguments().foreach(println)
      val (a,b,c) = ret.inferShape(Map("data"->shape,"label"->Shape(batch_size)))
//      a.foreach {println}
      
      
      val dataArr = NDArray.ones(shape)
      val fc1_weight = NDArray.ones(Shape(hidden,num_input))
      val fc1_bias = NDArray.ones(Shape(hidden))
      val labelArr = NDArray.ones(Shape(batch_size))*3
      
      val executor = ret.easy_bind(ctx = Context.cpu(), args = Map("data"->dataArr,"fc1_weight"->fc1_weight,"fc1_bias"->fc1_bias,"label"->labelArr))
      executor.forward(isTrain=true)
      val out2 = executor.outputs(0)
      println(out2)
      executor.backward()
      executor.gradArrays.foreach {println}
  }
  /**
   * @author liuxianggen
   * @date 20160726
   * @brief here, you can test the symbol softmaxOutput operation, and know its loss output and the input regulation
   * 		more information please refer to the definition of softmaxOutput
   * @note
   */
  test("softmax Operation　simple"){
      val data = Symbol.CreateVariable("data")
      val label = Symbol.CreateVariable("label")
      
      val batch_size = 10
      val num_input = 3
      val shape = Shape(batch_size, num_input)
      
      val ret =  Symbol.SoftmaxOutput("softmax")(Map("data" -> data,"label"->label))  
//      ret.listArguments().foreach(println)
      val (a,b,c) = ret.inferShape(Map("data"->shape))
//      a.foreach {println}
      
      
      val dataArr = NDArray.ones(shape)
      dataArr(1,1) = 2
      println(math.exp(2)/(math.exp(1)*2+math.exp(2)))
      val labelArr = NDArray.ones(Shape(batch_size))*3
      
      val executor = ret.easy_bind(ctx = Context.cpu(), args = Map("data"->dataArr,"label"->labelArr))
      executor.forward(isTrain=true)
      val out2 = executor.outputs(0)
      println(out2)
//      executor.backward()
//      executor.gradArrays.foreach {println}
  }
  
  /**
   *
   */
  test("operation:*"){
    val lhs = Symbol.CreateVariable("lhs")
    val rhs = Symbol.CreateVariable("rhs")
    val ret = lhs*rhs
    val shape = Shape(10, 3)
    ret.inferShape(Map("lhs"->shape,"rhs"->shape))
    val lhsArr = NDArray.ones(shape)
    val rhsArr = NDArray.ones(shape)*8
    lhsArr(1,1) = 12
    val executor = ret.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr,"rhs"->rhsArr))
    executor.forward()
    val out2 = executor.outputs(0)
    println(out2)
  }
  

  
  /**
   * i have no idea about the embeding operation
   */
  test("embeding"){
	  val data = Symbol.CreateVariable("data")
	  val embedWeight = Symbol.CreateVariable("embed_W")
	  val embed = Symbol.Embedding("embed")(Map("data" -> data, "input_dim" -> 30,
                                           "weight" -> embedWeight, "output_dim" -> 5))
 	  val shape = Shape(3, 2)
//      val (a,b,c) = embed.inferShape(Map("data"->shape))
//      a.foreach {println}
//	  b.foreach(println)
 	  
 	  val dataarr = NDArray.diag(Shape(2,3))
// 	  dataarr(0,2) = 4
      val embedWeightarr = NDArray.ones(Shape(30,5))
//    lhsArr(1,1) = 12
      val executor = embed.easy_bind(ctx = Context.cpu(), args = Map("data"->dataarr,"embed_W"->embedWeightarr))
      executor.forward()
      val out2 = executor.outputs(0)
      println(out2)
  }
  
  /**
   * member functions
   */
 	test("listAuxTest"){
  	  val data = Symbol.CreateVariable("data")
      val conv1 = Symbol.Convolution()(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))

      conv1.listAuxiliaryStates().foreach(println)
      println("listAuxTest end ")
   }
  
  
  def main1(args:Array[String]):Unit = {
    println("<-----------TEST Symbol Part------------>")
//  	createTest  	
//    createVariableTest
//    composeTest
//    ToStaticGraphTest
//    unzipTest
//    mapTest
//    foldLeftTest
//    SetAttrTest
//    DFSVisitTest
//    stackTest
//    inferShapeTest
//    inferShape_plusTest_fc1
//    ToStaticGraphTest_2
//    inferShape_plusTest_fc12
//		operatorIntegrateTest
//    simpleBindTest
    listAuxTest
//    listArguments_
  }
  
  
  /**
   * 2016-3-21
   * test create function
   *  succeed!
   */
  def createTest{
//  	def Create(op: OperatorPropertyRef): Symbol
  	val operatorName = "Activation" 
  	val kwargs = Map("name" -> "relu1", "act_type" -> "relu")
    val sb:Symbol = Symbol.Create(operatorName,kwargs)
    sb.heads_.foreach { x => {
//    	println("the op of heads:")
    	(x.source.value.opRef.value.printParam())}
    }
  }
  
  /**
   * 2016-3-21
   *  succeed!
   */
  def createVariableTest{
  	val name = "input"
  	val sb = Symbol.CreateVariable(name)
  	sb.heads_.foreach { x => {
    	println("the name of heads:")
    	println(x.source.value.name)}
    }
  }
  
  
  /**
   * 2016-3-20
   * succeed!
   */
  def composeTest{
//  	def Compose(kwargs: Map[String, Symbol], name: String) {
  	val dataS = Symbol.CreateVariable("data")
  	val weightS = Symbol.CreateVariable("weight")
  	val biasS = Symbol.CreateVariable("bias")
  	val sb:Symbol = Symbol.Create("FullyConnected")
  	val kwargs:Map[String,Symbol] = Map("data"->dataS,"weight"->weightS,"bias"->biasS) 
  	
  	sb.Compose(kwargs, "FullyConnectedS")
  	sb.heads_(0).source.value.inputs.foreach { x => println(x.Info) }
//  	println(sb.is_atomic())//true
  	
  }
  
  /**
   * 2016-3-23
   */
  def  ToStaticGraphTest{
//  		def ToStaticGraph(out_graph: StaticGraph) {
  	
  	val sgref = new StaticGraphHandleRef
  	val sg:StaticGraph = new StaticGraph()
  	val dataS = Symbol.CreateVariable("data")
  	val weightS = Symbol.CreateVariable("weight")
  	val biasS = Symbol.CreateVariable("bias")
  	val sb:Symbol = Symbol.Create("FullyConnected")
//  	val kwargs:Map[String,Symbol] = Map("data"->dataS,"weight"->weightS,"bias"->biasS)
  	val kwargs:Map[String,Symbol] = Map("data"->dataS)
  	
  	sb.Compose(kwargs, "FullyConnectedS")
  	
  	val weightS1 = Symbol.CreateVariable("weight1")
  	val biasS1 = Symbol.CreateVariable("bias1")
  	val sb1:Symbol = Symbol.Create("FullyConnected")
    val kwargs1:Map[String,Symbol] = Map("data"->sb)
    sb1.Compose(kwargs1, "FullyConnectedS1")
  
//  	sb1.ToStaticGraph(sg)
  	
  	println(sg.debug)
  
  }
  
  
   
  def ToStaticGraphTest_2{
  	
  	val dataS = Symbol.CreateVariable("data")
  
  	val kwargs_type = Map("name" -> "fc2", "num_hidden" -> "10")
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "FullyConnectedS")
//  	var  out_graph= new StaticGraph()
//  	sb.ToStaticGraph(out_graph)
// 		println(out_graph.debug)
  	println("\n---------------------------------------------------")
  }
  
  
  /**
   * 2016-3-23
   * test dfs
   */
  def DFSVisitTest{
  	val dataS = Symbol.CreateVariable("data")
  	val weightS = Symbol.CreateVariable("weight")
  	val biasS = Symbol.CreateVariable("bias")
  	
    val sb:Symbol = Symbol.Create("FullyConnected")
  	val kwargs:Map[String,Symbol] = Map("data"->dataS,"weight"->weightS,"bias"->biasS) 
  	sb.Compose(kwargs, "FullyConnectedS")
  	
  	sb.DFSVisit { noderef => {
  		println("node:")
  		println(noderef.value.name) 
  		}
  	}
  }
  
  
  /**
   * 2016-3-23
   */
  def stackTest{
  	var stack: Stack[(String, Int)] = Stack()
  	stack.push(("a",1),("b",2),("c",3))
  	stack.update(0, ("c",0))
  	while (!stack.isEmpty) {
			println(stack.pop())
  	}
  }
  
  /**
   * 2016-3-25
   */
  def unzipTest{
  	val ve = Vector((1,"a"),(3,"v"))
  	val (a,b) = ve.unzip
  	a.foreach(print)
  }
  
  def mapTest{
  	val m:scala.collection.mutable.Map[String,Int] =scala.collection.mutable.Map()
  	m += ("a"->1,"b"->2)
  	val (ms,mi) = m.unzip
  	println(ms)
  	println(m)
  }
  
  
  def foldLeftTest{
  	val arr = Array(1,2,3,4,5,5)
  	println(arr.foldLeft(0)(_ + _))
  }
  
  /**
   * 2016-3-25
   * inferShape function will call ToStaticGraph(g),so it needs to convert StaticGraph 
   * from java to C++ first 
   */
  def inferShapeTest{
  	
  	val dataS = Symbol.CreateVariable("data")
  	val weightS = Symbol.CreateVariable("weight")
  	val biasS = Symbol.CreateVariable("bias")
  	val kwargs_type = Map("name" -> "fc2", "num_hidden" -> "10")
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "FullyConnectedS")
  	
  	val kwargs_shape = Map("data"->Shape(200,15))
  	val keys = ArrayBuffer.empty[String]
    val indPtr = ArrayBuffer(0)
    val sdata = ArrayBuffer.empty[Int]
    kwargs_shape.foreach { case (key, shape) =>
      keys += key
      sdata ++= shape.toVector
      indPtr += sdata.size
    }
  	
  	println("keys:")
  	keys.foreach {println}
  	println("\nsdata:")
  	sdata.foreach(println)
  	println("\nindPtr:"+indPtr)

  	println("\n---------------------------------------------------")
//    val (argShapes, _, auxShapes) = sb.inferShape(keys.toArray, indPtr.toArray, sdata.toArray)
  }
  
  
  /**
   * 2016-3-25
   * inferShape function will call ToStaticGraph(g),so it needs to convert StaticGraph 
   * from java to C++ first 
   */
  def inferShape_plusTest_fc12{
  	
  	val dataS = Symbol.CreateVariable("data")
  
  	val kwargs_type = Map("name" -> "fc1", "num_hidden" -> "12")
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "fc1")
  	
  	val kwargs_type1 = Map("name" -> "fc2", "num_hidden" -> "10")
  	val sb1:Symbol = Symbol.Create("FullyConnected",kwargs_type1)
  	val kwargs_symbol1:Map[String,Symbol] = Map("data"->sb) 
  	sb1.Compose(kwargs_symbol1, "fc2")
  	
  	
  
  	sb1.ToStaticGraph()
 		println(sb1.staticGraph.debug)
  	println("\n---------------------------------------------------")
  	val kwargs_shape = Map("data"->Shape(200,15))
//  	
//    val (argShapes, _, auxShapes) = sb1.inferShape1(sb1.staticGraph,kwargs_shape)
//    argShapes.foreach {println}
  }
  
  
   /**
   * 2016-3-25
   * inferShape function will call ToStaticGraph(g),so it needs to convert StaticGraph 
   * from java to C++ first 
   */
  def inferShape_plusTest_fc1{
  	
  	val dataS = Symbol.CreateVariable("data")
  
  	val kwargs_type = Map("name" -> "fc1", "num_hidden" -> "12")
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "fc1")
	
  	
//  	var  out_graph= new StaticGraph()
  	sb.ToStaticGraph()
 		println(sb.staticGraph.debug)
  	println("\n---------------------------------------------------")
  	val kwargs_shape = Map("data"->Shape(200,15))
  	
//    val (argShapes, outShapes, auxShapes) = sb.inferShape1(sb.staticGraph,kwargs_shape)
//    argShapes.foreach {println}
//  	outShapes.foreach {println}
  }
  
  /**
   * 2016-3-24
   * by liuxianggen
   * not sure
   */
  def SetAttrTest(){
  	val dataS = Symbol.CreateVariable("data")
  	val weightS = Symbol.CreateVariable("weight")
  	val biasS = Symbol.CreateVariable("bias")
  	val sb:Symbol = Symbol.Create("FullyConnected")
  	val kwargs:Map[String,Symbol] = Map("data"->dataS,"weight"->weightS,"bias"->biasS) 
  	sb.Compose(kwargs, "FullyConnectedS")
  	sb.SetAttr("name", "FullyConnected")
  	sb.SetAttr("hidden", "10")
  }
  
  
	
	/**
	 * 
	 *by liuxianggen
	 * 2016-4-5
	 * succeed!!
	 */
	def operatorIntegrateTest{
		val num_instance = 15
  	val input_dim = 10
		val hidden_1 =5
		val hidden_2 =5 
		
		val dataS = Symbol.CreateVariable("data")
		
		val fc1 = Symbol.FullyConnected()(Map("name" -> "fc1", "num_hidden" -> hidden_1 ,"data"->dataS))
  	
		val fc2 = Symbol.FullyConnected()(Map("name" -> "fc2", "num_hidden" -> hidden_2 ,"data"->fc1))
  	
  	val sm = Symbol.SoftmaxOutput()(Map("name" -> "sm","grad_scale"->"1","data"->fc2))
  	
  	val data = NDArray.rangeRows(0, num_instance, input_dim)//num_instance,10
  	val label = NDArray.range(0,5,3)
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
  	
//  	var  out_graph= new StaticGraph()
//  	sm.ToStaticGraph(out_graph)
// 		println(out_graph.debug)
// 		out_graph.ToStaticGraph
//  	val executor = out_graph.bind(in_args, arg_grad_store, grad_req_type)

//  	val executor = sm.bindHelper(in_args, arg_grad_store,grad_req_type)
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
//  	
//  	for(i<- 0 until 10){
//  		 println("epoch:"+i)
//			 	executor.forward()
//			 	executor.backward()
//			 	println(executor.outputs(0))
//			 	val acc: Float = mathTool.output_accuracy(executor.outputs(0), label)
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
	
   def simpleBindTest{
  	 import thu.brainmatrix.Context
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
                           
    softmax.listArguments().foreach(println)
    
  	 val  dataShapes = Map("data" -> Shape(100,1,28, 28))
  	 
  	 val dataShapes_ =collection.immutable.Map(dataShapes.toList: _*) 
     softmax.simpleBind(Context.cpu(), "write", shapeDict = dataShapes_)
  	 
  	 
  }
  
   
   def listArguments_{
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
                
    softmax.listArguments().foreach(println)
   }
  
   
   def listAuxTest{
  	  val data = Symbol.CreateVariable("data")
      val conv1 = Symbol.Convolution()(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))
      conv1.listAuxiliaryStates().foreach(println)
   }
   
   
   
   
}