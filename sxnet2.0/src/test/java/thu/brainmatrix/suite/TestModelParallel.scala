//import thu.brainmatrix.Symbol
//import thu.brainmatrix.AttrScope
//import thu.brainmatrix.Context
//import thu.brainmatrix.NDArray
//import thu.brainmatrix.Base._
//
//
//object ModelParallel {
//  
//	def main(args:Array[String]){
//		testChain
////		testChain_simple
//	}
//	
//
//	def testChain {
//		val n = 2
//		
//		val data1 = Symbol.Variable("data1")
//    	val data2 = Symbol.Variable("data2")
//    	println("-------------------------------------------")
//    	var net: Symbol = null
//    	new AttrScope(Map("ctx_group" -> "dev1")).withScope {
//    		println("here!!")
//      		net = (data1 + data2) * 3
//      		println("here!!")
//    	}
//		new AttrScope(Map("ctx_group" -> "dev2")).withScope {
//      		net = net + data1
//    	}
//    	println("-------------------------------------------")
//		val shape:Shape = Vector(4, 5)
//    	val (arr, arrGrad) =
//      	new Context(Context.cpu(1)).withScope {
//        	val arr = (0 until n).map(_ => NDArray.ones(shape))
//        	val arrGrad = (0 until n).map(_ => NDArray.empty(shape))
//        	(arr, arrGrad)
//      	}
//		
//		println("-------------------------------------------")
//		val exec1 = net.bind(Context.cpu(),
//	    args = arr,
//	    argsGrad = arrGrad,
//	    gradReq = "write",
//	    auxStates = Nil,
//	    group2ctx = Map("dev1" -> Context.cpu(0), "dev2" -> Context.cpu(1)))
//
//	    println("-------------------------------------------")
//	    print(exec1.debugStr)
//	    println("-------------------------------------------")
//	    
////    	arr(0).set(1f)
////    	arr(1).set(2f)
//    	
//    	val arr2 = arr.map(_.copyTo(Context.cpu()))
//    	val arrGrad2 = arrGrad.map(_.copyTo(Context.cpu()))
//    	
//    	
//    	val exec2 = net.bind(Context.cpu(), args = arr2, argsGrad = arrGrad2)
//    	
//		// Show the execution plan that involves copynode
//    	// scalastyle:off println
//    	print(exec1.debugStr)
//    	// scalastyle:on println
//    	
//    	exec1.forward()
//    	exec2.forward()
//    	println(exec1.outputs(0))
//    	println(exec2.outputs(0))
//    	
//    	
//    	val outGrad = NDArray.ones(shape, Context.cpu(1))
//	    exec1.backward(Array(outGrad))
//	    exec2.backward(Array(outGrad.copyTo(Context.cpu())))
//	    
//	    (arrGrad zip arrGrad2) foreach { case (a, b) =>
////      		assert(reldiff(a, b) < 1e-6f)
//	    	println(a-b)
//    	}
//	}
//
//
//		def testChain_simple{
//		val n = 2
//		val data1 = Symbol.Variable("data1")
//    	val data2 = Symbol.Variable("data2")
//    	var net: Symbol = null
//    	net = data1 + (data2 * 3)
////    	net = net + data1
//    	val shape:Shape = Vector(4, 5)
//    	
//    	val (arr, arrGrad) =
//      	new Context(Context.cpu(0)).withScope {
//        	val arr = (0 until n).map(_ => NDArray.ones(shape))
//        	val arrGrad = (0 until n).map(_ => NDArray.empty(shape))
//        	(arr, arrGrad)
//      	}
//    	
//		
//		val exec1 = net.bind(Context.cpu(),
//	    args = arr,
//	    argsGrad = arrGrad,
//	    gradReq = "write",
//	    auxStates = Nil,
//	    group2ctx = Map("dev1" -> Context.cpu(0), "dev2" -> Context.cpu(1)))
//
//	    println("-------------------------------------------")
//	    print(exec1.debugStr)
//	    println("-------------------------------------------")
//	    
//
////    	
////    	val arr2 = arr.map(_.copyTo(Context.cpu()))
////    	val arrGrad2 = arrGrad.map(_.copyTo(Context.cpu()))
////    	val exec2 = net.bind(Context.cpu(), args = arr2, argsGrad = arrGrad2)
////    	
////		// Show the execution plan that involves copynode
////    	// scalastyle:off println
////    	print(exec1.debugStr)
////    	// scalastyle:on println
////    	
////    	exec1.forward()
////    	exec2.forward()
////    	println(exec1.outputs(0))
////    	println(exec2.outputs(0))
////    	
////    	
////    	val outGrad = NDArray.ones(shape, Context.cpu(1))
////	    exec1.backward(Array(outGrad))
////	    exec2.backward(Array(outGrad.copyTo(Context.cpu())))
////	    
////	    (arrGrad zip arrGrad2) foreach { case (a, b) =>
//////      		assert(reldiff(a, b) < 1e-6f)
////	    	println(a-b)
////    	}
//		
//	}
//
//	
//	
//	
//	
//	
//	
//	
////	
//	
//	
//	
//	
//	
//	
////    assert(reldiff(exec1.outputs(0), exec2.outputs(0)) < 1e-6f)
////
//
////    (arrGrad zip arrGrad2) foreach { case (a, b) =>
////      assert(reldiff(a, b) < 1e-6f)
////    }
////  }
//	
//}