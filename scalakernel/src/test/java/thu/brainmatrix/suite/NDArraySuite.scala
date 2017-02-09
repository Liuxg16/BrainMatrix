package thu.brainmatrix.suite

import thu.brainmatrix.NDArray
import thu.brainmatrix.Random
import thu.brainmatrix.Shape
import thu.brainmatrix.Context
import scala.Vector
import org.scalatest.{ BeforeAndAfterAll, FunSuite }

/**
 * by liuxianggen,guoshen
 * 2016-8-19
 * to test the operations of NDArray
 */
class NDArraySuite extends FunSuite with BeforeAndAfterAll {
	
	/**
	 * 2016-12-23
	 * 
	 */
	
	
	
	/**
	 * 2016-12-10
	 * 
	 */
	test("NDArray.concatenate"){
		val ctx = Context.cpu(0)
		val nda = NDArray.ones(Shape(2,3),ctx)
//		println(nda)
//		println(NDArray.concatenate(nda,nda))
	}
	
	/**
	 * 2016-12-10
	 * 
	 */
	test("NDArray.argmaxChannel"){
		val ctx = Context.cpu(0)
		val e_ik = Array.fill[Array[Float]](3)(Array.fill[Float](4)(0f))
		var n = 0;
		val arr = e_ik.map(e_i => e_i.map(eij =>{
			n += 1
			eij+ 2*(n%2)+ n
					
		}))
		
		val nda = NDArray.array(arr.flatten,Shape(4,3), ctx)
//		println(nda)
//		println(NDArray.argmaxChannel(nda).shape)
		
		
	}
	
	
	
	
	/**
	 * 2016-12-10
	 * test a ndarray operator of my own
	 * this function can not be use for gpu
	 */
	test("NDArray.array"){
		val ctx = Context.cpu(0)
		val e_ik = Array.fill[Array[Float]](3)(Array.fill[Float](4)(0f))
		var n = 0;
		val arr = e_ik.map(e_i => e_i.map(eij =>{
			n += 1
			eij+n
					
		}))
		
//		println(NDArray.array(arr.flatten,Shape(4,3), ctx))
		
		
	}
	
	
	
	/**
	 * 2016-12-10
	 * test a ndarray operator of my own
	 * this function can not be use for gpu
	 */
	test("Normalize"){
		val ctx = Context.cpu(0)
		val nda = NDArray.ones(Shape(2,3))*4
//		println(NDArray.Normalize(nda))
	}
	
	
	
	
	
	
	/**
	 * 2016-11-29
	 * test a ndarray operator of my own
	 * this function can not be use for gpu
	 */
	test("Random-uniform"){
		val ctx = Context.cpu(0)
		import thu.brainmatrix.Random
		val nda = Random.uniform(0,1, Shape(3,4), ctx, null)
//		println(nda)
	}
	
	/**
	 * 2016-11-30
	 */
	test("one-hot"){
		val ctx = Context.cpu(0)
		val indices = NDArray.range(0,4)+1.8f
//		println(indices)
		val out = NDArray.zeros(Shape(4,4), ctx)
		NDArray.onehotEncode(indices, out)
//		println(out)
	}
	
	/**
	 * 2016-12-2
	 */
	test("one-hot-bigger"){
		val ctx = Context.cpu(0)
		val indices = NDArray.range(4,8)+0.5f
//		println(indices)
		val out = NDArray.ones(Shape(4,10), ctx)
		val out1 = NDArray.ones(Shape(4,10), ctx)*9
		NDArray.onehotEncode(indices, out)
		
//		println(out * out1)
	}
	
	
	
	
	/**
	 * 2016-11-10
	 * test a ndarray operator of my own
	 * this function can not be use for gpu
	 */
	test("run gpu"){
		
		val ctx = Context.cpu(0)
//		val ctxg= Context.gpu(0)
		var nda1 = NDArray.ones(ctx, 10,10)*2
	  	var nda2 = NDArray.ones(ctx, 10,10)*3
	  	var n=0
//	  	while(n<1000){
//	  		var j=0
//	  		println(n)
//	  		while(j<10000){
//	  		  	var nda3 = NDArray.exp(-nda1)*NDArray.sigmod(nda1)
//	  		  	var nda4   = NDArray.ones(ctx, 10,10)/nda3 
//	  		  	var nda5   = NDArray.ones(ctx, 10,10)/nda4 
//	  		  	var nda6   = NDArray.ones(ctx, 10,10)/nda5 
//	  		  	var nda7   = NDArray.ones(ctx, 10,10)/nda6 
//	  		  	var nda8   = NDArray.ones(ctx, 10,10)/nda7 
//	  		  	var nda9   = NDArray.ones(ctx, 10,10)/nda8 
//	  		  	var nda10  = NDArray.ones(ctx, 10,10)/nda9 
//	  		  	var nda11  = NDArray.ones(ctx, 10,10)/nda10
//	  		  	var nda12  = NDArray.ones(ctx, 10,10)/nda11
//	  		  	var nda13  = NDArray.ones(ctx, 10,10)/nda12
//	  		  	var nda14  = NDArray.ones(ctx, 10,10)/nda13
//	  		  	var nda15  = NDArray.ones(ctx, 10,10)/nda14
//	  		  	var nda16  = NDArray.ones(ctx, 10,10)/nda15
//	  		  	var nda17  = NDArray.ones(ctx, 10,10)/nda16
//	  		  	var nda18  = NDArray.ones(ctx, 10,10)/nda17
//	  		  	var nda19  = NDArray.ones(ctx, 10,10)/nda18
//	  		  	var nda20  = NDArray.ones(ctx, 10,10)/nda19
//	  		  	
//	  			j += 1
//	  		}
//	  		n += 1
//	  	}
	  	
//	    println(nda2)	  	
	
	}
	
	
	
	/**
	 * 2016-11-10
	 * test a ndarray operator of my own
	 * this function can not be use for gpu
	 */
	test("copy"){
		
		val ctx = Context.cpu(0)
		val ctxg= Context.gpu(0)
		val nda1 = NDArray.ones(ctx, 1,3)*2
	  	val nda2 = NDArray.ones(ctx, 1,3)*3
	  
	  	var tt  = nda1 *3
//	  	tt +=  NDArray.ones(ctx, 1,3)*3

	  	
//	  	nda1.copyTo(nda2.slice(1))
	  	
//	    println(nda1)	  	
	
	}
	
	
	/**
	 * test the computational order
	 */
	test("arithmetic assosiation "){
		
		val ctx = Context.cpu(0)
		val nda1 = NDArray.ones(ctx, 2,3)*2
	  	val nda2 = NDArray.ones(ctx, 2,3)*3
	  	
//	  	println(nda1 - nda1 * nda2)
//	    println(nda1 -( nda1 * nda2))
//	    println(nda1 * nda1 - nda2)
	
	}

	
	
	/**
	 * 2016-11-10
	 * test a ndarray operator of my own
	 * this function can not be use for gpu
	 */
	test("integate_lxg"){
		
		val ctx = Context.cpu(0)
		val nda1 = NDArray.ones(ctx, 2,3)
	  	val nda2 = NDArray.ones(ctx, 2,3)
	  
//		  println(nda2)
//		  println(NDArray.integate_lxg(nda2,nda1))
	}

	
	/**
	 * 2016-11-10
	 * test a ndarray operator of my own
	 * 
	 */
	test("setslice_lxg"){
//		val ctx = Context.gpu(0)
//		val nda1 = NDArray.ones(ctx, 9,14) * 10
//	  	val nda2 = NDArray.ones(ctx, 9, 1)
//	    println(nda1)
//	    println(nda2)
	  	
//	     NDArray.setColumnSlice(nda1,nda2,0)
//	     println(nda1)
//	     println(nda2)
	}
	
	test("dot0"){
		val arr = NDArray.ones(5, 4)
//		println(arr)
		val arr1 = NDArray.ones(4, 1)
		val res  = NDArray.dot(arr, arr1)
//		println(res)
	}
	
	
	test("transpose"){
		val arr = NDArray.range(0, 5, 4)
//		println(arr)
		val arrr = NDArray.transpose(arr)
//		println(arrr)
	}
	
	
	test("reshape"){
		val arr = NDArray.range(0, 5, 4)
//		println(arr)
		val arrr = arr.reshape(Array(5,4))
//		println(arrr)
	}
	
	
	
	test("+"){
		val arr = NDArray.range(0, 5, 4)
		val arrr = NDArray.array(arr.toArray,Shape(2,10))
	}
	
  
	test("toarray"){
		val arr = NDArray.zeros(Shape(2,2))
		val arrr = NDArray.ones(Shape(2,1))
		
	}
	
	
	test("toString"){
		val av = NDArray.ones(Shape(2,3,4))
		av(1,1,1) =2
//		println(av)
		
	}
	
	
    test("load2map"){
//		val pretrained = NDArray.load2Map("./model/charLSTM.params_6")
//    	println(pretrained.head)
  	}
  
  	test("save NDArray"){
    	val nda = Map("data"->NDArray.ones(2,3))
    	NDArray.save("./model/test", nda)
  	}
  
  	test("slice"){
  		val ind = NDArray.ones(Shape(4,3,2))
//  		println(ind)
  		ind(1,1,1) = 3
//  		println(ind.slice(1).slice(0))
  	}
  	
  	test("copyto"){
//  		val ctx = Context.gpu(0)
//  		val ind = NDArray.ones(Shape(4,3),ctx)
//  		val ind2 = NDArray.zeros(Shape(4,3))
  		
//  		println(ind.copyTo(ctx))
  	}
  	
  	test("argmaxChannelTest") {
	    val nmArr = Random.normal(0f, 1f, Shape(4, 8))
//	    println(nmArr)
	
	    val py = NDArray.argmaxChannel(nmArr)
//	    println(py)
	  }
  
  
  def main1(args: Array[String]) {
    TestSet
    //  	TestSetloop
    //	  TestSize
    //  	TestListArrayFunc
    //  	TestRange
    //	  ndarrayOperationTest
    //  	argmaxChannelTest
    //      meanTest
  }

  

  def TestSet {
    val num_instance = 15
    val input_dim = 10
    val data = NDArray.ones(Shape(15, 10))
    val label = NDArray.zeros(Shape(num_instance))

    for (i <- 0 until num_instance) {
      for (j <- 0 until input_dim) {
        data(i, j) = i % 5 * 1.0f + (scala.util.Random.nextFloat - 0.5f)
      }
      label(i) = i % 5
      println(label(i))
    }

    println(label)
  }

  def TestSize() {
    var label = NDArray.zeros(Shape(15, 12))
    println(label.size)
  }

  def TestSetloop {
    var label = NDArray.zeros(Shape(15))

    for (i <- 0 until 15) {
      val temp = (i / 5).floor
      println(temp)
      label(i) = temp
    }
    println(label)

  }

  def TestListArrayFunc {
    val lhsArr = Random.uniform(-10f, 10f, Shape(3, 4))
  }

  def TestRange {
    //  	val arr = NDArray.range(0,10)
    //  	val arr = NDArray.rangeRows(0, 10, 5)
    val arr = NDArray.range(0, 10, 3)
    println(arr)
  }

  def ndarrayOperationTest {
    val lhs = NDArray.ones(Shape(3, 4))
    val rhs = NDArray.ones(Shape(3, 4))
    val sum = lhs + rhs
    println(sum)
  }

  def meanTest {
    val arr = Random.uniform(0, 10, Shape(4, 5))
    print(NDArray.mean(arr))
  }
  def TestTan {
    val Pi = scala.math.Pi.toFloat
    val h = NDArray.tan(NDArray.array(Array(0, Pi / 4, Pi / 2, 3 * Pi / 4), Shape(1, 4)))
    println(h)
  }
  def TestTanh {
    val Pi = scala.math.Pi.toFloat
    val h = NDArray.tanh(NDArray.array(Array(-1, 0, 1, 2), Shape(1, 4)))
    println(h)
  }
   def TestTranspose {
    val pre = NDArray.array(Array(1, 2, 3, 4, 5, 6), Shape(1, 6))
    val after = NDArray.transpose(pre)
    println(pre)
    println(after)
  }
  
}