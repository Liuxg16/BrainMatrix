package thu.brainmatrix.suite
import thu.brainmatrix.Symbol
import thu.brainmatrix.Random
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class ExecutorSuite extends FunSuite with BeforeAndAfterAll {
  test("bind") {
    val shape = Shape(10, 3)
    val lhs = Symbol.Variable("lhs")
    val rhs = Symbol.Variable("rhs")
    val ret1 = lhs + rhs
    val ret =ret1/4+rhs 
    assert(ret.listArguments().toArray === Array("lhs", "rhs"))
//    println(ret.debug())
    val lhsArr = NDArray.ones(shape)
    val rhsArr = NDArray.ones(shape)*2
    val lhsGrad = NDArray.zeros(shape)
    val rhsGrad = NDArray.empty(shape)

    val executor = ret.easy_bind(ctx = Context.cpu(), args = Map("lhs"->lhsArr, "rhs"->rhsArr),
                            argsGrad = Map("lhs"->lhsGrad, "rhs"-> rhsGrad))
//    
    executor.forward()
//   
//    val out1 = lhsArr + rhsArr
    val out2 = executor.outputs(0)
//   
//    
//    // test gradient
//    val outGrad = NDArray.ones(shape)
//    val (lhsGrad2, rhsGrad2) = (outGrad, outGrad)
//    executor.backward(Array(outGrad))
//    
//    println(out2)
  }

  
}