

package thu.brainmatrix.suite

import thu.brainmatrix.Base._
import thu.brainmatrix.KVStore
import thu.brainmatrix.MXKVStoreUpdater
import thu.brainmatrix.NDArray
import thu.brainmatrix.Shape
import scala.Vector
object kvstoreTest {

    def main(args:Array[String]){
      test1
//      val kv = KVStore.create()
//      val shape:Shape =Vector(2,3,4)
//      val shape1 = Vector(1,2)
////      val shape1 = Shape (3,4,5)
//      val ndArray = NDArray.ones(shape)
//      val ndArray1 = NDArray.ones(1,2)*4
//      
//      val keys = Array(2,3)
//      val values = Array(ndArray,ndArray1)
//      kv.init(keys, values)
//      kv.pull(2, ndArray)
////      kv.init(2, NDArray.ones(shape1))
//      kv.pull(3, ndArray1)
//      println(kv.numWorkers)
    }
    
    def test {
      val kv = KVStore.create()
      val shape = Shape(2, 1)
      val ndArray = NDArray.zeros(shape)
  
    //  kv.init(3, NDArray.ones(shape))
      kv.push(3, NDArray.ones(shape) * 4)
      kv.pull(3, ndArray)
//      println(ndArray.toString)
//      assert(ndArray.toArray === Array(4f, 4f))
  }

    def test1{
      
      val kv = KVStore.create()
      val updater = new MXKVStoreUpdater {
        override def update(key: Int, input: NDArray, stored: NDArray): Unit = {
          // scalastyle:off println
//          println(s"update on key $key")
          // scalastyle:on println
          stored += input * 2
        }
        override def dispose(): Unit = {}
      }
    kv.setUpdater(updater)
    val shape = Shape(2, 2)
    val ndArray = NDArray.zeros(shape)

    kv.init(3, NDArray.ones(shape) * 4)
    kv.pull(3, ndArray)
    println(ndArray)
//    assert(ndArray.toArray === Array(4f, 4f))

    kv.push(3, NDArray.ones(shape))
    kv.pull(3, ndArray)
    println(ndArray)
    
    kv.push(3, NDArray.ones(shape))
    kv.pull(3, ndArray)
    println(ndArray)
//    kv.finalize()
  }

  def test2{
    val kv = KVStore.create("local")
    
  }

  def test3{
    val kv = KVStore.create("local")
//    assert(kv.numWorkers === 1)
//    assert(kv.rank === 0)
  }

}