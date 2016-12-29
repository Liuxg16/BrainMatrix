package thu.brainmatrix.lstmbyguo

import thu.brainmatrix.NDArray
import java.io.File
import java.io.FileWriter
import thu.brainmatrix.Shape

class Test {

}

object Test {
  private val matrixfilepath: String = "./seqData/test.txt"
  var matrixfile = new File(matrixfilepath)

  def test_transpose(src: NDArray) {
    var shapes = src.shape
    val head = shapes.apply(0)
    val tail = shapes.apply(1)
    println(shapes)
    var res = NDArray.zeros(tail, head).toArray
    var tempsrc = src.toArray
    //    for (i <- 0 until head; j <- 0 until tail) {
    //
    //    }

  }
  def main(args: Array[String]): Unit = {
    var test: NDArray = NDArray.ones(Shape(5, 6))
    for (i <- 0 to 4) {
      test.slice(i) *= i
    }
    println(NDArray.transpose(test))
    println(test)
    //    println(NDArray.transpose(test))
    //    var test2:NDArray = NDArray.ones(Shape(3,4))
    //    println("--------------------------\n" + test.reshape(Array(2, 3)))
    //    if (matrixfile.exists()) {
    //      matrixfile.delete()
    //    }
    //    matrixfile.createNewFile()
    //    var n = 0
    //    while (n < 10) {
    //      n += 1
    //      val writer = new FileWriter(matrixfilepath, true)
    //      writer.write("" + "\n" + NDArray.ones(2, 3) + "\n")
    //      writer.close()
    //    }
    //    println("ren zha")
  }
}