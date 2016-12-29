package thu.brainmatrix.suite

import scala.io.Source
import thu.brainmatrix.IO
import thu.brainmatrix.Shape
import thu.brainmatrix.DataIter
import thu.brainmatrix.DataBatch
import thu.brainmatrix.NDArray
import thu.brainmatrix.io.NDArrayIter
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import thu.brainmatrix.lxg.CVTool
/**
 * @author liuxianggen
 * @date 20160712
 * @brief test some functions related IO module
 * @param
 * @return
 * @example
 * @note
 */
class IOSuite extends FunSuite with BeforeAndAfterAll{
    
    test("mnist dataset") {
        val batchSize = 100
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
        
        val data = takeIterElemt(trainDataIter,30).data.head
//        println(NDArray.max(data))
//        
//        CVTool.saveGrayImage(data*255, "mnist.jpg")

    }
    
    def takeIterElemt(Iter: DataIter,idx:Int):DataBatch = {
        Iter.reset()
        var n = 0
        while(n<idx-1){
            Iter.next()
            n +=1
        }
        Iter.next()
    }
    
    test("readCorpus"){
        val fileName = "./seqData/input.txt"
        var dict = Map[String,Int]()
        val source  = Source.fromFile(fileName)
        val lineIter = source.getLines()
        for(l<- lineIter){
            val words = l.split("\\s+")
            words.map(w => {
                 dict = dict.updated(w, dict.getOrElse(w,0)+1)
            })
        }
//        println(dict.size)
    }

    /**
     * @author liuxianggen
     * @date 20160718
     * @brief there is the encoder of INPUT_FILE,make each char have a id,
     * 		which increase as the frequency decrease. For example：
     * input file:
     * 		I love you
     * vocab:O->1,I->2,l->3...
     * @param
     * @return
     * @example
     * @note
     */
    test("genVocab"){
        val fileName = "./seqData/input1.txt"
        var dict = Map[Char,Int]()
        val source  = Source.fromFile(fileName)
        val lineIter = source.getLines()
        for(l<- lineIter){
            l.map(w => {
                 dict = dict.updated(w, dict.getOrElse(w,0)+1)
            })
        }
//        println(dict)
    }
    
    /**
     * @author liuxianggen
     * @date 20160719
     * @brief test the construction of NDArrayIter
     * @param
     * @return
     * @example
     * @note
     */    
    test("test NDArrayIter") {
        val shape0 = Shape(1000, 2, 2)
        val data = IndexedSeq(NDArray.ones(shape0), NDArray.zeros(shape0))
        val shape1 = Shape(1000, 1)
        val label = IndexedSeq(NDArray.ones(shape1))
        val batchData0 = NDArray.ones(Shape(128, 2, 2))
        val batchData1 = NDArray.zeros(Shape(128, 2, 2))
        val batchLabel = NDArray.ones(Shape(128, 1))
    
        // test pad
        val dataIter0 = new NDArrayIter(data, label, 128, false, "pad")
        var batchCount = 0
        val nBatch0 = 8
        while(dataIter0.hasNext) {
          val tBatch = dataIter0.next()
          batchCount += 1
    
          assert(tBatch.data(0).toArray === batchData0.toArray)
          assert(tBatch.data(1).toArray === batchData1.toArray)
          assert(tBatch.label(0).toArray === batchLabel.toArray)
        }
    
        assert(batchCount === nBatch0)
    
        // test discard
        val dataIter1 = new NDArrayIter(data, label, 128, false, "discard")//the rest will discard
        val nBatch1 = 7
        batchCount = 0
        while(dataIter1.hasNext) {
          val tBatch = dataIter1.next()
          batchCount += 1
    
          assert(tBatch.data(0).toArray === batchData0.toArray)
          assert(tBatch.data(1).toArray === batchData1.toArray)
          assert(tBatch.label(0).toArray === batchLabel.toArray)
        }
        assert(batchCount === nBatch1)
  }
    
}