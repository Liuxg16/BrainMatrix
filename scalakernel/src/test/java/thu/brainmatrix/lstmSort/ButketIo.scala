package thu.brainmatrix.lstmSort

import thu.brainmatrix.{DataBatch, DataIter, NDArray, Shape}
import org.slf4j.LoggerFactory
import scala.io.Source
import scala.util.Random

/**
 * @author Depeng Liang
 */
object ButketIo {

  type Text2Id = (String, Map[String, Int]) => Array[Int]
  type ReadContent = String => String

  def defaultReadContent(path: String): String = {
	  Source.fromFile(path).mkString
                        .replaceAll("\n", " <eos> ")
 
  }

  def defaultText2Id(sentence: String, theVocab: Map[String, Int]): Array[Int] = {
    val words = {
      val tmp = sentence.split(" ").filter(_.length() > 0)
      for (w <- tmp) yield theVocab(w)
    }
    words.toArray
  }


  
  def defaultGenBuckets(sentences: Array[String], batchSize: Int,
                        theVocab: Map[String, Int]): List[Int] = {
    val lenDict = scala.collection.mutable.Map[Int, Int]()
    var maxLen = -1
    for (sentence <- sentences) {
      val wordsLen = defaultText2Id(sentence, theVocab).length
      if (wordsLen > 0) {
        if (wordsLen > maxLen) {
          maxLen = wordsLen
        }
        if (lenDict.contains(wordsLen)) {
          lenDict(wordsLen) = lenDict(wordsLen) + 1
        } else {
          lenDict += wordsLen -> 1
        }
      }
    }

    var tl = 0
    var buckets = List[Int]()
    lenDict.foreach {
      case (l, n) =>
        if (n + tl >= batchSize) {
          buckets = buckets :+ l
          tl = 0
        } else tl += n
    }
    if (tl  > 0) buckets = buckets :+ maxLen
    buckets
  }

  class BucketSentenceIter(
      path: String, vocab: Map[String, Int], var buckets: List[Int],
      _batchSize: Int, initStates: IndexedSeq[(String, (Int, Int))],
      seperateChar: String = " <eos> ", text2Id: Text2Id = defaultText2Id,
      readContent: ReadContent = defaultReadContent) extends DataIter {

    private val logger = LoggerFactory.getLogger(classOf[BucketSentenceIter])

    private val content = readContent(path)
    private val sentences = content.split(seperateChar)
    // println(sentences.length)

    if (buckets.length == 0) {
      buckets = defaultGenBuckets(sentences, batchSize, vocab)
    }
    buckets = buckets.sorted//List(129)
    // pre-allocate with the largest bucket for better memory sharing,129
    private val defaultBucketKey = (buckets(0) /: buckets.drop(1)) { (max, elem) =>
      if (max < elem) elem else max
    }
    // we just ignore the sentence it is longer than the maximum
    // bucket size here
    private val data = buckets.indices.map(x => sentences.map(text2Id(_,vocab).map(_.toFloat))).toArray
    
//    val t = sentences.map(text2Id(_,vocab).map(_.toFloat))
//    
//    for (sentence <- sentences) {
////    	println(sentence)
//      val ids = text2Id(sentence, vocab)
//      if (ids.length > 0) {
//        buckets.indices.foreach { idx =>//data(0) is a collection contains many NDArrays,each is a sentence
//          if (buckets(idx) >= ids.length) {
//            data(idx) = data(idx) :+
//            //ids and ++ Array(129-ids.length) to compose a 129 data
//            (ids.map(_.toFloat) ++ Array.fill[Float](buckets(idx) - ids.length)(0f))
//          }
//        }
//      }
//    }

    // Get the size of each bucket, so that we could sample
    // uniformly from the bucket
    private val bucketSizes = data.map(_.length)
    // println("Summary of dataset ==================")
    // buckets.zip(bucketSizes).foreach {
    //   case (bkt, size) => println(s"bucket of len $bkt : $size samples")
    // }

     // make a random data iteration plan
     // truncate each bucket into multiple of batch-size
    private var bucketNBatches = Array[Int]()//Array(1024),the length is the same as the length of data
    for (i <- data.indices) {
      bucketNBatches = bucketNBatches :+ (data(i).length / _batchSize)
      data(i) = data(i).take(bucketNBatches(i) * _batchSize)//delete the redundant bucket
    }

    private val bucketPlan = {//List(0,0,0,...),length=1024
      val plan = bucketNBatches.zipWithIndex.map(x => Array.fill[Int](x._1)(x._2)).flatten
      Random.shuffle(plan.toList)
    }

    private val bucketIdxAll = data.map(_.length).toList
                                        .map(l => Random.shuffle((0 until l).toList))//List(2,5,2,6),a random permutation index
    private val bucketCurrIdx = data.map(x => 0)

    private var dataBuffer = Array[NDArray]()//length is always 1 
    private var labelBuffer = Array[NDArray]()//length is always 1
    for (iBucket <- data.indices) {
      dataBuffer = dataBuffer :+ NDArray.zeros(_batchSize, buckets(iBucket))//(_batchSize,129)
      labelBuffer = labelBuffer :+ NDArray.zeros(_batchSize, buckets(iBucket))
    }

    private val _provideData = {
      val tmp = Map("data" -> Shape(_batchSize, defaultBucketKey))
      tmp ++ initStates.map(x => x._1 -> Shape(x._2._1, x._2._2))
    }
    private val _provideLabel = Map("softmax_label" -> Shape(_batchSize, defaultBucketKey))//(batchsize,129)

    private var iBucket = 0

    override def next(): DataBatch = {
      val bucketIdx = bucketPlan(iBucket)//the index for choice data slice,always 0
      val dataBuf = dataBuffer(bucketIdx)
      val iIdx = bucketCurrIdx(bucketIdx)//the idx that has used
      val idx = bucketIdxAll(bucketIdx).drop(iIdx).take(_batchSize)
      bucketCurrIdx(bucketIdx) = bucketCurrIdx(bucketIdx) + _batchSize

      val datas = idx.map(i => data(bucketIdx)(i)).toArray
      for (sentence <- datas) {
        assert(sentence.length == buckets(bucketIdx))//129
      }
//      println(dataBuf.shape)
//      val a = datas.flatten
//      println(a.length)
      dataBuf.set(datas.flatten)

      val labelBuf = labelBuffer(bucketIdx)
      
      val labels = idx.map(i => data(bucketIdx)(i).sorted).toArray
      
      labelBuf.set(labels.flatten)

      iBucket += 1
      new DataBatch(IndexedSeq(dataBuf.copy),//NDArray，(_batchSize,129)
                    IndexedSeq(labelBuf.copy),//NDArray，(_batchSize,129)
                    getIndex(),
                    getPad())
    }

    /**
     * reset the iterator
     */
    override def reset(): Unit = {
      iBucket = 0
      bucketCurrIdx.indices.map(i => bucketCurrIdx(i) = 0)
    }

    override def batchSize: Int = _batchSize

    /**
     * get data of current batch
     * @return the data of current batch
     */
    override def getData(): IndexedSeq[NDArray] = IndexedSeq(dataBuffer(bucketPlan(iBucket)))

    /**
     * Get label of current batch
     * @return the label of current batch
     */
    override def getLabel(): IndexedSeq[NDArray] = IndexedSeq(labelBuffer(bucketPlan(iBucket)))

    /**
     * the index of current batch
     * @return
     */
    override def getIndex(): IndexedSeq[Long] = IndexedSeq[Long]()

    // The name and shape of label provided by this iterator
    override def provideLabel: Map[String, Shape] = this._provideLabel

    /**
     * get the number of padding examples
     * in current batch
     * @return number of padding examples in current batch
     */
    override def getPad(): Int = 0

    // The name and shape of data provided by this iterator
    override def provideData: Map[String, Shape] = this._provideData

    override def hasNext: Boolean = {
      if (iBucket < bucketPlan.length) true else false
    }
  }
}
