
package thu.brainmatrix.lstmSort

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import thu.brainmatrix.util.IOHelper
import thu.brainmatrix.lstmSort.ButketIo
class LstmSortSuite extends FunSuite with BeforeAndAfterAll {
  	test(""){
  		val path_train = "./data/sort.train.txt"
		val path_test = "./data/sort.valid.txt"
	    val batch_size = 100
	    val buckets = List(5)
	    val num_hidden = 300
	    val num_embed = 512
	    val num_lstm_layer = 2
		val seqLen = 5
	    val num_epoch = 8
	    val learningRate = 0.1f
	    val momentum = 0.9
	//    # a dict that contains the  word and the index
	    val vocab =  IOHelper.buildVocab("./data/sort.train.txt")
	    println(vocab)

        // initalize states for LSTM
      	val initC = for (l <- 0 until num_lstm_layer) yield (s"l${l}_init_c", (batch_size, num_hidden))
	  	val initH = for (l <- 0 until num_lstm_layer) yield (s"l${l}_init_h", (batch_size, num_hidden))
	  	val initStates = initC ++ initH
  		val dataTrain = new ButketIo.BucketSentenceIter(path_train, vocab, buckets,batch_size, initStates)
  		
  		val batch = dataTrain.next()
  		println(batch.data(0))
  		println(batch.label(0))
  	}
}