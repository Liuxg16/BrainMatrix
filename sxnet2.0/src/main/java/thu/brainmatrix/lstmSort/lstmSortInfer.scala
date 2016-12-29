package thu.brainmatrix.lstmSort
import thu.brainmatrix._
import thu.brainmatrix.util.IOHelper
import thu.brainmatrix.lstmSort.RnnModel.Bi_LSTMInferenceModel

object lstmSortInfer {
  
	
	def main(args:Array[String]){
		val path_train = "./data/sort.train.txt"
		val path_test = "./data/sort.valid.txt"
		val saveModelPath = "./model"
	    val batch_size = 100
	    val buckets = List(5)
	    val num_hidden = 300
	    val num_embed = 512
	    val num_lstm_layer = 2
		val seqLen = 5
	    val num_epoch = 1
	    val learningRate = 0.01f
	    val momentum = 0.9
	    
	    val ctx = Context.gpu(0)
	//    # a dict that contains the  word and the index
	    val vocab =  IOHelper.buildVocab("./data/sort.train.txt")
	    var bacov = for((k,v)<- vocab) yield (v,k)
		// load from check-point
        val (_, argParams, _) = Model.loadCheckpoint(s"${saveModelPath}/lstmSort", 1)
        
        val model = new Bi_LSTMInferenceModel(numLstmLayer = 2, inputSize = vocab.size, seq_len = 5, 
        		numHidden = num_hidden,numEmbed = num_embed, numLabel = vocab.size, argParams = argParams)
        println("----------------")
//        S0V4C 94TMV NDKQ2 NEJVU GW2CJ
//		KS51G 1KMG4 2R6OQ NDKQ2 FA4HP
        val inputString = "S0V4C 94TMV NDKQ2 NEJVU GW2CJ"
        val inputS =inputString.split(" ").map(vocab(_).toFloat) 
        val data = NDArray.array(inputS,Shape(1,inputS.length))
        val prob = model.forward(data)
        (NDArray.argmaxChannel(prob)).toArray.map(x => println(bacov(x.toInt)))
        
	}
/**
 * 	S0V4C 94TMV NDKQ2 NEJVU GW2CJ
 * 	94TMV
	GW2CJ
	NEJVU
	NEJVU
	S0V4C
	note:这例子可以看到，输出都是排好序的，但是输出的字符串不一定都是输入的字符串。这说明LSTM并没有完全学到如何排序，但基本学会了。
	
 */
	
	
	
}