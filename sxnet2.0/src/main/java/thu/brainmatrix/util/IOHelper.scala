package thu.brainmatrix.util
import scala.io.Source
import thu.brainmatrix._
import scala.collection.immutable.Set
object IOHelper {

	
	def read_content(path:String):String = {
		val content = Source.fromFile(path).mkString
		content.replaceAll("\n"," <eos> ")
	}
	  // Build  a vocabulary of what word we have in the content
	def buildVocab(path: String): Map[String, Int] = {
	    val content = read_content(path)
	    var words = content.split(" ")
	    var vocab = words.filter { _.length()>0 }.toSet
//	    words.foreach {println}
	    val vocabs = vocab.toArray.sorted
	    
	    var idx = 1 // 0 is left for zero padding
	    var theVocab = Map[String, Int]()
	    for (word <- vocabs) {
	        if (!theVocab.contains(word)) {
	          theVocab = theVocab + (word -> idx)
	          idx += 1
	        }
	      }
	    theVocab
	  }
	
	def doCheckpoint(prefix: String): EpochEndCallback = new EpochEndCallback {
	    override def invoke(epoch: Int, symbol: Symbol,
	                        argParams: Map[String, NDArray],
	                        auxStates: Map[String, NDArray]): Unit = {
	      Model.saveCheckpoint(prefix, epoch + 1, symbol, argParams, auxStates)
	    }
  }
	
	
	
}