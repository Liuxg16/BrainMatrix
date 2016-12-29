package thu.brainmatrix.nce_loss
import thu.brainmatrix._

class NceAccuracy extends EvalMetric("NceAccuracy") {
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
  	val label = NDArray.argmaxChannel(labels(1))
    val pred = NDArray.argmaxChannel(preds(0))
  	
    for ((labelElem, predElem) <- label.toArray zip pred.toArray) {
        if (math.abs(labelElem - predElem)<1e-6) {
//        	println(s"labelElem:$labelElem,predElem:$predElem")
          this.sumMetric += 1
        }
      }
      this.numInst += pred.shape(0)
      pred.dispose()
      label.dispose()
    }
    
    
}