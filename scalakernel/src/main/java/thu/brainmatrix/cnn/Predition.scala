package thu.brainmatrix.cnn

import java.io.File

import com.sksamuel.scrimage.Image
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.Model
import thu.brainmatrix.Symbol

import thu.brainmatrix.util.CVTool

object Predition {
    val newWidth = 28
    val newHeight = 28
    val ctx = Context.cpu(0)
    
    val (_, argParams, _) = Model.loadCheckpoint("lenet", 10)
      
      
      val sym  = TestTraininglxg.getLenet()
      
      val inputShape = Map("data"->Shape(1,1,newWidth,newHeight))
      
      val executor = sym.simpleBind(ctx = ctx, shapeDict = inputShape)

      for (key <- executor.argDict.keys) {
        if (!inputShape.contains(key) && argParams.contains(key) && key != "sm_label") {
          argParams(key).copyTo(executor.argDict(key))
        }
      }
      

    
    
    
    
    def pred(picPath:String):Float = {
      val img = Image(new File(picPath))
      val resizedImg = img.scaleTo(newWidth, newHeight)
      
      val rgbs = resizedImg.iterator.toArray.map { p =>
        (255*3f-(p.blue+p.red+p.green))/(3.0f*255)
      }
      val inputData = NDArray.array(rgbs, Shape(1,1,newWidth,newHeight), ctx)
      
      
      CVTool.saveFlattenImage(inputData, "checkpic")
      
      inputData.copyTo(executor.argDict("data"))
      executor.forward()
      
      val prob = executor.outputs(0)
      
      val index = NDArray.argmaxChannel(prob).toScalar
      
      
      index
    }
    
    def main(args:Array[String]){
      println(pred("/home/agen/workspace-python/flask/recognizer/output.png"))
    }
}