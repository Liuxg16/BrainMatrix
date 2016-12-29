package thu.brainmatrix.lxg
import thu.brainmatrix.NDArray
import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.Pixel
import com.sksamuel.scrimage.filter.GaussianBlurFilter
import com.sksamuel.scrimage.nio.JpegWriter
object CVTool {
  
     def saveImage(img: NDArray, filename: String, radius: Int): Unit = {    
       val out = postprocessImage(img)
       val gauss = GaussianBlurFilter(radius).op
       val result = Image(out.width, out.height)
       gauss.filter(out.awt, result.awt)
       out.output(filename)(JpegWriter())
     }
     
     def saveGrayImage(img: NDArray, filename: String): Unit = {    
       val out = processGrayImage(img)
       out.output(filename)(JpegWriter())
  }
  
   
   
    /**
     * @author 
     * @date 
     * @brief process:ndarray to image
     * @param
     * @return
     * @example
     * @note
     */
    def postprocessImage(img: NDArray): Image = {
        val datas = img.toArray
        val spatialSize = img.shape(2) * img.shape(3)
        val r = clip(datas.take(spatialSize).map(_ + 123.68f))
        val g = clip(datas.drop(spatialSize).take(spatialSize).map(_ + 116.779f))
        val b = clip(datas.takeRight(spatialSize).map(_ + 103.939f))
        val pixels = for (i <- 0 until spatialSize)
          yield Pixel(r(i).toInt, g(i).toInt, b(i).toInt, 255)
        Image(img.shape(3), img.shape(2), pixels.toArray)
      }

    /**
     * @author 
     * @date 
     * @brief process:ndarray to Gray image
     * @param
     * @return
     * @example
     * @note
     */
    def processGrayImage(img: NDArray): Image = {
        
        val datas = img.toArray
        val spatialSize = img.shape(2) * img.shape(3)
        val r = clip(datas.take(spatialSize))
        val g = clip(datas.take(spatialSize))
        val b = clip(datas.take(spatialSize))
        val pixels = for (i <- 0 until spatialSize)
          yield Pixel(r(i).toInt, g(i).toInt, b(i).toInt, 255)
        Image(img.shape(3), img.shape(2), pixels.toArray)
      }

    
    /**
     * @author 
     * @date 
     * @brief process:make all the element in param between [0,255]
     * @param Array[Float]
     * @return
     * @example
     * @note
     */
    private def clip(array: Array[Float]): Array[Float] = array.map { a =>
        if (a < 0) 0f
        else if (a > 255) 255f
        else a
    }
}