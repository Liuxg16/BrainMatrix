package thu.brainmatrix.util
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
     
     
     // can process (n,1,-1,-1) ndarray
     def saveFlattenImage(img: NDArray, filename: String): Unit = {   
    	  
    	val shape = img.shape
    	assert(shape.length == 4)

    	val (n, c, h, w) = (shape(0), shape(1), shape(2), shape(3))
    	
    	val spatialSize = n*c*h*w 
    	
    	val totals = h * w
    	img *= 255
    	val row, col = Math.sqrt(n).toInt
    	
    	val imgs_rows = (0 until row) map{r => 
    		val imgs_row = (0 until col) map{c =>
    			img.slice(r*col+c).reshape(Array(h,w))
    		}
    		val temp = NDArray.transpose(NDArray.concatenate(imgs_row:_*))
    		imgs_row.foreach {_.dispose()}
    		temp
    	}
    	
    	val imgs_nda = NDArray.transpose(NDArray.concatenate(imgs_rows:_*))
    	imgs_rows.foreach {_.dispose()}
    	
    	val out = process2DImage(imgs_nda)
    	imgs_nda.dispose()
    	
    	out.output(filename)(JpegWriter())
   	
//    	val lineArrs = rawData.grouped(col * c * totals)
//    	for (line <- lineArrs) {
//    		val imgArr = line.grouped(c * totals)
//    		for(arr <- imgArr) 
//    			src.add(getImg(arr, c, h, w, flip))
//    		
//    	}
//    	
//    	val pixels = for (i <- 0 until spatialSize)
//          yield Pixel(r(i).toInt, g(i).toInt, b(i).toInt, 255)
//        Image(img.shape(3), img.shape(2), pixels.toArray)
//    	
//    	  
//       val out = postprocessImage(img)
//       val gauss = GaussianBlurFilter(radius).op
//       val result = Image(out.width, out.height)
//       gauss.filter(out.awt, result.awt)
//       out.output(filename)(JpegWriter())
     }
     
     
     
     def saveGrayImage(img: NDArray, filename: String): Unit = {    
       val out = processGrayImage(img)
       out.output(filename)(JpegWriter())
  	}
  
     
    def saveRGBImage(img: NDArray, filename: String): Unit = {    
       val out = postprocessImage(img)
       out.output(filename)(JpegWriter())
  	} 
     
   
   
    /**
     * @author :Liu Xianggen
     * @date 
     * @brief process:ndarray to image whose shape matches (3,m,n)
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
     * @brief process:ndarray to Gray image
     * @param
     * @return
     * @example
     * @note
     */
    def process2DImage(img: NDArray): Image = {
        
        val datas = img.toArray
        val spatialSize = img.shape(0) * img.shape(1)
        val r = clip(datas.take(spatialSize))
        val g = clip(datas.take(spatialSize))
        val b = clip(datas.take(spatialSize))
        val pixels = for (i <- 0 until spatialSize)
          yield Pixel(r(i).toInt, g(i).toInt, b(i).toInt, 255)
        Image(img.shape(0), img.shape(1), pixels.toArray)
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