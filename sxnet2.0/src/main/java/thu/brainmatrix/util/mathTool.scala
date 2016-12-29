package thu.brainmatrix.util
import thu.brainmatrix.NDArray
import thu.brainmatrix.Shape
import scala.util.control.Breaks
//import org.opencv.core.Core;
//import org.opencv.core.CvType;
//import org.opencv.core.Mat;
//import org.opencv.core.MatOfDouble
//import org.opencv.highgui.Highgui;

object mathTool {
    
	// Evaluation
  def perplexity(label: NDArray, pred: NDArray): Float = {
    val shape = label.shape
    val size = shape(0) * shape(1)
    val labelT = {
      val tmp = label.toArray.grouped(shape(1)).toArray
      val result = Array.fill[Float](size)(0f)
      var idx = 0
      for (i <- 0 until shape(1)) {
        for (j <- 0 until shape(0)) {
          result(idx) = tmp(j)(i)
          idx += 1
        }
      }
      result
    }
    var loss = 0f
    val predArray = pred.toArray.grouped(pred.shape(1)).toArray
    for (i <- 0 until pred.shape(0)) {
      loss += -Math.log(Math.max(1e-10, predArray(i)(labelT(i).toInt)).toFloat).toFloat
    }
    loss / size
  }

	
	
  def output_accuracy(pred: NDArray, target: NDArray): Float = {
    val num_instance = pred.shape(0)
    val eps = 1e-6
    var right = 0
    for (i <- 0 until num_instance) {
      var mx_p = pred(i, 0)
      var p_y: Float = 0
      for(j <- 0 until 5){
        if(pred(i,j) > mx_p){
          mx_p = pred(i,j)
          p_y = j
        }
      }
      if(scala.math.abs(p_y - target(i)) < eps) right += 1
    }
    right * 1.0f / num_instance
  }
 
  /**
   * @author guoshen
   * @date 2016/7/21
   * @brief
   * 通过加权的方式进行概率抽样，主要思路如下：
   *   假设，概率分布为pro[0.2,0.3,0.5]
   *   那么计算一个概率和数组sum[0.2,0.5,1.0]
   *   然后随机生成一个[0,1]之间的数x，将x与sum里面的数依次比较
   *   选择第一个比x大的sum，不妨设sum[i]>=x
   *   返回sum[i]的index -> i
   * @source http://blog.csdn.net/blueyyc/article/details/51538885
   */
  def SampleByPro1D(pro: NDArray): Int = {
	  var require_flag = true
	  
	  pro.shape.toVector match {
	 	  case Vector(x,y) => if(x==1 ||y==1) require_flag = true  
	 	  case Vector(x)   => require_flag = true
	 	  case _           => require_flag = false
	  }
	   
	  
	   if(!require_flag)
	 	  throw new Exception("the parameter wrong!!")
	val proArr = pro.toArray
//	  require(pro.shape.length==1 || pro.shape)
    var sum: Array[Float] = NDArray.zeros(pro.shape).toArray
    var temp_sum: Float = 0
    for (i <- 0 until proArr.size) {
      temp_sum += proArr(i)
      sum(i) = temp_sum
    }
    var rand = Math.random().toFloat
    var res = 0
    val loop = new Breaks
    loop.breakable {
      for (i <- 0 until sum.length) {
        if (rand <= sum(i)) { res = i; loop.break() }
      }
    }

    res
  }
  
  	/**
   * @author guoshen
   * @date 2016/7/21
   * @brief
   * 通过加权的方式进行概率抽样，主要思路如下：
   *   假设，概率分布为pro[0.2,0.3,0.5]
   *   那么计算一个概率和数组sum[0.2,0.5,1.0]
   *   然后随机生成一个[0,1]之间的数x，将x与sum里面的数依次比较
   *   选择第一个比x大的sum，不妨设sum[i]>=x
   *   返回sum[i]的index -> i
   * @source http://blog.csdn.net/blueyyc/article/details/51538885
   */
  def SampleByPro2D(pro: NDArray): Array[Int] = {
	  
	  var require_flag = true
	  var (rows,cols) = (0,0)
	  pro.shape.toVector match {
	 	  case Vector(x,y) =>{
	 	 				require_flag = true
	 	 				rows = x
	 	 				cols = y
	 	  }
	 	  case _           => require_flag = false
	  }
	  if(!require_flag)
	 	  throw new Exception("the parameter wrong!!")
	  val sample_arr = for(i <- 0 until rows) yield{
					 	  val proi = pro.slice(i)
					 	  SampleByPro1D(proi)
	  					}
	  require(sample_arr.length==rows,s"required:$rows, found:${sample_arr.length}")
	  
	  sample_arr.toArray
  }
  
  
  
//  /**
//   * arr: an arrary of one dimension
//   * return the mat with the shape:rows x cols
//   * 
//   */
//  def ArrayToMat(arr:Array[Float],rows:Int,cols:Int):Mat={
//	   System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//	   val m:Mat = Mat.eye(cols, rows, CvType.CV_8UC1);
//	   for(i<-0 until cols;j<-0 until rows){
//	  	   m.put(i,j,arr(j+i*rows))
//	   }
//	   m
//  }
//  
//  /**
//   * arr: an arrary of one dimension
//   * return the mat with the shape:rows x cols
//   * 
//   */
//  def NDArrayToMat(nda:NDArray):Mat={
//	  if(nda.shape.length!=2)
//	 	  throw new java.lang.UnsupportedOperationException("This function only surport two dimension NDArray"); 
//	  val arr= nda.toArray
//	  ArrayToMat(arr,nda.shape(0),nda.shape(1))
//  }
//  
//  def showNDArray(nda:NDArray,name:String){
//	  val mat = NDArrayToMat(nda)
//	  Highgui.imwrite(name+".png", mat);
//  }
//  
  
    /**
     * Author: Liuxianggen
     * data:2016-11-10
     * 
     * 
     */
    def times[T](arr:Array[T]){
    	???
    }
  
  def main(args:Array[String]){
//	  val mat =  ArrayToMat(Array(230,230,230,5,6,230),3,2)
//	  val mat = NDArrayToMat(NDArray.ones(3,5)*240)
//	  Highgui.imwrite("image.png", mat);
//	  println("mat:"+mat.dump())
  }
}