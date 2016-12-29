package thu.brainmatrix.util

import breeze.linalg._
import breeze.plot._

class Draw(val subplots:Int*){
	val f = Figure()
	var p:Plot = f.subplot(0)
	
	def subplot(row:Int, col:Int, selected:Int){
			this.p = f.subplot(row,col,selected)
	} 
	
	def add_line[@specialized(Int, Float, Double) V,@specialized(Int, Float, Double) V1](x: Array[V],y:Array[V1],style:Char = '-'){
	  if(x.length == y.length){
	 	  val xa = DenseVector.create(x.map(_.toString().toDouble),0,1,x.length)//choose all
		  val ya = DenseVector.create(y.map(_.toString().toDouble),0,1,y.length)//choose all
		  this.p     += plot(xa, ya,style)
	  }else{
	 	  throw new java.lang.VerifyError("the data for two axis dismatched!")
	  }
  }
	
	def addInfo(xlabel:String, ylable:String,title:String = null){
		this.p.xlabel = xlabel
		this.p.ylabel = ylable
		if(title!=null)
			this.p.title  = title
	} 
	
	
	def add_hist[@specialized(Int, Float, Double) V](x: Array[V],n_hist:Int = 10){
		this.p +=hist(x.map(_.toString().toDouble), n_hist)
	}
	def draw(){
//		  p.xlabel = "x axis"
//		  p.ylabel = "y axis"
		  f.saveas("lines.png") 
	}
}


object Util {
  def Util_plot[@specialized(Int, Float, Double) V,@specialized(Int, Float, Double) V1](x: Array[V],y:Array[V1]){
	  val f = Figure()
	  val p = f.subplot(0)
	  if(x.length == y.length){
	 	  val xa = DenseVector.create(x.map(_.toString().toDouble),0,1,x.length)//choose all
		  val ya = DenseVector.create(y.map(_.toString().toDouble),0,1,y.length)//choose all
		  p     += plot(xa, ya)
		  p.xlabel = "x axis"
		  p.ylabel = "y axis"
		  
		  f.saveas("lines.png") 
	  }else{
	 	  throw new java.lang.VerifyError("the data for two axis dismatched!")
	  }
  }
  
  def hist_test(){
	  val f = Figure()
	  val p = f.subplot(0)
	  val x = Array.fill[Float](1000)(0.7f)
	  x.indices.foreach(i => {
	 	  x(i) = scala.util.Random.nextFloat()
	  })
//	  val x = (0, 1000).map(_.toFloat/1000)
	  
//	  x(3) = 0.08f
//	  x.foreach(print)
//	  val y = Array.range(0, 10)
// 	  val xa = DenseVector.create(x.map(_.toString().toDouble),0,1,x.length)//choose all
//	  val ya = DenseVector.create(y.map(_.toString().toDouble),0,1,y.length)//choose all
//	  val g = breeze.stats.distributions.Gaussian(0,1)
//	  val gs = g.sample(100)
//	  gs.foreach(print(_))
//	  p += hist(g.sample(100000),10)
	  p +=hist(x,100)
	  f.saveas("lines.png") 
  }
  
   def plot_test(){
	  val f = Figure()
val p2 = f.subplot(2,1,1)
val g = breeze.stats.distributions.Gaussian(0,1)
p2 += hist(g.sample(100000),100)
p2.title = "A normal distribution"
f.saveas("subplots.png")
  }
  
  
  def main(args:Array[String]){
//	  Util_plot(Array(1,2,3), Array(3f,40f,5f))
//	  hist_test()
	  plot_test()
  }
}