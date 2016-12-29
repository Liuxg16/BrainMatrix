package thu.brainmatrix.util
import scala.collection.mutable.ArrayBuffer
import thu.brainmatrix.NDArray
import thu.brainmatrix.Shape
import thu.brainmatrix.Context
//import util.Draw
//import util.ArrOps
/** 
 * @Author: Xianggen Liu
 * Runge-Kutta method
 * functions: the functions：
 * f(t,dy0,y1,y2,y3,...) = dy0,dy1,dy2,dy3,...  each element is a vector, the shapes of every element are the same
 * RKM4.solve([y1_0,y2_0,y3_0,...],stepSize,endStep)
 * the length of return of function equals to the numbers of parameters -1. In other words, y.length == dy.length
 * note: There is one difference with the class RKM4, its parameter is only a function!
 */
class RK4(val function:((NDArray,Array[NDArray]) => Array[NDArray])) {
  
	/**
	 * num_parallel
	 * num_funcions
	 * inputs:
	 * t0: 1 * num_parallel
	 * y_init: num_functions  [1 * num_parallel]
	 * a array from y(0)-y(n), y(i) is a vector for parallel
	 * h :   1 * num_parallel
	 * Num_Step
	 * 
	 * note: not very precise, becuase it only integrate to k*delta, which is a little smaller than n
	 * returns:
	 * 1: times => record of the step each episode
	 * 2: record of the y(0)-y(n), y(i) is matrix ,each line restores the episode of y(i)
	 */
	
	def solve(t0:NDArray, y_init:Array[NDArray],h:NDArray,Num_Step:Int)(ctx:Context=Context.cpu()) :(NDArray,Array[NDArray])= {
		/**
		 * ts Num_Step * num_parallel 
		 * res:  num_functions * [Num_Step * num_parallel]
		 */
		
		var yt = y_init;
		var t = t0
		
		
		var delta = h
		val ts = NDArray.zeros(Shape(Num_Step,t0.shape(1)),ctx)
		val res = Array.fill(y_init.length)(NDArray.zeros(Shape(Num_Step,t0.shape(1)), ctx))
		var step = 0
		
		while(step < Num_Step){
			println("step:"+step)
	
			t.copyTo(ts.slice(step))
//			println("lemonman")
			yt = this.right_calculate(yt, t, delta) 
//			println("lemonman")
			yt.indices.foreach {i => yt(i).copyTo(res(i).slice(step)) }
			t = t + h
			step += 1
		}
		(ts,res)
	}
	
	
	/**
	 * t: 1 * num_parallel
	 * y: num_functions * [1 * num_parallel]
	 * delta: 1 * num_parallel
	 */
	def right_calculate(y:Array[NDArray],t:NDArray,delta:NDArray):Array[NDArray] = {
		/**
		 * The fourth-order Runge-Kutta method requires four evaluations of the 
		 * right-hand side per step h
		 */
		
		val function = this.function
//		println("lemonman1")
		val k1 = function(t,y).map(_ * delta).toArray
//		println("lemonman1")
		val k2 = function(t+ delta*0.5f, y.indices.map(i =>{ y(i)+(delta*0.5f)*k1(i)}).toArray).map(_ * delta).toArray 
		
		val k3 = function(t+ delta*0.5f,y.indices.map(i =>{ y(i)+ (delta*0.5f) * k2(i)}).toArray).map(_ * delta).toArray

		val k4 = function(t+ delta,y.indices.map(i =>{ y(i)+ delta * k3(i)}).toArray).map(_ * delta).toArray

		val res = y.indices.map(i => {y(i) + (k1(i)+k2(i)*2f+ k3(i)*2 +k4(i))/6f}).toArray	
		
		k1.foreach { _.dispose()}
		k2.foreach { _.dispose()}
		k3.foreach { _.dispose()}
		k4.foreach { _.dispose()}
//        y .foreach { _.dispose()} 		
		
		res
		
	}
	
	
	
	
	
}

object  RK4{
	
	//sketch a cirlce
	def main(args:Array[String]){
		
		def xdot(t:NDArray,y:Array[NDArray]) = NDArray.cos(t)
		def ydot(t:NDArray,y:Array[NDArray]) = -y(0)		//cos
		type f_Norm = (Double, Array[Double]) => Double
		def f(t:NDArray,y:Array[NDArray]) = Array(xdot(t,y),ydot(t,y))  
//		
		val ctx = Context.gpu();
		val rkm4 = new RK4(f)
		val num_funs = 2
		val num_parallel = 10
		val y0_0 = NDArray.array(Array(0f,1f,0f,1f,0f,1f,0f,1f,0f,1f), Shape(num_parallel,1), ctx)
		val y0_1 = NDArray.array(Array(1f,0f,1f,0f,1f,0f,1f,0f,1f,0f), Shape(num_parallel,1), ctx)
		val h = NDArray.ones(Shape(num_parallel,1),ctx) * (0.01 * 2* math.Pi).toFloat
		val t0 = NDArray.zeros(Shape(num_parallel,1),ctx)
		val y_init = Array(y0_0,y0_1)
		val (t,y) = rkm4.solve(t0, y_init,h ,100)(ctx)
		
		val rest = NDArray.transpose(t).slice(0).toArray
		val resx = NDArray.transpose(y(0)).slice(0).toArray
		val resy = NDArray.transpose(y(1)).slice(0).toArray
		
		
		val draw = new Draw()
//		draw.add_line(res,resx)
		draw.add_line(resx,resy)
		draw.draw()
	
		
		
		
	}
}