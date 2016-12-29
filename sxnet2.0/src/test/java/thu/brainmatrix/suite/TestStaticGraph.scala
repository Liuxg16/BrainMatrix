
package thu.brainmatrix.suite

import thu.brainmatrix.Base._
import thu.brainmatrix.Symbol
import thu.brainmatrix.StaticGraph
import scala.Vector
import thu.brainmatrix.Shape

/**
 * 
 * 2016-3-25
 * by liuxianggen
 * as the objuect name says
 * 
 */
object staticGraphTest {
	def main(args:Array[String]){
	
//		identifyTest
		toStaticGraphTest
//		handleTest
	}
	
	def toStaticGraphTest{
		val dataS = Symbol.CreateVariable("data")
  
  	val kwargs_type = Map("name" -> "fc2", "num_hidden" -> "10")
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "FullyConnectedS")
//  	var  out_graph= new StaticGraph()
  	sb.ToStaticGraph()
  	println(sb.staticGraph.debug)
  	println("--------------------------------------------")
//  	sb.staticGraph.ToStaticGraph
//  	sb.staticGraph.printOperator
		
	}
	
	def OperatorTest{
		val dataS = Symbol.CreateVariable("data")
  
  	val kwargs_type = Map("name" -> "fc2", "num_hidden" -> "10")
    val sb:Symbol = Symbol.Create("FullyConnected",kwargs_type)
  	val kwargs_symbol:Map[String,Symbol] = Map("data"->dataS) 
  	sb.Compose(kwargs_symbol, "FullyConnectedS")
//  	var  out_graph= new StaticGraph()
  	sb.ToStaticGraph()
//  	println(sg.debug)
  	println("--------------------------------------------")
  	
  		sb.staticGraph.printOperator
		
	}
	
	
  /**
   * 2016-3-23
   */
  def  identifyTest{
//  		def ToStaticGraph(out_graph: StaticGraph) {
  	
  	val sg:StaticGraph = new StaticGraph()
  	val dataS = Symbol.CreateVariable("data1")
//  	val weightS = Symbol.CreateVariable("weight")
//  	val biasS = Symbol.CreateVariable("bias")
  	val sb:Symbol = Symbol.Create("FullyConnected")
//  	val kwargs:Map[String,Symbol] = Map("data"->dataS,"weight"->weightS,"bias"->biasS)
  	val kwargs:Map[String,Symbol] = Map("data"->dataS)
  	
  	sb.Compose(kwargs, "FullyConnectedS")
  	
//  	val weightS1 = Symbol.CreateVariable("weight1")
//  	val biasS1 = Symbol.CreateVariable("bias1")
  	val sb1:Symbol = Symbol.Create("FullyConnected")
    val kwargs1:Map[String,Symbol] = Map("data"->sb)
    sb1.Compose(kwargs1, "FullyConnectedS1")
  
  	sb1.ToStaticGraph()
  	
  	val kwargs_ :Map[String,Shape] = Map("data"->Shape(10,20),"data1"->Shape(2,4))
  	val (a, b) = 	sb1.staticGraph.identifyVar(kwargs_)
  	a.foreach {println}
  	

  }
  
  /**
   * by liuxianggen 
   * 2016-4-4
   */
  def handleTest{
  	val sg = new StaticGraph()
  	println(sg.handle)
  }
  	
}