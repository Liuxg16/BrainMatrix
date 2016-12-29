//import 
package thu.brainmatrix.suite

import thu.brainmatrix.OperatorProperty
import thu.brainmatrix.OperatorProperty

/***
 *  by liuxianggen 
 *  2016-3-9
 *  brief to test the functions in OperatorProperty class
 */
object OPTest {
 
	def main(args:Array[String]){
  	println("<--------------TEST OperatorProperty----------------->")
//  	mapTest
//  	ClassTest
//  	InitTest
//		ListArgsTest
//  	copyTest
//  	ListAuxsTest
  	NumOutputTest
  }
	
	/**
	 *  2016-3-20:Failed to write core dump. Core dumps have been disabled
	 *  solved use :  OperatorProperty *op = static_cast<OperatorProperty*>(opHandle);
	 *  to replace :
	 *   OperatorPropertyReg *e = static_cast<OperatorPropertyReg *>(creator);
   *	 op = e->body();  
	 */
	def ListArgsTest{
		val op = OperatorProperty("FullyConnected")//FullyConnected,Activation
  	op.ListArguments.foreach {println}
	}
	
	
	def ListAuxsTest{
		val op = OperatorProperty("FullyConnected")//FullyConnected,Activation
  	op.ListAuxiliaryStates().foreach {println}
	}
	
	//by liuxianggen
	//2016-3-9
	def InitTest(){
		val keys = Array("act_type") 
		val values = Array("relu")
		val op = OperatorProperty("Activation")//FullyConnected,Activation
		op.Init(keys,values)
		op.printParam()
	}
	
	/**
	 *  2016-3-20:Failed to write core dump. Core dumps have been disabled
	 *  solved use :  OperatorProperty *op = static_cast<OperatorProperty*>(opHandle);
	 *  to replace :
	 *   OperatorPropertyReg *e = static_cast<OperatorPropertyReg *>(creator);
   *	 op = e->body();  
	 */
	def NumOutputTest{
		val op = OperatorProperty("FullyConnected")//FullyConnected,Activation
		
  	    println(op.NumVisibleOutputs())
	}
	
		//by liuxianggen
	//2016-3-9
	def ClassTest(){
		var keys = Array("act_type") 
		var values = Array("relu")
		var op = OperatorProperty("Activation")//FullyConnected,Activation
		op.Init(keys,values)
		op.printParam()

		val keys1 = Array("act_type") 
		val values1 = Array("sigmoid")
		val op1 = OperatorProperty("Activation")//FullyConnected,Activation
		op1.Init(keys1,values1)
		op1.printParam()
	}
	
	
	/**
	 * 2016-3-15
	 * by liuxianggen
	 */
//	def SymbolBaseTest(){
//		var node_order:Vector[NodeRef] = Vector()
//		val oprf = new OperatorPropertyRef
//		val node = new Node(oprf)
//		val noderef = new NodeRef
//		node_order :+= noderef
////		node.inputs :+= new DataEntry(noderef,2)
//		println(node_order.size)
//	}
//	
	/**
	 * 2016-3-15
	 * by liuxianggen
	 * 
	 */
	def mapTest{
		var map1 = Map("a"-> 1,"b"-> 3)		
		var map2 = map1
		map2 += "c"-> 4
		println(map1)
		println(map2)
	}
	
	
	/**
	 * 2016-3-23
	 * by liuxianggen
	 * succeed!
	 */
	def copyTest{
		val op =  OperatorProperty("FullyConnected")
		val op_1 = op.Copy()
		op_1.ListArguments.foreach(println)
	}
	
}