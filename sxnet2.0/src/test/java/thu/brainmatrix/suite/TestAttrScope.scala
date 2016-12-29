package thu.brainmatrix.suite

import thu.brainmatrix.AttrScope
import thu.brainmatrix.Symbol

object TestAttrScope {
  def main(args:Array[String]){
	  
	  val (data, gdata) =AttrScope(Map("group" -> "4", "data" -> "great")).withScope {
//	      val data = Symbol.CreateVariable("data", attr = Map("dtype" -> "data", "group" -> "1"))
	      val data = Symbol.CreateVariable("data")
	      val gdata = Symbol.CreateVariable("data2")
	      (data, gdata)
      }
	   
//	  println(gdata.attr("group").get)
//	  println(data.attr("group").get)
//    assert(gdata.attr("group").get === "4")
//    assert(data.attr("group").get === "1")

      val exceedScopeData = Symbol.CreateVariable("data3")
//    assert(exceedScopeData.attr("group") === None, "No group attr in global attr scope")
  }
}