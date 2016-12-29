package thu.brainmatrix.synapse_symbol

import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol

abstract class Module {
	
  	var variable_table:Array[String]
	var variableindices:Array[Int]
	
	def getSymbol():Symbol = {
  		null
  	}
  	
  	
  	
  	def getInitialY():Array[NDArray] = {
  		Array[NDArray]()
  	}
  	
  	def getInitialVar():Array[String] = {
  		Array[String]()
  	}
  	
	def getInitial(map : Map[String,NDArray]=null): Map[String,NDArray] = {
  		Map[String,NDArray]()
  	}
	
  	def getSymbolMap():Map[String,NDArray] = {
  		Map[String,NDArray]()
  	}
  	
	def setIndices(indices:Array[Int]){
		this.variableindices=indices;
	}
	
	def setIndices(startIndex:Int){
		var index = startIndex;
		val numvariables=this.variable_table.length;
		this.variableindices = Array.fill[Int](numvariables)(0)
		for(i <- 0 until numvariables){
			this.variableindices(i) = index;
			index = index + 1 
		}
	}
	
	def getVarIndices():Array[Int] =  {
		this.variableindices;
	}
	
	def getVarNumber():Int = {
		this.variable_table.length;
	}
	
	def getVarsName():Array[String] = {
		this.variable_table;
	}
	
	/**
	 * @param name
	 * @return >-1, the index; -1 means null
	 */
	def getResindex(name:String):Int = {
		var res = -1;
		for(i <- 0 until this.variable_table.length){
			if(name.equals(this.variable_table(i)))
				res = this.variableindices(i);
		}
		return res;
	}
	
	
	// t: not the variable time, but the one-hot encode of time
	
	def update(t_onehot: Symbol, y:Array[Symbol],yDot:Array[ Symbol],indices:Array[Int]):Array[Symbol] = {
		Array.fill[Symbol](y.length)(null)
	} 
	
}