package thu.brainmatrix.synapse

import thu.brainmatrix.NDArray

abstract class Module {
	
  	var variable_table:Array[String]
	var variableindices:Array[Int]
	
	def getInitial(): Array[NDArray] = {
  		Array.fill[NDArray](variable_table.length)(null)
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
	
	def update(t: NDArray, y:Array[NDArray],yDot:Array[NDArray],indices:Array[Int]):Array[NDArray] = {
		Array.fill[NDArray](y.length)(null)
	} 
	
}