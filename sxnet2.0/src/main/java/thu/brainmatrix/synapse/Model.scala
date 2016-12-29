package thu.brainmatrix.synapse
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
class Model(val ctx:Context) {
  
	var modules = Vector[Module]();
	var indices = Vector[Array[Int]]();
	var variables:Array[String] = Array[String]() 
	var varNumber :Int = 0;
	
	var initialVector = Vector[NDArray]();
	
	def addModule(module:Module){
		
		//add modules
		this.modules :+= (module);
		
		//add initial numbers
		for(i <- 0 until module.getInitial().length){
			initialVector :+= (module.getInitial()(i));
		}
		
		// set indices in each module
		module.setIndices(this.varNumber);
		// update the number of variable number
		this.varNumber += module.getVarNumber();
		
		// add the variable indices 
		this.indices :+= (module.getVarIndices());
		
	}
	
	def update(t: NDArray,y:Array[NDArray]):Array[NDArray] = {
		// TODO Auto-generated method stub
		
		var yDot:Array[NDArray] = y.map { x => x.copy() }
		
		for(i <- 0 until this.modules.length){
//			println(s"lemonman3$i")
			yDot = this.modules(i).update(t, y, yDot,this.modules(i).getVarIndices());
		}
//		println("lemonman3")
		yDot
	}
	
	def getInitial():Array[NDArray] = {
		this.initialVector.toArray
//		var temp = Array.fill[NDArray](this.initialVector.length)(NDArray());
//		for(i <- 0 until this.initialVector.length){
//			temp(i)  = this.initialVector(i);
//		}
//		
//		return temp;
	}
	
	def printIndices(){
		for(i <- 0 until this.indices.length){
			for(j<- 0 until this.indices(i).length){
				System.out.print(this.indices(i)(j)+"  ");
			}
			System.out.println();
		}
	}
	
	def printVarsName(){
		for(i <- 0 until this.indices.length){
			var module = this.modules(i);
			for(j<- 0 until module.getVarsName().length){
				System.out.print(module.getVarsName()(j) + "  ");
			}
			System.out.println();
		}
	}
	

	

	
}