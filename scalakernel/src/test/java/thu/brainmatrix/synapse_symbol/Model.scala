package thu.brainmatrix.synapse_symbol
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.Symbol
class Model(val ctx:Context) {
  
	var modules = Vector[Module]();
	var indices = Vector[Array[Int]]();
	var variables:Array[String] = Array[String]() 
	var varNumber :Int = 0;
	
	var initialMap = Map[String,NDArray]()
	
	var symbolMap = Map[String,NDArray]()
	
	var initialVector = Vector[NDArray]();
	
	var initialName   = Vector[String]()
	
	var model_sym:Symbol = null
	
	
	def addModule(module:Module){
		
		//add modules
		this.modules :+= (module);
		// set indices in each module
		module.setIndices(this.varNumber);
		// update the number of variable number
		this.varNumber += module.getVarNumber();
		
		//add initial numbers
		for(i <- 0 until module.getInitialY().length){
			initialVector :+= (module.getInitialY()(i));
			initialName   :+= module.getInitialVar()(i)
		}
		
		// add the variable indices 
		this.indices :+= (module.getVarIndices());
		
		this.symbolMap ++= module.getSymbolMap()
		
		//add initial numbers
		this.initialMap ++= module.getInitial()
	}
	
	
	
	
	def update():Symbol = {
		// TODO Auto-generated method stub
		val t_onehot = Symbol.CreateVariable("t_onehot")
		val y = (for(i<- 0 until this.varNumber) yield {
			Symbol.CreateVariable(s"y$i")
		}).toArray
//		val y = Array.fill[Symbol](this.varNumber)(Symbol.CreateVariable("y0"))
		
		var yDot:Array[Symbol] = y
		
		for(i <- 0 until this.modules.length){
			
			
			yDot = this.modules(i).update(t_onehot, y, yDot,this.modules(i).getVarIndices());
		}
		this.model_sym = Symbol.Group(yDot:_*)
		this.model_sym
	}
	
	def getInitialMap(): Map[String,NDArray] = {
		val vec = 
		this.initialVector zip this.initialName map{case(x,y)=>{
			(y->x)
		}}
		
		vec.toMap
	}
	
	def getInitialY():Array[NDArray] = {
		val indicess = this.indices.flatten
		indicess.indices.map(i=>{
			this.initialVector(indicess(i))
		}).toArray
		
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