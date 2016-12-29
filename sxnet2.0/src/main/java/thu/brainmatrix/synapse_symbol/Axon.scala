package thu.brainmatrix.synapse_symbol

import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.Executor

class Axon(val ctx: Context = Context.defaultCtx,val name:String) extends Module {
  
	
	
    override var variable_table = Array[String]("preVm")
	override var variableindices = Array(-1)
    
	//connectivity
	var synapses = Vector[Synapse]();
    var input :Input = null
    
    var input_s:Symbol = null
    
    override def getSymbol() = this.input_s
    
    // graphic model
	val gK         = Symbol.CreateVariable(s"gK_$name")
	val Vk         = Symbol.CreateVariable(s"Vk_$name")
	val Cm         = Symbol.CreateVariable(s"Cm_$name")
	val SensorIn   = Symbol.CreateVariable(s"SensorIn_$name")
	var preVm      = Symbol.CreateVariable(s"preVm_$name")
	var freeSensor = Symbol.CreateVariable(s"freeSensor_$name")
	    
	val onenda = NDArray.ones(Config.SHAPE,ctx)
	
	//parameters
	var gK_nda      :NDArray = onenda;
	var Vk_nda      :NDArray =  onenda* -70f;
	var Cm_nda      :NDArray =  onenda * 10f; // membran capacitance
	var SensorIn_nda:NDArray = onenda * 2;
    
	//others
    var freeSensor_nda:NDArray =  onenda * 0f

    // variables
    var preVm_nda: NDArray =  onenda * -70f
    var y_preVm_nda: NDArray =  onenda * -70f
    
    override def getSymbolMap():Map[String,NDArray] = {
    	Map(s"gK_$name"->gK_nda,s"Vk_$name"->Vk_nda,s"Cm_$name"->Cm_nda,s"SensorIn_$name"->SensorIn_nda,
    			s"preVm_$name"->y_preVm_nda,s"freeSensor_$name"->freeSensor_nda,s"current_${this.input.name}"->this.input.current_nda)
    }

    
    
//    def setValue(gK: NDArray,Vk: NDArray,Cm: NDArray,SensorIn: NDArray,preVm: NDArray){
//    	
//    	this.gK_nda = gK;
//    	this.Vk_nda = Vk;
//    	this.Cm_nda = Cm;
//    	this.SensorIn_nda = SensorIn;
//    	this.preVm_nda = preVm;
//    }
    
    def getSynapses(idx:Int):Synapse = {
    	synapses(idx)
    }        
    
    def addSynapse(s:Synapse){
    	s.axon = this;
    	synapses = synapses.:+(s);
    } 
      
    def addSpikeInput(input:Input){
    	this.input = input;
    	
    }
    
    override def getInitialY():Array[NDArray] = {
    	Array(this.y_preVm_nda)
    }

    override def getInitialVar():Array[String] = {
  		Array(s"y${this.variableindices(0)}")
  	}
    
    override def getInitial(map : Map[String,NDArray]): Map[String,NDArray] = {
  		Map(s"y${this.variableindices(0)}"->this.y_preVm_nda)
  	}
    
    /**
     * indices: the variable indexs that this module needs
     * vector operations
     */
    override def update(t_onehot: Symbol, y:Array[Symbol],yDot:Array[ Symbol],indices:Array[Int]):Array[Symbol] = {
    	this.preVm = y(indices(0))
	
		this.input_s = this.input.getinput(t_onehot);
    	
		val d_preVm = ((input_s+this.gK*(this.preVm-this.Vk))/this.Cm)*(-1f); 
	     
	        // Sensor can diffuse between synapses
	    this.freeSensor = this.SensorIn;
	    for(i <- 0 until this.synapses.length){
	        	this.freeSensor = this.freeSensor - this.synapses(i).preSensor;
//	        	println("dddddddddddd")
	    }
            
	    yDot(indices(0)) = d_preVm;
        
        yDot
    }
    
}

