package thu.brainmatrix.synapse

import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

class Axon(val ctx: Context = Context.defaultCtx) extends Module {
  
    override var variable_table = Array[String]("preVm")
	override var variableindices = Array(-1)
    
	//connectivity
	var synapses = Vector[Synapse]();
	var input :Input = null 
	
	
	val onenda = NDArray.ones(Config.SHAPE,ctx)
	
	//parameters
	var gK      :NDArray = onenda;
	var Vk      :NDArray = - onenda* 70;
	var Cm      :NDArray =  onenda * 10; // membran capacitance
	var SensorIn:NDArray = onenda * 2;
    
	//others
    var freeSensor:NDArray =  onenda * 0f

    // variables
    var preVm: NDArray =  onenda * -70f
    
    
    def setValue(gK: NDArray,Vk: NDArray,Cm: NDArray,SensorIn: NDArray,preVm: NDArray){
    	
    	this.gK = gK;
    	this.Vk = Vk;
    	this.Cm = Cm;
    	this.SensorIn = SensorIn;
    	this.preVm = preVm;
    }

    
    
	
    
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
    
    override def getInitial():Array[NDArray] = {
    	Array(this.preVm)
    }

    
    
    
    /**
     * indices: the variable indexs that this module needs
     * vector operations
     */
    override def update(t: NDArray, y:Array[NDArray],yDot:Array[ NDArray],indices:Array[Int]):Array[NDArray] = {
    	this.preVm = y(indices(0))
    	
    	val input = this.input.getinput(t);
//    	val input = NDArray.zeros(Config.SHAPE, ctx)
    	
//    	println(this.preVm.shape)
//    	println(input.context)
    	
    	val d_preVm = - (input+this.gK*(this.preVm-this.Vk))/this.Cm; 
        
        // Sensor can diffuse between synapses
        this.freeSensor = this.SensorIn;
        for(i <- 0 until this.synapses.length){
        	this.freeSensor = this.freeSensor - this.synapses(i).preSensor;
        }
        
        yDot(indices(0))=d_preVm;
        
        input.dispose()
        
        yDot
    }
}