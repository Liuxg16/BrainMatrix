package thu.brainmatrix.synapse

import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

class Dendrite(val ctx: Context = Context.defaultCtx) extends Module{
	val onenda = NDArray.ones(Config.SHAPE,ctx)
	var currentinput = NDArray.zeros(Config.SHAPE,ctx);
	
    override var variable_table = Array[String]("postVm")
	override var variableindices = Array(-1)
    
	//connectivity
	var synapses = Vector[Synapse](); 
     
    
    
    
     // parameters
    var  gK :NDArray = onenda               //
    var  Vk :NDArray = onenda * -70f;                   // reversal potential for K channel
    var  Cm :NDArray = onenda * 10;                    // membran capacitance
//    variables
    var  postVm = onenda * -70f;
     
     
    def set(gK:NDArray, Vk:NDArray,Cm:NDArray,postVm:NDArray){
    	 this.gK = gK;
    	 this.Vk = Vk;
    	 this.Cm = Cm;
    	 this.postVm = postVm;
     }
     
     
     
   
     
    def getSynapses(idx:Int) :Synapse = {
     	return synapses(idx);
     }        
     
    def addSynapse(s:Synapse){
    	s.dendrite = this
    	synapses = synapses.:+(s)
     } 
       
     override def getInitial():Array[NDArray] = {
    	Array(this.postVm)
     }
    
    override def update(t: NDArray,y:Array[NDArray],yDot:Array[NDArray],indices:Array[Int]):Array[NDArray] = {
    	 
     	this.postVm = y(indices(0));
     	var postI = onenda
     	if(this.currentinput!=0){
     		postI = this.currentinput;
     	}
     	var tEPSC = NDArray.zeros(Config.SHAPE,ctx);
     	for(i<- 0 until this.synapses.length){
     		tEPSC += this.synapses(i).EPSC;
     	}

        val d_postVm = - (tEPSC+postI+this.gK*(this.postVm-this.Vk))/this.Cm;
                  
        yDot(indices(0)) = d_postVm; 
     	yDot
     	
     }
    
}