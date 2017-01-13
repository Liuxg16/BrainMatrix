package thu.brainmatrix.synapse_symbol

import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

class Dendrite(val ctx: Context = Context.defaultCtx,val name:String) extends Module{
	val onenda = NDArray.ones(Config.SHAPE,ctx)
	
	
    override var variable_table = Array[String]("postVm")
	override var variableindices = Array(-1)
    var tmp_symbol :Symbol      = null
	override def getSymbol() = this.tmp_symbol
	
	
	//connectivity
	var synapses = Vector[Synapse](); 
     
    
	
    // symbol graph
	
    var  gK = Symbol.CreateVariable(s"gK_$name")
    var  Vk = Symbol.CreateVariable(s"Vk_$name")               // reversal potential for K channel
    var  Cm = Symbol.CreateVariable(s"Cm_$name")
    var  postVm = Symbol.CreateVariable(s"postVm_$name")
     
//    var currentinput = Symbol.CreateVariable(s"currentinput")//no use 
    
    
    
    
    
     // parameters
    var currentinput_nda = NDArray.zeros(Config.SHAPE,ctx);
    var  gK_nda :NDArray = onenda               //
    var  Vk_nda :NDArray = onenda * -70f;                   // reversal potential for K channel
    var  Cm_nda :NDArray = onenda * 10;                    // membran capacitance
//    variables
    var  postVm_nda = onenda * -70f;
    var  y_postVm_nda = onenda * -70f;
     
     
//    def set(gK:NDArray, Vk:NDArray,Cm:NDArray,postVm:NDArray){
//    	 this.gK = gK;
//    	 this.Vk = Vk;
//    	 this.Cm = Cm;
//    	 this.postVm = postVm;
//     }
     
     
    def getSynapses(idx:Int) :Synapse = {
     	return synapses(idx);
     }        
     
    def addSynapse(s:Synapse){
    	s.dendrite = this
    	synapses = synapses.:+(s)
     } 
       
    override def getSymbolMap():Map[String,NDArray] = {
    	Map(s"gK_$name"->gK_nda,s"Vk_$name"->Vk_nda,s"Cm_$name"->Cm_nda,s"postVm_$name"->y_postVm_nda)
    }
    
    
     override def getInitial(map : Map[String,NDArray]=null): Map[String,NDArray] = {
    	 if(map==null)
    		
    	 	Map(s"y${this.variableindices(0)}"->this.postVm_nda)
    	 else {
    	 	  map
    	 	}
  	}
     
     override def getInitialY():Array[NDArray] = {
    	Array(this.y_postVm_nda)
     }
     
     override def getInitialVar():Array[String] = {
  		Array(s"y${this.variableindices(0)}")
  	 }
    
    override def update(t_onehot: Symbol, y:Array[Symbol],yDot:Array[ Symbol],indices:Array[Int]):Array[Symbol] = {
    	
     	this.postVm = y(indices(0));
     	this.tmp_symbol = this.postVm*Config.one_s
     	var postI = Config.one_s  // some difference
     	var tEPSC = Config.zero_s
     	for(i<- 0 until this.synapses.length){
     		tEPSC += this.synapses(i).EPSC;
     	}

        val d_postVm = (tEPSC+postI+this.gK*(this.postVm-this.Vk))/this.Cm*(-1);
                  
        yDot(indices(0)) = d_postVm; 
     	yDot
     	
     }
    
}