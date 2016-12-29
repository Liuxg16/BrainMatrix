package thu.brainmatrix.synapse_symbol


import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

class Synapse(val ctx: Context = Context.defaultCtx,val name:String) extends Module{
  	// a basic synapse
	override var variable_table = Array[String]("preCa","preCaBuff","aPreCDK","preSensor","aPreTrkB","preNR2B","preAbeta","preMg",
    		"postCa","postCaBuff","aPostCN","aPostTrkB","qNMDAR")
	override var variableindices = Array.fill[Int](this.variable_table.length)(-1)
	
	var BDNF :Symbol      = null
	var tmp :Symbol      = null
	override def getSymbol() = this.tmp
	
	
	
    // connectivity
    var axon:Axon = null
    
    var dendrite:Dendrite = null
    
    var Vs       = Symbol.CreateVariable(s"Vs_$name")            // threshold for the activation of VGCC
    var gK       = Symbol.CreateVariable(s"gK_$name")              //
    var Vk       = Symbol.CreateVariable(s"Vk_$name")              // reversal potential for K channel
    var Vr       = Symbol.CreateVariable(s"Vr_$name")              // resting Vm
    var Ve       = Symbol.CreateVariable(s"Ve_$name")              // reversal potential for AMPA channel
    var Vca      = Symbol.CreateVariable(s"Vca_$name")             // reversal potential for Ca flux though NMDAR
    var Cm       = Symbol.CreateVariable(s"Cm_$name")              // membran capacitance
    var Mgo      = Symbol.CreateVariable(s"Mgo_$name")              // [Mg]o = 1.2 mM
    var kMgNMDAR = Symbol.CreateVariable(s"kMgNMDAR_$name")         // NMDAR Mg block
    var kNMDARIn = Symbol.CreateVariable(s"kNMDARIn_$name")          // constitutive insertion of NMDAR
    var kNMDARCa = Symbol.CreateVariable(s"kNMDARCa_$name")         // converting Inmda to Ca flux
    var CaBasal  = Symbol.CreateVariable(s"CaBasal_$name")           // basal Ca influx
    var Mg50     = Symbol.CreateVariable(s"Mg50_$name")                                         // 50//
    var MgSlope  = Symbol.CreateVariable(s"MgSlope_$name")
    
    // sensor
    var aCDK50 		= Symbol.CreateVariable(s"aCDK50_$name")
    var aCDKSlope 	= Symbol.CreateVariable(s"aCDKSlope_$name")
        
    //presynaptic constants
    var preVol 			= Symbol.CreateVariable(s"preVol_$name")    // volume of presynaptic terminals
    var a1 				= Symbol.CreateVariable(s"a1_$name")      // k+ of Ca buffer
    var b1 				= Symbol.CreateVariable(s"b1_$name")      // k- of Ca buffer
    var tPreCaBuffer 	= Symbol.CreateVariable(s"tPreCaBuffer_$name")// presynaptic Ca buffer
    var kSensorDeg 		= Symbol.CreateVariable(s"kSensorDeg_$name")       // rate of sensor degradation
    var a2 				= Symbol.CreateVariable(s"a2_$name")     // k+ of CDK
    var b2 				= Symbol.CreateVariable(s"b2_$name")      // k- of CDK
    var aCDKNR2B50 		= Symbol.CreateVariable(s"aCDKNR2B50_$name")
    var aCDKNR2BSlope 	= Symbol.CreateVariable(s"aCDKNR2BSlope_$name")
    var a4 				= Symbol.CreateVariable(s"a4_$name")       // k+ of pre TrkB
    var b4 				= Symbol.CreateVariable(s"b4_$name")      // k- of pre TrkB
    var aPreTrkB50 		= Symbol.CreateVariable(s"aPreTrkB50_$name")
    var aPreTrkBSlope 	= Symbol.CreateVariable(s"aPreTrkBSlope_$name")
    var kpreNR2BIn 		= Symbol.CreateVariable(s"kpreNR2BIn_$name")
    var kpreAbetaDeg 	= Symbol.CreateVariable(s"kpreAbetaDeg_$name")
    var kMgIn 			= Symbol.CreateVariable(s"kMgIn_$name")           // constitive Mg influx (TRPM7...)
    var kpreMgOut 		= Symbol.CreateVariable(s"kpreMgOut_$name")       // kMgIn/kMgOut = 0.5
    var kBDNFMg 		= Symbol.CreateVariable(s"kBDNFMg_$name")         // influence of BDNF to Mg influx
    
    //postsynaptic constants
    
    var postVol 		= Symbol.CreateVariable(s"postVol_$name")           // volume of postsynaptic spine
    var qAMPA 			= Symbol.CreateVariable(s"qAMPA_$name")             // AMPAR
    var kpostCaOut 		= Symbol.CreateVariable(s"kpostCaOut_$name")        // Ca extrusion from spine
    var CN50 			= Symbol.CreateVariable(s"CN50_$name")
    var CNSlope 		= Symbol.CreateVariable(s"CNSlope_$name")
    var TrkB50 			= Symbol.CreateVariable(s"TrkB50_$name")
    var TrkBSlope 		= Symbol.CreateVariable(s"TrkBSlope_$name")
    var a5 				= Symbol.CreateVariable(s"a5_$name")    // k+ of Ca buffer
    var b5 				= Symbol.CreateVariable(s"b5_$name")    // k- of Ca buffer
    var tPostCaBuffer   = Symbol.CreateVariable(s"tPostCaBuffer_$name")
    var a6 				= Symbol.CreateVariable(s"a6_$name")    // k+ of CN activation
    var b6 				= Symbol.CreateVariable(s"b6_$name")    // k- of CN
    var a7	 			= Symbol.CreateVariable(s"a7_$name")    // TrkB activation
    var b7 				= Symbol.CreateVariable(s"b7_$name")    // TrkB deactivation
    
    
    //presynaptic variables
    var preCa		:Symbol= Symbol.CreateVariable(s"preCa_$name")  // presynaptic [Ca]i
    var preCaBuff	:Symbol= Symbol.CreateVariable(s"preCaBuff_$name")// presynaptic [Mg]i
    var aPreCDK		:Symbol= Symbol.CreateVariable(s"aPreCDK_$name")
    var preSensor	:Symbol= Symbol.CreateVariable(s"preSensor_$name")      // presynaptic [sensor]
    var aPreTrkB	:Symbol= Symbol.CreateVariable(s"aPreTrkB_$name")
    var preNR2B		:Symbol= Symbol.CreateVariable(s"preNR2B_$name")
    var preAbeta	:Symbol= Symbol.CreateVariable(s"preAbeta_$name")
    var preMg		:Symbol= Symbol.CreateVariable(s"preMg_$name")
                    :Symbol
    var postCa		:Symbol= Symbol.CreateVariable(s"postCa_$name")
    var postCaBuff	:Symbol= Symbol.CreateVariable(s"postCaBuff_$name")
    var aPostCN		:Symbol= Symbol.CreateVariable(s"aPostCN_$name")
    var aPostTrkB	:Symbol= Symbol.CreateVariable(s"aPostTrkB_$name")
    var qNMDAR		:Symbol= Symbol.CreateVariable(s"qNMDAR_$name")  // qNMDAR
    
    // other variables for communication between compartments
    var EPSC        = Config.zero_s
   
    
    val onenda = NDArray.ones(Config.SHAPE,ctx)
    
    // parameters
    
    // common constants
    var Vs_nda       :NDArray = onenda * -50f                 // threshold for the activation of VGCC
    var gK_nda       :NDArray = onenda * 1;                     //
    var Vk_nda       :NDArray = onenda * -70;                   // reversal potential for K channel
    var Vr_nda       :NDArray = onenda * -70;                   // resting Vm
    var Ve_nda       :NDArray = onenda * 0;                     // reversal potential for AMPA channel
    var Vca_nda      :NDArray = onenda * 30;                   // reversal potential for Ca flux though NMDAR
    var Cm_nda       :NDArray = onenda * 10;                    // membran capacitance
    var Mgo_nda      :NDArray = onenda * 1.2f;                  // [Mg]o = 1.2 mM
    var kMgNMDAR_nda :NDArray = onenda * 4.5f;             // NMDAR Mg block
    var kNMDARIn_nda :NDArray = onenda * 0.00004f;          // constitutive insertion of NMDAR
    var kNMDARCa_nda :NDArray = onenda * -0.2f;            // converting Inmda to Ca flux
    var CaBasal_nda  :NDArray = onenda * 0.001f;             // basal Ca influx
    // magnesium
    var Mg50_nda    :NDArray = onenda * 400f;                                             // 50//
    var MgSlope_nda :NDArray = onenda * 50f;
    
    // sensor
    var aCDK50_nda :NDArray = onenda *  0.5f;
    var aCDKSlope_nda :NDArray = onenda * 0.1f;
        
    //presynaptic constants
    var preVol_nda 			:NDArray = onenda                 // volume of presynaptic terminals
    var a1_nda 				:NDArray = onenda * 0.02f;          // k+ of Ca buffer
    var b1_nda 				:NDArray = onenda * 0.0001f;        // k- of Ca buffer
    var tPreCaBuffer_nda 	:NDArray = onenda * 5f;       // presynaptic Ca buffer
    var kSensorDeg_nda 		:NDArray = onenda * 0.0001f;         // rate of sensor degradation
    var a2_nda 				:NDArray = onenda * 0.01f          // k+ of CDK
    var b2_nda 				:NDArray = onenda * 0.0001f;        // k- of CDK
    var aCDKNR2B50_nda 		:NDArray = onenda * 0.7f;
    var aCDKNR2BSlope_nda 	:NDArray = onenda * 0.3f;
    var a4_nda 				:NDArray = onenda * 0.00025f;        // k+ of pre TrkB
    var b4_nda 				:NDArray = onenda * 0.0002f;        // k- of pre TrkB
    var aPreTrkB50_nda 		:NDArray = onenda * 0.4f;
    var aPreTrkBSlope_nda 	:NDArray = onenda * 0.1f;
    var kpreNR2BIn_nda 		:NDArray = onenda * 0.000025f;
    var kpreAbetaDeg_nda 	:NDArray = onenda * 0.0001f;
    var kMgIn_nda 			:NDArray = onenda * 0.004f;              // constitive Mg influx (TRPM7...)
    var kpreMgOut_nda 		:NDArray = onenda * 0.00002f;        // kMgIn/kMgOut = 0.5
    var kBDNFMg_nda 		:NDArray = onenda * 0.04f;             // influence of BDNF to Mg influx
    
    //postsynaptic constants
    
    var postVol_nda 		:NDArray = onenda * 1f;                // volume of postsynaptic spine
    var qAMPA_nda 			:NDArray = onenda * 0.2f;                // AMPAR
    var kpostCaOut_nda 		:NDArray = onenda * 0.1f;           // Ca extrusion from spine
    var CN50_nda 			:NDArray = onenda * 0.55f;
    var CNSlope_nda 		:NDArray = onenda * 0.1f;
    var TrkB50_nda 			:NDArray = onenda * 0.4f;
    var TrkBSlope_nda 		:NDArray = onenda * 0.1f;
    var a5_nda 				:NDArray = onenda * 0.005f;     // k+ of Ca buffer
    var b5_nda 				:NDArray = onenda * 0.001f;     // k- of Ca buffer
    var tPostCaBuffer_nda   :NDArray = onenda * 1f;
    var a6_nda 				:NDArray = onenda * 0.005f;     // k+ of CN activation
    var b6_nda 				:NDArray = onenda * 0.001f;     // k- of CN
    var a7_nda	 			:NDArray = onenda * 0.0003f;    // TrkB activation
    var b7_nda 				:NDArray = onenda * 0.0002f;    // TrkB deactivation
    
    
    //presynaptic variables
    var preCa_nda		:NDArray = onenda * 0;     // presynaptic [Ca]i
    var preCaBuff_nda	:NDArray = onenda * 5;   // presynaptic [Mg]i
    var aPreCDK_nda		:NDArray = onenda * 0;
    var preSensor_nda	:NDArray = onenda * 0.7f;      // presynaptic [sensor]
    var aPreTrkB_nda	:NDArray = onenda * 0;
    var preNR2B_nda		:NDArray = onenda * 1;
    var preAbeta_nda	:NDArray = onenda * 1;
    var preMg_nda		:NDArray = onenda * 400;
    
    //postsynaptic variables
    var postCa_nda		:NDArray = onenda * 0f;
    var postCaBuff_nda	:NDArray = onenda * 1f;
    var aPostCN_nda		:NDArray = onenda * 0f;
    var aPostTrkB_nda	:NDArray = onenda * 0f;
    var qNMDAR_nda		:NDArray = onenda * 1f;  // qNMDAR
    
    // other variables for communication between compartments
    var EPSC_nda :NDArray = onenda * 0f;
    

    //initial y
    var y_preCa_nda		:NDArray = onenda * 0;     // presynaptic [Ca]i
    var y_preCaBuff_nda	:NDArray = onenda * 5;   // presynaptic [Mg]i
    var y_aPreCDK_nda		:NDArray = onenda * 0;
    var y_preSensor_nda	:NDArray = onenda * 0.7f;      // presynaptic [sensor]
    var y_aPreTrkB_nda	:NDArray = onenda * 0;
    var y_preNR2B_nda		:NDArray = onenda * 1;
    var y_preAbeta_nda	:NDArray = onenda * 1;
    var y_preMg_nda		:NDArray = onenda * 400;
    
    //postsynaptic variables
    var y_postCa_nda		:NDArray = onenda * 0f;
    var y_postCaBuff_nda	:NDArray = onenda * 1f;
    var y_aPostCN_nda		:NDArray = onenda * 0f;
    var y_aPostTrkB_nda	:NDArray = onenda * 0f;
    var y_qNMDAR_nda		:NDArray = onenda * 1f;  // qNMDAR
    
    
    
    override def getSymbolMap():Map[String,NDArray] = {
    	Map (s"Vs_$name"      -> Vs_nda     ,
    		s"gK_$name"       -> gK_nda     , 
    		s"Vk_$name"       -> Vk_nda     , 
            s"Vr_$name"       -> Vr_nda     , 
            s"Ve_$name"       -> Ve_nda     , 
            s"Vca_$name"      -> Vca_nda    , 
            s"Cm_$name"       -> Cm_nda     , 
            s"Mgo_$name"     -> Mgo_nda    , 
            s"kMgNMDAR_$name" -> kMgNMDAR_nda,
            s"kNMDARIn_$name" -> kNMDARIn_nda,
            s"kNMDARCa_$name" -> kNMDARCa_nda,
            s"CaBasal_$name"  -> CaBasal_nda ,
            s"Mg50_$name"     -> Mg50_nda   ,
            s"MgSlope_$name"  -> MgSlope_nda ,
          
            s"aCDK50_$name"          ->       aCDK50_nda         ,         
    	    s"aCDKSlope_$name"       ->        aCDKSlope_nda     ,

    	    s"preVol_$name"	        ->	     preVol_nda 			,
    	    s"a1_$name"             ->        a1_nda 				,
    	    s"b1_$name"             ->        b1_nda 				,
    	    s"tPreCaBuffer_$name"   ->        tPreCaBuffer_nda 	 ,
    	    s"kSensorDeg_$name"     ->        kSensorDeg_nda 		,
    	    s"a2_$name"             ->        a2_nda 				,
    	    s"b2_$name"             ->        b2_nda 				,
    	    s"aCDKNR2B50_$name"     ->        aCDKNR2B50_nda 		,
    	    s"aCDKNR2BSlope_$name"   ->        aCDKNR2BSlope_nda ,    
    	    s"a4_$name"	             ->        a4_nda 				,
    	    s"b4_$name"             ->        b4_nda 				,
    	    s"aPreTrkB50_$name"     ->        aPreTrkB50_nda 		,
    	    s"aPreTrkBSlope_$name"   ->        aPreTrkBSlope_nda ,    
            s"kpreNR2BIn_$name"     ->        kpreNR2BIn_nda 		,
            s"kpreAbetaDeg_$name"   ->        kpreAbetaDeg_nda 	 ,
            s"kMgIn_$name"          ->        kMgIn_nda 		,    
            s"kpreMgOut_$name"      ->        kpreMgOut_nda 	,    
            s"kBDNFMg_$name"        ->        kBDNFMg_nda 		 ,

            s"postVol_$name"        ->        postVol_nda 		 ,
            s"qAMPA_$name"          ->        qAMPA_nda 		,    
            s"kpostCaOut_$name"     ->        kpostCaOut_nda 		,
            s"CN50_$name"           ->        CN50_nda 			 ,
            s"CNSlope_$name"        ->        CNSlope_nda 		 ,
            s"TrkB50_$name"         ->        TrkB50_nda 			,
            s"TrkBSlope_$name"      ->        TrkBSlope_nda 	,    
            s"a5_$name"		        ->        a5_nda 				,
            s"b5_$name"	    	    ->        b5_nda 				,
            s"tPostCaBuffer_$name"   ->        tPostCaBuffer_nda ,  
            s"a6_$name"			    ->        a6_nda 				,
            s"b6_$name"    		    ->        b6_nda 				,
            s"a7_$name"	    	   ->        a7_nda	 			 ,
            s"b7_$name" 	   	   ->        b7_nda 		,		
                                             
            s"preCa_$name"          ->      y_preCa_nda        ,		
            s"preCaBuff_$name"      ->      y_preCaBuff_nda	 ,
            s"aPreCDK_$name"        ->      y_aPreCDK_nda		 ,
            s"preSensor_$name"      ->      y_preSensor_nda	 ,
            s"aPreTrkB_$name"       ->      y_aPreTrkB_nda	 ,
            s"preNR2B_$name"        ->      y_preNR2B_nda		 ,
            s"preAbeta_$name"       ->      y_preAbeta_nda	 ,
            s"preMg_$name"          ->      y_preMg_nda        ,	

            s"postCa_$name"   	    ->        y_postCa_nda		,
    	    s"postCaBuff_$name"       ->        y_postCaBuff_nda	,
            s"aPostCN_$name"          ->        y_aPostCN_nda		,
            s"aPostTrkB_$name"        ->        y_aPostTrkB_nda	 ,
            s"qNMDAR_$name"           ->        y_qNMDAR_nda		
    	)
    }
             
    override def getInitialVar():Array[String] = {
  		Array(s"y${this.variableindices(0)}" ,
  			  s"y${this.variableindices(1)}" ,
  		      s"y${this.variableindices(2)}" ,
  		      s"y${this.variableindices(3)}" ,
  		      s"y${this.variableindices(4)}" ,
  		      s"y${this.variableindices(5)}" ,
  		      s"y${this.variableindices(6)}" ,
  		      s"y${this.variableindices(7)}" ,
  		      s"y${this.variableindices(8)}" ,
  		      s"y${this.variableindices(9)}" ,
  		      s"y${this.variableindices(10)}" ,
  		      s"y${this.variableindices(11)}" ,
  		      s"y${this.variableindices(12)}")
  	 }
    
    override def getInitial(map : Map[String,NDArray]=null): Map[String,NDArray] = {
			 if(map==null)
				
			 	Map(s"y${this.variableindices(0)}"->this.preCa_nda,
		 			s"y${this.variableindices(1)}"->this.preCaBuff_nda,
		 			s"y${this.variableindices(2)}"->this.aPreCDK_nda,
		 			s"y${this.variableindices(3)}"->this.preSensor_nda,
		 			s"y${this.variableindices(4)}"->this.aPreTrkB_nda,
		 			s"y${this.variableindices(5)}"->this.preNR2B_nda,
		 			s"y${this.variableindices(6)}"->this.preAbeta_nda,
		 			s"y${this.variableindices(7)}"->this.preMg_nda,
		 			s"y${this.variableindices(8)}"->this.postCa_nda,
		 			s"y${this.variableindices(9)}"->this.postCaBuff_nda,
		 			s"y${this.variableindices(10)}"->this.aPostCN_nda,
		 			s"y${this.variableindices(11)}"->this.aPostTrkB_nda,
		 			s"y${this.variableindices(12)}"->this.qNMDAR_nda)
    	 else {
    	 	  map
    	 	}
  	}
    
    override def getInitialY():Array[NDArray] = {
    	  Array(y_preCa_nda,
    			y_preCaBuff_nda,
    			y_aPreCDK_nda,
    			y_preSensor_nda,
    			y_aPreTrkB_nda,
    			y_preNR2B_nda,
    			y_preAbeta_nda,
    			y_preMg_nda,
    	        y_postCa_nda,
    	        y_postCaBuff_nda,
    	        y_aPostCN_nda,
    	        y_aPostTrkB_nda,
    	        y_qNMDAR_nda)
    }
    
    
    override def update(t_onehot: Symbol, y:Array[Symbol],yDot:Array[ Symbol],indices:Array[Int]):Array[Symbol] = {
            
    	this.preCa = y(indices(0));
    	this.preCaBuff = y(indices(1));
    	this.aPreCDK = y(indices(2));
    	this.preSensor = y(indices(3));
    	this.aPreTrkB = y(indices(4));
    	this.preNR2B = y(indices(5));
    	this.preAbeta = y(indices(6));
    	this.preMg = y(indices(7));
    	this.postCa = y(indices(8));
    	this.postCaBuff = y(indices(9));
    	this.aPostCN = y(indices(10));
    	this.aPostTrkB = y(indices(11));
    	this.qNMDAR = y(indices(12));
    	
        //-------------------------Presyanptic dynamics-----------------------------
      
        var preVm=this.axon.preVm;
        
        val Ivgcc  = Symbol.Activation("relu")(Map("data"->(preVm-this.Vs),"act_type"->"relu"))*0.05
        
        var IpreNR2B =  Config.zero_s
        var IAChR =  Config.zero_s
        var preCaIn = IpreNR2B*this.kNMDARCa+Ivgcc+IAChR;
        var fCaBuff = this.tPreCaBuffer-this.preCaBuff;                     // Ca buffer
        var kpreCaOut = Config.one_s *(0.1f) / (Symbol.exp((this.Mg50-this.preMg)/this.MgSlope)+1);        // Ca efflux is funciton of [Mg]i (Boltzmann sigmoid function)
        var d_preCa = (preCaIn+this.b1 * this.preCaBuff - (kpreCaOut+this.a1*fCaBuff)*this.preCa)/this.preVol;           // dx/dt = (Jin-Jout)/vol
        var d_preCaBuff = this.a1*fCaBuff*this.preCa-this.b1*this.preCaBuff;  // presynaptic Ca buffer
        
       //CDK
        var d_aPreCDK = this.a2*(this.aPreCDK*(-1)+1)*this.preCa-this.b2*this.aPreCDK;  // presynaptic CDK activation depends on Ca level
        
       // sensor: insertion of sensor inhibited by Ca depend activation of CDK5
        var freeSensor = this.axon.freeSensor;
       	// freeSensor needs to be shared across synapses
        var kSensorIn =(Config.one_s * 0.0001f)/(Symbol.exp((this.aPreCDK-this.aCDK50)/this.aCDKSlope)+1);
        var d_preSensor = (kSensorIn*freeSensor-this.kSensorDeg*this.preSensor)/this.preVol;
        
       //BDNF=this.matrix.BDNF
       // right now BDNF retrograde signalling is synapse specific
        this.tmp = this.qNMDAR * Config.one_s
        this.BDNF = (this.dendrite.postVm-this.Vr)*this.tmp;                           // retrograde signalling following coinsident detection
        
       	// TrkB activation depends on BDNF concentration
        var d_aPreTrkB = this.a4*(Config.one_s+this.aPreTrkB*(-1))*BDNF-this.b4*this.aPreTrkB;
         
	    // degradation of presynaptic NR2B by Calpain/CDK5 dependent process
        var kpreNR2BDeg = (Config.one_s * 0.0002f)/(Symbol.exp((this.aCDKNR2B50-this.aPreCDK)/this.aCDKNR2BSlope)+1);
        var d_preNR2B = (this.kpreNR2BIn - kpreNR2BDeg*this.preNR2B)/this.preVol;
        
    	// synthesis of Abeta is inhibited by BDNF
        var kpreAbetaIn =(Config.one_s * 0.0001f)/(Symbol.exp((this.aPreTrkB-this.aPreTrkB50)/this.aPreTrkBSlope)+1);
        var d_preAbeta = (kpreAbetaIn - this.kpreAbetaDeg*this.preAbeta)/this.preVol;
        
       // Mg
        var preMgIn = this.kMgIn+this.kBDNFMg*BDNF;                         // presyanptic Mg influx = constitive + regrade signalling
        var d_preMg = (preMgIn-this.kpreMgOut*this.preMg)/this.preVol;
        
		    // 	------------------postsynaptic--------------------------
        var Pr = this.preCa*this.preSensor;     // Pr update probability of release
        var postVm = this.dendrite.postVm;
        var Iampa = Pr*this.qAMPA*(postVm-this.Ve);                        // EPSCampa
        var pMgBlock = Config.one_s /((this.Mgo/this.kMgNMDAR)*Symbol.exp(postVm*(-2f/25.4f))+1);  // NMDAR Mg block
        var Inmda = Pr*this.qNMDAR*(postVm - this.Vca)*pMgBlock;         // EPSCnmda
        this.EPSC = Iampa+Inmda;
        
       	//Calcium
        var postCaIn = Inmda*this.kNMDARCa+this.CaBasal;                    // total postsynaptic Ca influx = NMDAR + VGCC
        var fpostCaBuff = this.tPostCaBuffer-this.postCaBuff;

        var d_postCa = (postCaIn+this.b5*this.postCaBuff - (this.kpostCaOut+this.a5*fpostCaBuff)*this.postCa)/this.postVol;
        var d_postCaBuff = this.a5*fpostCaBuff*this.postCa-this.b5*this.postCaBuff;  // postsynaptic Ca buffer
		    
		    // degradation of NMDAR is promoted by CN Calcineurin, which is activated by [Ca], and
		    // protected by BDNF via activation of Src Kinase.
        var pCN = Config.one_s /(Symbol.exp((this.CN50-this.aPostCN)/this.CNSlope)+1);
        var d_aPostCN = this.a6*(this.aPostCN*(-1)+1)*this.postCa-this.b6*this.aPostCN;  // postsynaptic CN activation
        var pTrkB = Config.one_s/(Symbol.exp((this.aPostTrkB-this.TrkB50)/this.TrkBSlope)+1);
        var d_aPostTrkB = this.a7*(this.aPostTrkB*(-1)+Config.one_s)*BDNF-this.b7*this.aPostTrkB;
        var kNMDARdeg = pCN*pTrkB*0.005f; // BDNF/CN
        var d_qNMDAR = this.kNMDARIn - kNMDARdeg*this.qNMDAR;

        
        yDot(indices(0))  = d_preCa;
        yDot(indices(1))  = d_preCaBuff;
        yDot(indices(2))  = d_aPreCDK;
        yDot(indices(3))  = d_preSensor;
        yDot(indices(4))  = d_aPreTrkB;
        yDot(indices(5))  = d_preNR2B;
        yDot(indices(6))  = d_preAbeta;
        yDot(indices(7))  = d_preMg;
        yDot(indices(8))  = d_postCa;
        yDot(indices(9))  = d_postCaBuff;
        yDot(indices(10)) = d_aPostCN;
        yDot(indices(11)) = d_aPostTrkB;
        yDot(indices(12)) = d_qNMDAR
 
        yDot
        
    }        
        
    
    //connectivity
    def getaxon():Axon = {
        this.axon;
    }
    
    def getdendrite():Dendrite = {
        this.dendrite;
    }
    
    
}