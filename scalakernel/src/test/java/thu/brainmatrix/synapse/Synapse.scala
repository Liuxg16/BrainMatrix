package thu.brainmatrix.synapse


import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

class Synapse(val ctx: Context = Context.defaultCtx) extends Module{
  	// a basic synapse
	override var variable_table = Array[String]("preCa","preCaBuff","aPreCDK","preSensor","aPreTrkB","preNR2B","preAbeta","preMg",
    		"postCa","postCaBuff","aPostCN","aPostTrkB","qNMDAR")
	override var variableindices = Array.fill[Int](this.variable_table.length)(-1)
	
	
    // connectivity
    var axon:Axon = null
    
    var dendrite:Dendrite = null
    
    val onenda = NDArray.ones(Config.SHAPE,ctx)
    
    // parameters
    
    // common constants
    var Vs       :NDArray = onenda * -50f                 // threshold for the activation of VGCC
    var gK       :NDArray = onenda * 1;                     //
    var Vk       :NDArray = onenda * -70;                   // reversal potential for K channel
    var Vr       :NDArray = onenda * -70;                   // resting Vm
    var Ve       :NDArray = onenda * 0;                     // reversal potential for AMPA channel
    var Vca      :NDArray = onenda * 30;                   // reversal potential for Ca flux though NMDAR
    var Cm       :NDArray = onenda * 10;                    // membran capacitance
    var Mgo      :NDArray = onenda * 1.2f;                  // [Mg]o = 1.2 mM
    var kMgNMDAR :NDArray = onenda * 4.5f;             // NMDAR Mg block
    var kNMDARIn :NDArray = onenda * 0.00004f;          // constitutive insertion of NMDAR
    var kNMDARCa :NDArray = onenda * -0.2f;            // converting Inmda to Ca flux
    var CaBasal  :NDArray = onenda * 0.001f;             // basal Ca influx
    // magnesium
    var Mg50    :NDArray = onenda * 400f;                                             // 50//
    var MgSlope :NDArray = onenda * 50f;
    
    // sensor
    var aCDK50 :NDArray = onenda *  0.5f;
    var aCDKSlope :NDArray = onenda * 0.1f;
        
    //presynaptic constants
    var preVol 			:NDArray = onenda                 // volume of presynaptic terminals
    var a1 				:NDArray = onenda * 0.02f;          // k+ of Ca buffer
    var b1 				:NDArray = onenda * 0.0001f;        // k- of Ca buffer
    var tPreCaBuffer 	:NDArray = onenda * 5f;       // presynaptic Ca buffer
    var kSensorDeg 		:NDArray = onenda * 0.0001f;         // rate of sensor degradation
    var a2 				:NDArray = onenda * 0.01f          // k+ of CDK
    var b2 				:NDArray = onenda * 0.0001f;        // k- of CDK
    var aCDKNR2B50 		:NDArray = onenda * 0.7f;
    var aCDKNR2BSlope 	:NDArray = onenda * 0.3f;
    var a4 				:NDArray = onenda * 0.00025f;        // k+ of pre TrkB
    var b4 				:NDArray = onenda * 0.0002f;        // k- of pre TrkB
    var aPreTrkB50 		:NDArray = onenda * 0.4f;
    var aPreTrkBSlope 	:NDArray = onenda * 0.1f;
    var kpreNR2BIn 		:NDArray = onenda * 0.000025f;
    var kpreAbetaDeg 	:NDArray = onenda * 0.0001f;
    var kMgIn 			:NDArray = onenda * 0.004f;              // constitive Mg influx (TRPM7...)
    var kpreMgOut 		:NDArray = onenda * 0.00002f;        // kMgIn/kMgOut = 0.5
    var kBDNFMg 		:NDArray = onenda * 0.04f;             // influence of BDNF to Mg influx
    
    //postsynaptic constants
    
    var postVol 		:NDArray = onenda * 1f;                // volume of postsynaptic spine
    var qAMPA 			:NDArray = onenda * 0.2f;                // AMPAR
    var kpostCaOut 		:NDArray = onenda * 0.1f;           // Ca extrusion from spine
    var CN50 			:NDArray = onenda * 0.55f;
    var CNSlope 		:NDArray = onenda * 0.1f;
    var TrkB50 			:NDArray = onenda * 0.4f;
    var TrkBSlope 		:NDArray = onenda * 0.1f;
    var a5 				:NDArray = onenda * 0.005f;     // k+ of Ca buffer
    var b5 				:NDArray = onenda * 0.001f;     // k- of Ca buffer
    var tPostCaBuffer   :NDArray = onenda * 1f;
    var a6 				:NDArray = onenda * 0.005f;     // k+ of CN activation
    var b6 				:NDArray = onenda * 0.001f;     // k- of CN
    var a7	 			:NDArray = onenda * 0.0003f;    // TrkB activation
    var b7 				:NDArray = onenda * 0.0002f;    // TrkB deactivation
    
    
    //presynaptic variables
    var preCa		:NDArray = onenda * 0;     // presynaptic [Ca]i
    var preCaBuff	:NDArray = onenda * 5;   // presynaptic [Mg]i
    var aPreCDK		:NDArray = onenda * 0;
    var preSensor	:NDArray = onenda * 0.7f;      // presynaptic [sensor]
    var aPreTrkB	:NDArray = onenda * 0;
    var preNR2B		:NDArray = onenda * 1;
    var preAbeta	:NDArray = onenda * 1;
    var preMg		:NDArray = onenda * 400;
    
    //postsynaptic variables
    var postCa		:NDArray = onenda * 0f;
    var postCaBuff	:NDArray = onenda * 1f;
    var aPostCN		:NDArray = onenda * 0f;
    var aPostTrkB	:NDArray = onenda * 0f;
    var qNMDAR		:NDArray = onenda * 1f;  // qNMDAR
    
    // other variables for communication between compartments
    var EPSC :NDArray = onenda * 0f;
    
    override def getInitial():Array[NDArray] = {
    	Array(preCa,preCaBuff,aPreCDK,preSensor,aPreTrkB,preNR2B,preAbeta,preMg,
    	           postCa,postCaBuff,aPostCN,aPostTrkB,qNMDAR)
    }
    
    
    //cache variable
//    var preVm		= NDArray.empty(shape,ctx)
//    //Calcium         NDArray.empty(shape,ctx)
    var Ivgcc_arr =  Array.fill[Float](Config.NUMBER)(0f)
    
//    var Ivgcc 		= NDArray.empty(shape,ctx)
//          pMgBlock 	= NDArray.empty(shape,ctx)
//    var IpreNR2B 	= NDArray.empty(shape,ctx)
//    var IAChR 		= NDArray.empty(shape,ctx)
//    var preCaIn 	= NDArray.empty(shape,ctx)
//    var fCaBuff 	= NDArray.empty(shape,ctx)
//    var kpreCaOut 	= NDArray.empty(shape,ctx)
//    var d_preCa 	= NDArray.empty(shape,ctx)
//    var d_preCaBuff = NDArray.empty(shape,ctx)
//    //CDK             NDArray.empty(shape,ctx)
//    var d_aPreCDK 	= NDArray.empty(shape,ctx)
//    // sensor: insert NDArray.empty(shape,ctx)
//    var freeSensor 	= NDArray.empty(shape,ctx)
//    // freeSensor nee NDArray.empty(shape,ctx)
//    var kSensorIn 	= NDArray.empty(shape,ctx)
//    var d_preSensor = NDArray.empty(shape,ctx)
//    //BDNF=this.matri NDArray.empty(shape,ctx)
//    // right now BDNF NDArray.empty(shape,ctx)
//    var BDNF 		= NDArray.empty(shape,ctx)
//    var d_aPreTrkB 	= NDArray.empty(shape,ctx)
//    var kpreNR2BDeg = NDArray.empty(shape,ctx)
//    var d_preNR2B 	= NDArray.empty(shape,ctx)
//    var kpreAbetaIn = NDArray.empty(shape,ctx)
//    var d_preAbeta 	= NDArray.empty(shape,ctx)
//    var preMgIn 	= NDArray.empty(shape,ctx)
//    var d_preMg 	= NDArray.empty(shape,ctx)
//    var Pr 			= NDArray.empty(shape,ctx)
//    var postVm		= NDArray.empty(shape,ctx)
//    var Iampa 		= NDArray.empty(shape,ctx)
//    var pMgBlock 	= NDArray.empty(shape,ctx)
//    var Inmda 		= NDArray.empty(shape,ctx)
//    var postCaIn 	= NDArray.empty(shape,ctx)
//    var fpostCaBuff = NDArray.empty(shape,ctx)
//    var d_postCa 	= NDArray.empty(shape,ctx)
//    var d_postCaBuff= NDArray.empty(shape,ctx)
//    var pCN 		= NDArray.empty(shape,ctx)
//    var d_aPostCN 	= NDArray.empty(shape,ctx)
//    var pTrkB 		= NDArray.empty(shape,ctx)
//    var d_aPostTrkB = NDArray.empty(shape,ctx)
//    var kNMDARdeg 	= NDArray.empty(shape,ctx)
//    var d_qNMDAR 	= NDArray.empty(shape,ctx)
    
    
    
    //bak
    val preVmbak = NDArray.ones(Config.SHAPE,Context.defaultCtx)
    val vsbak = NDArray.ones(Config.SHAPE,ctx)
    
    override def update(t: NDArray,y:Array[NDArray],yDot:Array[NDArray],indices:Array[Int]):Array[NDArray] = {
            
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
      
    	this.axon.preVm.copyTo(this.preVmbak)
    	this.Vs.copyTo(this.vsbak)
//        var preVm=this.axon.preVm;
        
       //Calcium
        
        
        
//        for(i<-0 until Config.NUMBER){
//        	if(this.preVmbak(0,i) > this.vsbak(0,i)){
//        		Ivgcc_arr(i) = 0.05f * (this.preVmbak(0,i)-this.vsbak(0,i))                    // simply assume the size of Ivgcc linearly correlated with Vm after it pass V threshold	
//        	}
//        }
        var Ivgcc = NDArray.array(Ivgcc_arr, Config.SHAPE, ctx)
        
        
//        pMgBlock = 1/(1+(this.Mgo/this.kMgNMDAR)*Math.exp(-2*preVm/25.4));   // NMDAR Mg block
        var IpreNR2B =  NDArray.zeros(Config.SHAPE,ctx) //Pr1*preNR2B1*(preVm1 - Vca)*pMgBlock1;         // EPSCnmda
        var IAChR =  NDArray.zeros(Config.SHAPE,ctx);
        var preCaIn = IpreNR2B*this.kNMDARCa+Ivgcc+IAChR;
        var fCaBuff = this.tPreCaBuffer-this.preCaBuff;                     // Ca buffer
        var kpreCaOut = onenda *(0.1f) / (NDArray.exp((this.Mg50-this.preMg)/this.MgSlope)+1);        // Ca efflux is funciton of [Mg]i (Boltzmann sigmoid function)
        var d_preCa = (preCaIn+this.b1 * this.preCaBuff - (kpreCaOut+this.a1*fCaBuff)*this.preCa)/this.preVol;           // dx/dt = (Jin-Jout)/vol
        var d_preCaBuff = this.a1*fCaBuff*this.preCa-this.b1*this.preCaBuff;  // presynaptic Ca buffer
        
       //CDK
        var d_aPreCDK = this.a2*(-this.aPreCDK+1)*this.preCa-this.b2*this.aPreCDK;  // presynaptic CDK activation depends on Ca level
        
       // sensor: insertion of sensor inhibited by Ca depend activation of CDK5
        var freeSensor = this.axon.freeSensor;
       	// freeSensor needs to be shared across synapses
        var kSensorIn =(onenda * 0.0001f)/(NDArray.exp((this.aPreCDK-this.aCDK50)/this.aCDKSlope)+1);
        var d_preSensor = (kSensorIn*freeSensor-this.kSensorDeg*this.preSensor)/this.preVol;
        
       //BDNF=this.matrix.BDNF
       // right now BDNF retrograde signalling is synapse specific
        var BDNF = (this.dendrite.postVm-this.Vr)*this.qNMDAR;                           // retrograde signalling following coinsident detection
        
       	// TrkB activation depends on BDNF concentration
        var d_aPreTrkB = this.a4*(-this.aPreTrkB+1)*BDNF-this.b4*this.aPreTrkB;
         
	    // degradation of presynaptic NR2B by Calpain/CDK5 dependent process
        var kpreNR2BDeg = (onenda * 0.0002f)/(NDArray.exp((this.aCDKNR2B50-this.aPreCDK)/this.aCDKNR2BSlope)+1);
        var d_preNR2B = (this.kpreNR2BIn - kpreNR2BDeg*this.preNR2B)/this.preVol;
        
    	// synthesis of Abeta is inhibited by BDNF
        var kpreAbetaIn =(onenda * 0.0001f)/(NDArray.exp((this.aPreTrkB-this.aPreTrkB50)/this.aPreTrkBSlope)+1);
         var d_preAbeta = (kpreAbetaIn - this.kpreAbetaDeg*this.preAbeta)/this.preVol;
        
       // Mg
        var preMgIn = this.kMgIn+this.kBDNFMg*BDNF;                         // presyanptic Mg influx = constitive + regrade signalling
        var d_preMg = (preMgIn-this.kpreMgOut*this.preMg)/this.preVol;
        
		    // 	------------------postsynaptic--------------------------
        var Pr = this.preCa*this.preSensor;     // Pr update probability of release
        var postVm = this.dendrite.postVm;
        var Iampa = Pr*this.qAMPA*(postVm-this.Ve);                        // EPSCampa
        var pMgBlock = onenda /((this.Mgo/this.kMgNMDAR)*NDArray.exp(postVm*(-2f/25.4f))+1);  // NMDAR Mg block
        var Inmda = Pr*this.qNMDAR*(postVm - this.Vca)*pMgBlock;         // EPSCnmda
        this.EPSC = Iampa+Inmda;
        
       	//Calcium
        var postCaIn = Inmda*this.kNMDARCa+this.CaBasal;                    // total postsynaptic Ca influx = NMDAR + VGCC
        var fpostCaBuff = this.tPostCaBuffer-this.postCaBuff;

        var d_postCa = (postCaIn+this.b5*this.postCaBuff - (this.kpostCaOut+this.a5*fpostCaBuff)*this.postCa)/this.postVol;
        var d_postCaBuff = this.a5*fpostCaBuff*this.postCa-this.b5*this.postCaBuff;  // postsynaptic Ca buffer
		    
		    // degradation of NMDAR is promoted by CN Calcineurin, which is activated by [Ca], and
		    // protected by BDNF via activation of Src Kinase.
        var pCN = onenda /(NDArray.exp((this.CN50-this.aPostCN)/this.CNSlope)+1);
        var d_aPostCN = this.a6*(-this.aPostCN+1)*this.postCa-this.b6*this.aPostCN;  // postsynaptic CN activation
        var pTrkB = onenda/(NDArray.exp((this.aPostTrkB-this.TrkB50)/this.TrkBSlope)+1);
        var d_aPostTrkB = this.a7*(-this.aPostTrkB+1)*BDNF-this.b7*this.aPostTrkB;
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
 
        
        IpreNR2B 	dispose()
        IAChR 		dispose()
        preCaIn 	dispose()
        fCaBuff 	dispose()
        kpreCaOut 	dispose()
        kSensorIn 	dispose()
        BDNF 		dispose()
        kpreNR2BDeg dispose()
        kpreAbetaIn dispose()
        preMgIn 	dispose()
        Pr 			dispose()
        Iampa 		dispose()
        pMgBlock 	dispose()
        Inmda 		dispose()
        postCaIn 	dispose()
        fpostCaBuff dispose()
        pCN 		dispose()
        pTrkB 		dispose()
        kNMDARdeg 	dispose()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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