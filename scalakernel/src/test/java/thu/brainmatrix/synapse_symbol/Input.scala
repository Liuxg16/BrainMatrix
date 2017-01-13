package thu.brainmatrix.synapse_symbol
import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

/**
 * starttime,endtime,dt,rate,time_last: num_inputs *1
 *
 */

class Input(val name:String)(ctx:Context) {
     // parameters
	 // variable
	 /***
	  *  current:matrix(spikeNum,num_inputs)
	  *  input0
	  *  input1
	  *  ...
	  */
	val ctx_cpu = Context.cpu(0) 
	val num_inputs = Config.NUMBER
	 var current_nda:NDArray = null
	 
	 val current = Symbol.CreateVariable(s"current_$name")
	 
	 def initial(rate:Int){
//		 this.current_nda = NDArray.zeros(Shape(Config.SPIKENUM,num_inputs), ctx_cpu)
		 val current_tmp = NDArray.zeros(Shape(Config.SPIKENUM,num_inputs), ctx_cpu)
		 var spikeingI = NDArray.ones(Config.SHAPE, ctx_cpu) * -30f
		 for(i<- 10 until (Config.SPIKENUM-20) by Math.round(1000/(rate)).toInt){
			 for(j<- 0 until 15){
//				 for(k<- 0 until num_inputs){
//				 	this.current_nda(k,i+j) = -30f
//				 }
				 spikeingI.copyTo(current_tmp.slice(i+j))
			 }
			}
		 this.current_nda = NDArray.transpose(current_tmp.copyTo(ctx))
		 current_tmp.dispose()
	 }
	 
	
	// (NUMBER,SPIKENUM) => (number)
	 def getinput(t_onehot:Symbol):Symbol = {
		 val I = Symbol.Sum("sum")(Map("data"->t_onehot * this.current,"axis"->1))
//		 val I = Symbol.Dot(t_onehot * this.current,Config.spikes_ones_s , 1)
		 Symbol.Reshape("reshape")(Map("data"->I,"target_shape" -> s"(1,${Config.NUMBER})"))
	 }

}