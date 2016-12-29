package thu.brainmatrix.synapse
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape

/**
 * starttime,endtime,dt,rate,time_last: num_inputs *1
 *
 */

class Input(var spikeNum:Int)(ctx:Context) {
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
//	 val deltaT = (endtime-starttime)/spikeNum
	 var current:NDArray = null
	 
	 
	 def initial(rate:Int){
		 
		 this.current = NDArray.zeros(Shape(spikeNum,num_inputs), ctx_cpu)
		 var spikeingI = - NDArray.ones(Config.SHAPE, ctx_cpu) * 30f
//		 val dt = (endtime-starttime)/spikeNum 
		 for(i<- 10 until (spikeNum-20) by Math.round(1000/(rate)).toInt){
			 for(j<- 0 until 15){
				 spikeingI.copyTo(this.current.slice(i+j))
//		 		NDArray.setColumnSlice(this.current,, i+j)			
			 }
			}
		 
//		 val arrNda = NDArray.array(arr, Shape(spikeNum,1), ctx)
		 
	//	 (0 until this.num_inputs).foreach(i => NDArray.setColumnSlice(this.current, arrNda, i))
	 }
	 
	 def getinput(t:NDArray):NDArray = {
//		 var ttemp = NDArray.zeros(Shape(num_inputs,1), ctx_cpu)
		 
//		 	 println("lemonman-input")
//		 t.waitToRead()
//		 t.copyTo(ttemp)
		 val len = t.shape(1)
		 
	
//		 println(ttemp.shape)
//		 println(ttemp)
//		 
		 
		 val tt = (0 until len).map{i =>
			 this.current(t(0,i).toInt,i)
		 }.toArray
		 
//		 println("lemonman-input")
		 
		 NDArray.array(tt, Config.SHAPE, ctx)
	 }

}