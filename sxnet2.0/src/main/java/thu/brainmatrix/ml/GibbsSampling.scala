package thu.brainmatrix.ml

import scala.util.control.Breaks
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.Random
import thu.brainmatrix.util.mathTool

/**
 * 
 * HMM
 * properties
 * pi:
 * T: the transfer probabilities matrix (K,K)
 * Obs_pi: the probabilities of the observations,(K,D)
 * this model has K hidden different states and D observed states
 * 
 */
class GIbbsSampling(val pi:NDArray,val T:NDArray,val Obs_pi:NDArray) {
	
	val ctx = Context.cpu(0)
	var pi_est = NDArray.Normalize(NDArray.ones(pi.shape, ctx))
	var T_est = NDArray.Normalize(NDArray.ones(this.T.shape, ctx))
	var Obs_pi_est = NDArray.Normalize(NDArray.ones(this.Obs_pi.shape, ctx))
	
	def getObservation(nSteps:Int):(Array[Int],Array[Int]) =  {
		
		val observations = Array.fill[Int](nSteps)(0)
		val states = Array.fill[Int](nSteps)(0)
		val sampleStates = mathTool.SampleByPro1D(this.pi)
		val sampleObs    = mathTool.SampleByPro1D(this.Obs_pi.slice(states(0)))
		
		
		for(t<-1 until nSteps){
			states(t) = mathTool.SampleByPro1D(this.T.slice(states(t-1)))
			observations(t) = mathTool.SampleByPro1D(this.Obs_pi.slice(states(t-1)))
		}
		
		(states,observations)
	}
	
	
	
	
	def simulation(nSteps:Int,nRep:Int,x:Array[Int]) :Array[Int] =  {
		
//		val observations = Array.fill[Int](nSteps)(0)
		val states = Array.fill[Int](nSteps)(0)
		
		val T_T = NDArray.transpose(this.T_est)
		val obs_pi_T = NDArray.transpose(this.Obs_pi_est)
		
		// P(y_t|Y,X,\theta) = t_{y_t,y_{t+1}}e_{x_t,e_t}t_{y_{t-1},y_{t}}
		
		for(t<- 0 until nSteps){
			states(t) = mathTool.SampleByPro1D(this.pi_est)
//			observations(t) = mathTool.SampleByPro1D(this.Obs_pi.slice(states(t)))
		}
		
		for(iter <- 0 until nRep){
			for(t<-1 until nSteps-1){
				val pyt = NDArray.Normalize(T_T.slice(states(t+1)) * obs_pi_T.slice(x(t))*this.T_est.slice(states(t-1)))
				states(t) = mathTool.SampleByPro1D(pyt)
//				observations(t) = mathTool.SampleByPro1D(this.Obs_pi.slice(states(t)))
			}
		}
		
		states
	}
	
	def update(states:Array[Int],observations:Array[Int]) :Array[NDArray] = {
		val ctx = Context.cpu(0)
		val criterion = 0.5
		val I = this.Obs_pi.shape(0) //states n
		val K = this.Obs_pi.shape(1) // observations n
		
		
		val e_ik = Array.fill[Array[Float]](I)(Array.fill[Float](K)(0f))
		val Iy   = Array.fill[Float](I)(0.0000001f)
		val t_ij = Array.fill[Array[Float]](I)(Array.fill[Float](I)(0))
 		
		
		states zip observations foreach{case (s,o)=>{
			e_ik(s)(o) += 1
			Iy(s)      += 1
			
		}}
		
		e_ik.indices.foreach(id => {
			e_ik(id).indices.foreach { idx => e_ik(id)(idx) /= Iy(id)  }
		})
		
		states.indices.take(states.length-1).foreach(i => {
			t_ij(states(i))(states(i+1)) += 1f/Iy(states(i))
		})
		
		
		
		Array(NDArray.array(Iy.map(_/states.length),Shape(1,I),ctx),NDArray.array(t_ij.flatten,Shape(I,I),ctx),NDArray.array(e_ik.flatten, Shape(I,K), ctx))
		
	}
	
	
	def train(chainsNum:Int,x1:Array[Int]){
		
		var done = false
		var n = 0
		while(!done && n<1000){
			val y = simulation(chainsNum,3,x1)
			val Array(pi1,t1,obspi1) = update(y,x1)
			
			if(NDArray.norm(pi1-this.pi_est).toScalar<0.5 && NDArray.norm(t1-this.T_est).toScalar<0.5 && NDArray.norm(obspi1-this.Obs_pi).toScalar<0.5)
				done = !done
			
			
			println(obspi1)	
//			pi1.copyTo(this.pi_est)
			t1.copyTo(this.T_est)
			obspi1.copyTo(this.Obs_pi_est)
			n += 1
		}
		
		
	}
	
	
	def viterbiAlgorithm(pi_est:NDArray,T_est:NDArray,obs_pi_est_T:NDArray,x:Array[Int]):Array[Int] = {
		val ctx = Context.cpu(0)
		val nsamples = x.length 
		val nstates  = T_est.shape(0)
		val sobservations = obs_pi_est_T.shape(0)
		
		val delta  =  NDArray.zeros(Shape(nsamples,nstates), ctx)
		val phi  =  NDArray.zeros(Shape(nsamples,nstates), ctx)
		
		val T_est_T = NDArray.transpose(T_est)
		
		
		(pi_est*T_est.slice(x(0))).copyTo(delta.slice(0))
		
		delta.slice(0)
		
		for(t <-0 until nsamples-1){
			val nda = pi_est*obs_pi_est_T.slice(x(t))			
			for(i<- 0 until nstates){
				delta(t+1,i) += (NDArray.max(nda * T_est_T.slice(i))*obs_pi_est_T(x(t+1),i)).toScalar
			}
			val boardcast_nda = NDArray.concatenate(nda,nda,nda)
			(NDArray.argmaxChannel(boardcast_nda* T_est_T).reshape(Array(1,nstates))).copyTo(phi.slice(t+1))
		}
		
		val y = Array.fill[Int](nsamples)(0)
		y(nsamples-1) = NDArray.argmaxChannel(delta.slice(nsamples-1)).toScalar.toInt
		
		
		for(t <- (nsamples-2 to 0 by -1)){
			y(t) = NDArray.argmaxChannel(delta.slice(t)*T_est_T.slice(y(t+1))).toScalar.toInt
		}
		
		y
	}
	
	
}

object GIbbsSampling{
	def main(args:Array[String]){
//		test_homework(1000)
		test_homework1
	}
	
	def test{
		val ctx = Context.cpu(0)
		val num_states = 3 // A,B,C
		val num_obs    = 3
		val pi         = NDArray.Normalize((NDArray.array(Array(0.1f,0.4f,0.5f),Shape(1,num_states),ctx)))
		val obs_pi     = NDArray.array(Array(0.5f,0.3f,0.2f,0.1f,0.6f,0.3f,0.0f,0.3f,0.7f),Shape(num_states,num_obs),ctx)
		val T = NDArray.array(Array(0.7f,0.2f,0.1f,0.1f,0.6f,0.3f,0.4f,0.2f,0.4f),Shape(num_states,num_states),ctx)
		
		val hmm = new HMM(pi,T,obs_pi)
		val (y,x) = hmm.simulation(1000)

		x.foreach(println)
		val Array(pi1,t1,obspi1) = hmm.train(x)
			println(s"pi:$pi1")
		println(s"T:$t1")
		println(s"obspi:$obspi1")

	}
	
		def test1{
		val ctx = Context.cpu(0)
		val num_states = 2 // A,B,C
		val num_obs    = 3
		val pi         = NDArray.Normalize((NDArray.array(Array(0.5f,0.5f),Shape(1,num_states),ctx)))
		val obs_pi     = NDArray.array(Array(0.7f,0.2f,0.1f,0.1f,0.6f,0.3f),Shape(num_states,num_obs),ctx)
		val T = NDArray.array(Array(0.5f,0.5f,0.2f,0.8f),Shape(num_states,num_states),ctx)
		
		val hmm = new HMM(pi,T,obs_pi)
		val (y,x) = hmm.simulation(1000)
//		val x =  Array(2,  0,  0,  0,  0,  0, 0, 1,  0,  0)

//		x.foreach(println)
		hmm.train(x)
		val Array(pi1,t1,obspi1) = hmm.train(x)
		println(s"pi:$pi1")
		println(s"T:$t1")
		println(s"obspi:$obspi1")
	}

		
		def test_homework(num:Int){
		val ctx = Context.cpu(0)
		val num_states = 3 // A,B,C
		val num_obs    = 2
		val pi         = NDArray.Normalize((NDArray.array(Array(0.3f,0.3f,0.4f),Shape(1,num_states),ctx)))
		val obs_pi     = NDArray.array(Array(0.1f,0.9f,0.5f,0.5f,0.9f,0.1f),Shape(num_states,num_obs),ctx)
		val T = NDArray.array(Array(0.8f,0.2f,0f,0.1f,0.7f,0.2f,0.1f,0f,0.9f),Shape(num_states,num_states),ctx)
		
		val hmm = new HMM(pi,T,obs_pi)
		
		
		val Ts = NDArray.zeros(Shape(num,num_states,num_states), ctx)
		val obs_pis = NDArray.zeros(Shape(num,num_states,num_obs), ctx)
		
		for (i<- 0 until num){
			println(s"**************step $i****************")
			val (y,x) = hmm.simulation(1000)
			val res = hmm.train(x)
			println(s"T:${res(1)}")
			println(s"obs_pis:${res(2)}")
			
			res(1).reshape(Array(1,num_states,num_states)).copyTo(Ts.slice(i))
			res(2).reshape(Array(1,num_states,num_obs)).copyTo(obs_pis.slice(i))
			
		}
		
		println(s"T variance:"+NDArray.norm(Ts))
		println(s"obs_pis variance :"+NDArray.norm(obs_pis))
//		println(s"T:$t1")
//		println(s"obspi:$obspi1")
		
	}
		
		
	def test_homework1{
		val ctx = Context.cpu(0)
		val num_states = 3 // A,B,C
		val num_obs    = 2
		
		
		val pi         = NDArray.Normalize((NDArray.array(Array(0.3f,0.3f,0.4f),Shape(1,num_states),ctx)))
		val obs_pi     = NDArray.array(Array(0.1f,0.9f,0.5f,0.5f,0.9f,0.1f),Shape(num_states,num_obs),ctx)
		val T = NDArray.array(Array(0.8f,0.2f,0f,0.1f,0.7f,0.2f,0.1f,0f,0.9f),Shape(num_states,num_states),ctx)
		
		val chainsNum = 1000
		
		val gs = new GIbbsSampling(pi,T,obs_pi)
		val (y1,x1) = gs.getObservation(chainsNum)
		
		gs.train(chainsNum,x1)
		println(s"T:${NDArray.norm(gs.T_est-T)}")
		println(s"obspi:${NDArray.norm(gs.Obs_pi_est-obs_pi)}")
		
		
		val y = gs.simulation(chainsNum,3,x1)
//		y.foreach(println)
		val y_est = gs.viterbiAlgorithm(gs.pi_est,gs.T_est,NDArray.transpose(gs.Obs_pi_est),x1)
		var error = 0f
		y zip y_est foreach{case(yi,yie) =>{
			error += math.abs(yi-yie)
		}}
		
		println(s"TASK 2 estimate Y, error:${error/y.length}")
		
		
		
	}
		
}






