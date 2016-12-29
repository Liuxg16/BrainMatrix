package thu.brainmatrix.ml

import scala.util.control.Breaks
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.Random
import thu.brainmatrix.util.mathTool

/**
 * 
 * 
 * properties
 * pi:
 * T: the transfer probabilities matrix (K,K)
 * Obs_pi: the probabilities of the observations,(K,D)
 * this model has K hidden different states and D observed states
 * 
 */
class HMM(val pi:NDArray,val T:NDArray,val Obs_pi:NDArray) {
	def simulation(nSteps:Int):(Array[Int],Array[Int]) =  {
		
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
	
	def train(observations:Array[Int]):Array[NDArray] = {
		val ctx = Context.cpu(0)
		val criterion = 0.5
		
		val obs_T = NDArray.transpose(this.Obs_pi)
		
		var pi_est = NDArray.Normalize(NDArray.ones(this.pi.shape,ctx)) 
		var T_est  = NDArray.Normalize(NDArray.ones(this.T.shape, ctx))
//		var obs_pi_est_T  = NDArray.transpose(NDArray.array(Array(0.3f,0.3f,0.4f,0.2f,0.5f,0.3f,0.3f,0.3f,0.4f),this.Obs_pi.shape,ctx))
		var obs_pi_est_T  = NDArray.Normalize(NDArray.transpose(Random.uniform(0, 1, this.Obs_pi.shape,ctx)))
		
		val nsamples = observations.length
		val nstates  = this.pi.size 
		val nhiddenstates = this.Obs_pi.shape(1)
		
		var iter = 0
		var done:Boolean = false
		while(!done){
			val alpha = NDArray.zeros(Shape(nsamples,nstates),ctx)
			val alpha_theta = NDArray.zeros(Shape(nsamples,nstates),ctx) // model probability
			val alpha_real = NDArray.zeros(Shape(nsamples,nstates),ctx)  // estimated probability
			
			
			
			val c = Array.fill[Float](nsamples)(0f)
			// calculate  alpha_0
			val alpha_0  = pi_est *  obs_pi_est_T.slice(observations(0))
			c(0) = 1f/NDArray.sum(alpha_0).toScalar
			// and normalize
			(alpha_0*c(0)).copyTo(alpha.slice(0))

			(this.pi * obs_T.slice(observations(0))).copyTo(alpha_theta.slice(0))
			alpha_0.copyTo(alpha_real.slice(0))
			
//			println(this.pi * obs_T.slice(observations(0)))
//			println(alpha_theta.slice(0))
			
			
			for(t <- 1 until nsamples){
				// \alpha_{t}(i) = P(x_1\cdots,x_t,y_t=i|\theta) = \Sigma_j \{\alpha_{t-1}(j)t_{j,i}\} e_{i,x_t}
				val alpha_t = NDArray.dot(alpha.slice(t-1),T_est) * obs_pi_est_T.slice(observations(t))
				c(t) = 	1f/NDArray.sum(alpha_t).toScalar
				(alpha_t*c(t)).copyTo(alpha.slice(t))
				
				val alpha_theta_tmp = NDArray.dot(alpha_theta.slice(t-1),this.T) * obs_T.slice(observations(t))
				val max = 1f/(NDArray.max(alpha_theta_tmp).toScalar)
				(alpha_theta_tmp*max).copyTo(alpha_theta.slice(t))
				(NDArray.dot(alpha_real.slice(t-1),T_est) * obs_pi_est_T.slice(observations(t))*max).copyTo(alpha_real.slice(t))
//				println(alpha_theta.slice(t))
//				println(alpha_real.slice(t))
				alpha_theta_tmp.dispose()
				alpha_t.dispose()
			}
			
			// beta_t(i) = (x_{t+1},\cdots,x_T,y_{t+1}|\theta) = 
			val beta = NDArray.zeros(Shape(nsamples,nstates),ctx)
			(NDArray.ones(Shape(1,nstates),ctx)*c(nsamples-1)).copyTo(beta.slice(nsamples-1))
			
			// update beta backwards from end of sequence
			for(t<- (1 until nsamples).reverse ){
				
				val beta_t_minus = NDArray.dot(obs_pi_est_T.slice(observations(t))*beta.slice(t),NDArray.transpose(T_est))
				(beta_t_minus*c(t-1)).copyTo(beta.slice(t-1))
				beta_t_minus.dispose()
			}
			
			// \xi_t(i,j)
//			val xi = NDArray.zeros(Shape(nsamples,nstates,nstates),ctx)
			
			val xi = Array.fill[NDArray](nsamples)(NDArray.zeros(Shape(nstates,nstates),ctx))
			
			for(t<- (0 until nsamples-1)){
//				val denom = NDArray.dot(NDArray.dot(alpha.slice(t), T_est)*obs_pi_est_T.slice(observations(t+1)),NDArray.transpose(beta.slice(t+1))).toScalar
				val denom = (NDArray.sum(NDArray.dot(alpha.slice(t),T_est) * obs_pi_est_T.slice(observations(t+1)) *beta.slice(t+1))).toScalar
				
//				println(denom-denom1)
				for(i <- 0 until nstates){
					val numer =T_est.slice(i) * obs_pi_est_T.slice(observations(t+1)) *beta.slice(t+1) * alpha(t,i)
					(numer/denom).copyTo(xi(t).slice(i))
//					tmp += numer
					numer.dispose()
				}
				
//				xi(t) /= NDArray.sum(tmp).toScalar
			}			
			var gamma_arr = xi.map(xij => {
				(0 until nstates).map{i =>{
//					sum_gamma1(i) += NDArray.sum(xij.slice(i)).toScalar
					NDArray.sum(xij.slice(i)).toScalar
				}}
			}).flatten 
			
			var gamma = NDArray.array(gamma_arr, Shape(nsamples,nstates), ctx)
			
			//
			val newpi = gamma.slice(0)
			var gamma_t = NDArray.transpose(gamma)
			val newT  = xi.reduceRight(_+_)
			var sum_gamma = (0 until nstates).map(i => NDArray.sum(gamma_t.slice(i)).toScalar).toArray
//			println(sum_gamma)
			(0 until nstates).map(i => {
				newT.slice(i) /= sum_gamma(i)
			})
			
			val tmp1 = alpha.slice(nsamples-1)*beta.slice(nsamples-1)
			(tmp1/NDArray.sum(tmp1).toScalar).copyTo(gamma.slice(nsamples-1))
			
			//beta
			NDArray.transpose(gamma).copyTo(gamma_t)
			sum_gamma = (0 until nstates).map(i => NDArray.sum(gamma_t.slice(i)).toScalar).toArray
			val sum_gamma_nda = NDArray.array(sum_gamma, Shape(1,nstates), ctx)
			
			val newObs_pi_T = NDArray.zeros(obs_pi_est_T.shape,ctx)
			observations.indices.foreach(id =>{
				val obs = observations(id)
				newObs_pi_T.slice(obs) +=  gamma.slice(id)
			})
			
			(0 until nhiddenstates).map(id=>{
				newObs_pi_T.slice(id) /= sum_gamma_nda
			})
					
//			println(newpi)
//			println(newT)
//			println(newObs_pi_T)
//			println(alpha_real.slice(nsamples-1))
//			println(alpha_theta.slice(nsamples-1))
			
			
//			if(NDArray.norm(pi_est-newpi).toScalar<criterion && NDArray.norm(T_est-newT).toScalar<criterion && NDArray.norm(obs_pi_est_T-newObs_pi_T).toScalar<criterion)
			if(math.abs(NDArray.sum(alpha_theta.slice(nsamples-1)-alpha_real.slice(nsamples-1)).toScalar) < criterion || iter>100)
				done = !done	
			
				
				
			newObs_pi_T.copyTo(obs_pi_est_T)
			newpi.copyTo(pi_est)
			newT.copyTo(T_est)
			
			alpha_real.dispose()
			alpha_theta.dispose()
			alpha_0.dispose()
			alpha.dispose()
			beta.dispose()
			gamma.dispose()
			gamma_t.dispose()
			xi.foreach(_.dispose())
			
			iter += 1
			
		}
		Array(pi_est,T_est,obs_pi_est_T)
		
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

object HMM{
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
			val (y,x) = hmm.simulation(10000)
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
		
		val hmm = new HMM(pi,T,obs_pi)
		val (y,x) = hmm.simulation(10000)
//		x.foreach(println)
		
		val Array(pi1,t1,obspi1) = hmm.train(x)
		val y_est = hmm.viterbiAlgorithm(pi,T,NDArray.transpose(obs_pi),x)
		var error = 0f
		y zip y_est foreach{case(yi,yie) =>{
			error += math.abs(yi-yie)
		}}
		
		println(s"TASK 2 estimate Y, error:${error/y.length}")
		println("TASK 3, estimate model:")
//		println(s"pi:$pi1")
	println(s"T:${NDArray.norm(t1-T)}")
		println(s"obspi:${NDArray.norm(obs_pi-obspi1)}")
		
		
	}
		
}






