package thu.brainmatrix.Imperative
import scala.collection.mutable.ListBuffer
import thu.brainmatrix.Context
import thu.brainmatrix.NDArray
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.IO
import thu.brainmatrix.Context.ctx2Array
import thu.brainmatrix.Symbol
import thu.brainmatrix.FeedForward
import thu.brainmatrix.Shape
import thu.brainmatrix.Random
import thu.brainmatrix.DataBatch

object MLP {
	
	val batchSize  = 100
	val inputSize  = 784
	val hiddenSize = 40
	val classSize:Int = 10
	
  def mlp_entroy(implicit ctx:Context){
	val trainDataIter = IO.MNISTIter(scala.collection.immutable.Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 784)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0",
      "seed" -> "10"))
      
	val rates = Array(0.00000001f)//succeed!!
//	val rates = Array(0.00000001f,0.000001f,0.001f,0.04f,0.9f,3f,10f,50f,100f,1000f,10000f,100000f,1000000f)

	var  max = 0f
	
	rates.foreach(rate => {
		val mlp = new MLP(batchSize,inputSize,hiddenSize,classSize)
	    var n  = 0
	    var dataBatch = trainDataIter.next()  
		
	    for(k<-0 to 0){
			while(trainDataIter.hasNext && n<100){
				n += 1
	//			dataBatch = trainDataIter.next()  
				
				mlp.forward(dataBatch)
		        mlp.update(rate)
//		        
			    println(mlp.outputs(2))
//			    println(mlp.outputs(4))
//			    println(mlp.U_nda)
			    
			    
		//        if(n%10 == 0)
		        val error = mlp.error(dataBatch.label(0))
		        if(max<error)
		        	max = error
		        print(error+" ")
			}
	    }
			println(rate)
			mlp.dispose()
		
	})
	println(s"max:$max")
      
//      println(trainDataIter.getData()(0).shape)
	  
  }
	

	
	
	def main(args:Array[String]){
		implicit val ctx = Context.cpu(0)
		mlp_entroy
	}
	
}


class MLP(val batchSize:Int, val inputSize:Int,val hiddenSize:Int,val classSize:Int)(implicit ctx:Context){
	
	
	val eps  = 1e-8
	val data = Symbol.CreateVariable("data")
	val W = Symbol.CreateVariable("W")
	val U = Symbol.CreateVariable("U")
	val label = Symbol.CreateVariable("label")
	
	val h = Symbol.FullyConnected("h")(Map("data" -> data, "num_hidden" -> hiddenSize,"weight"->W,"no_bias"->true))
	val h_act1 = Symbol.Activation("h_act1")(Map("data" -> h, "act_type" -> "sigmoid"))  
	
	val z = Symbol.FullyConnected("z")(Map("data" -> h_act1, "num_hidden" -> classSize,"weight"->U,"no_bias"->true))
	val y = Symbol.SoftmaxActivation("y")(Map("data"->z))
	
	
	
	
	val d_z = y - label //(n,10)
	
//	val d_U = Symbol.FullyConnected("h")(Map("data" -> h, "num_hidden" -> hiddenSize,"weight"->(d_z),"no_bias"->true))
	
	val d_U = Symbol.Dot(Symbol.transpose(d_z),h_act1,hiddenSize)  //(10,n),(n,hn)=>(10,hiddenSize)
	val d_h_act1 = Symbol.Dot(d_z,U,hiddenSize)  //(n,10),(10,hn)=>(num,hn)
	val d_h = d_h_act1 * h_act1* (h_act1-1)*(-1) 
	val d_W = Symbol.Dot(Symbol.transpose(d_h),data,inputSize)  //(hn,num),(num,inputSize)=>(hn,inputSize)
	
	val out = Symbol.Group(y,d_W,d_U,h,h_act1)
	
	val data_nda =Random.uniform(0,1, Shape(batchSize,inputSize), ctx) 
	val W_nda = Random.uniform(0,1,Shape(hiddenSize,inputSize), ctx)*1e-8f
	val U_nda = Random.uniform(0,1, Shape(classSize,hiddenSize), ctx)*1e-8f
	val label_nda = Random.uniform(0,1, Shape(batchSize,classSize), ctx)
	
//	println(W_nda)
	
	
	
	
	// gradient
	 val data_nda_g =Random.uniform(0,1, Shape(batchSize,inputSize), ctx) 
	val W_nda_g = NDArray.zeros(Shape(hiddenSize,inputSize), ctx)
	val U_nda_g = NDArray.zeros(Shape(classSize,hiddenSize), ctx)
	val label_nda_g = Random.uniform(0,1, Shape(batchSize,classSize), ctx)
	
	val in_args = Map("data"->data_nda,"W"->W_nda,"U"->U_nda,"label"->label_nda) 
		
    val arg_grad_store = Map("data"->data_nda_g,"W"->W_nda_g,"U"->U_nda_g,"label"->label_nda_g)
	
	val executor = out.easy_bind(ctx,in_args, arg_grad_store)
	
	def forward(batch:DataBatch){
		
		assert(batch.data(0).shape(1)== inputSize)
		batch.data(0).copyTo(data_nda)
		NDArray.onehotEncode(batch.label(0),label_nda)
//		println(label_nda)
//		batch.label(0).copyTo(label_nda)
		
//		println(label_nda)
		executor.forward(true)
		
//		val h = NDArray.sigmod(NDArray.dot(W, data))
//		val z = NDArray.sigmod(NDArray.dot(U, h))
		
//		var expy = NDArray.exp(z)
//        p(t) = expy / (NDArray.sum(expy).toScalar)
//      println("hehe:" + p(t).toArray(targets(t - 1)))
//      loss += -scala.math.log(p(t).toArray(targets(t - 1))) //损失函数,交叉熵
	}
	
//	def backward(){
//		
////		executor.backward()
////		println(W_nda)
//	}
	
	def update(learningRate:Float = 0.9f){
		W_nda -= this.outputs(1) *learningRate
		U_nda -= this.outputs(2) * learningRate
		
//		
//		
//		W_nda_g 
//		W_nda -= W_nda_g
//		U_nda_g *= learningRate
//		U_nda -= U_nda_g
//		println(U_nda_g.slice(0))
//		println((U_nda.slice(0)))
//        arg_grad_store("W") *= learningRate
//        in_args("W") -= arg_grad_store("W")
//        arg_grad_store("U") *= learningRate
//        in_args("U") += arg_grad_store("U")
	}
	
	def error(label:NDArray):Float = {
		
		val label_pred = NDArray.argmaxChannel(executor.outputs(0))
//		println(label_pred)
		var right = 0
		val num_instance = label_pred.shape(0)
		 for (i <- 0 until num_instance) {
			if(scala.math.abs(label_pred(i) - label(i)) < this.eps) 
				right += 1
    	}
		 right * 1.0f / num_instance
	}
	
	def output_accuracy(pred: NDArray, target: NDArray): Float = {
    val num_instance = pred.shape(0)
    val eps = 1e-6
    var right = 0
    for (i <- 0 until num_instance) {
      var mx_p = pred(i, 0)
      var p_y: Float = 0
      for(j <- 0 until 5){
        if(pred(i,j) > mx_p){
          mx_p = pred(i,j)
          p_y = j
        }
      }
      if(scala.math.abs(p_y - target(i)) < eps) right += 1
    }
    right * 1.0f / num_instance
  }
	
	
	def outputs = executor.outputs
	def dispose(){
		executor.dispose()
	}
	
	
}

class MLP_auto(val batchSize:Int, val inputSize:Int,val hiddenSize:Int,val classSize:Int)(implicit ctx:Context){
	val eps  = 1e-8
	val data = Symbol.CreateVariable("data")
	val W = Symbol.CreateVariable("W")
	val U = Symbol.CreateVariable("U")
	val label = Symbol.CreateVariable("label")
	
	val h = Symbol.FullyConnected("h")(Map("data" -> data, "num_hidden" -> hiddenSize,"weight"->W,"no_bias"->true))
	val h_act1 = Symbol.Activation()(Map("data" -> h, "name" -> "h_act1", "act_type" -> "sigmoid"))  
	
	val z = Symbol.FullyConnected("z")(Map("data" -> h_act1, "num_hidden" -> classSize,"weight"->U,"no_bias"->true))
	val y = Symbol.SoftmaxActivation("y")(Map("data"->z))
	val ysoft = Symbol.SoftmaxOutput("ysoft")(Map("data"->z,"label"->label))
	
	val data_nda =Random.uniform(0,1, Shape(batchSize,inputSize), ctx) 
	val W_nda = Random.uniform(0,1,Shape(hiddenSize,inputSize), ctx)
	val U_nda = Random.uniform(0,1, Shape(classSize,hiddenSize), ctx)
	val label_nda = Random.uniform(0,1, Shape(batchSize), ctx)
	
//	println(W_nda)
	
	// gradient
	 val data_nda_g =Random.uniform(0,1, Shape(batchSize,inputSize), ctx) 
	val W_nda_g = NDArray.zeros(Shape(hiddenSize,inputSize), ctx)
	val U_nda_g = NDArray.zeros(Shape(classSize,hiddenSize), ctx)
	val label_nda_g = Random.uniform(0,1, Shape(batchSize), ctx)
	
	val in_args = Map("data"->data_nda,"W"->W_nda,"U"->U_nda,"label"->label_nda) 
		
    val arg_grad_store = Map("data"->data_nda_g,"W"->W_nda_g,"U"->U_nda_g,"label"->label_nda_g)
	
	val executor = ysoft.easy_bind(ctx,in_args, arg_grad_store)
	
	def forward(batch:DataBatch){
		
		assert(batch.data(0).shape(1)== inputSize)
		batch.data(0).copyTo(data_nda)
		batch.label(0).copyTo(label_nda)
//		println(label_nda)
		executor.forward(true)
		
//		val h = NDArray.sigmod(NDArray.dot(W, data))
//		val z = NDArray.sigmod(NDArray.dot(U, h))
		
//		var expy = NDArray.exp(z)
//        p(t) = expy / (NDArray.sum(expy).toScalar)
//      println("hehe:" + p(t).toArray(targets(t - 1)))
//      loss += -scala.math.log(p(t).toArray(targets(t - 1))) //损失函数,交叉熵
	}
	
	def backward(){
		executor.backward()
//		println(W_nda)
	}
	
	def update(learningRate:Float = 0.9f){
		
//		println(W_nda_g)
//		println(in_args("W"))
		W_nda_g *= learningRate
		W_nda -= W_nda_g
		U_nda_g *= learningRate
		U_nda -= U_nda_g
		println(U_nda_g.slice(0))
		println((U_nda.slice(0)))
//        arg_grad_store("W") *= learningRate
//        in_args("W") -= arg_grad_store("W")
//        arg_grad_store("U") *= learningRate
//        in_args("U") += arg_grad_store("U")
	}
	
	def error(label:NDArray):Float = {
		
		val label_pred = NDArray.argmaxChannel(executor.outputs(0))
//		println(label_pred)
		var right = 0
		val num_instance = label_pred.shape(0)
		 for (i <- 0 until num_instance) {
			if(scala.math.abs(label_pred(i) - label(i)) < this.eps) 
				right += 1
    	}
		 right * 1.0f / num_instance
	}
	
	def output_accuracy(pred: NDArray, target: NDArray): Float = {
    val num_instance = pred.shape(0)
    val eps = 1e-6
    var right = 0
    for (i <- 0 until num_instance) {
      var mx_p = pred(i, 0)
      var p_y: Float = 0
      for(j <- 0 until 5){
        if(pred(i,j) > mx_p){
          mx_p = pred(i,j)
          p_y = j
        }
      }
      if(scala.math.abs(p_y - target(i)) < eps) right += 1
    }
    right * 1.0f / num_instance
  }
	
	
	def outputs = executor.outputs
	def dispose(){
		executor.dispose()
	}
}
