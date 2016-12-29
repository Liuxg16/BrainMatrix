package thu.brainmatrix.lstmSort
import thu.brainmatrix._
object Network {
  def lenet:Symbol = {
	val data = Symbol.CreateVariable("data")
	val label = Symbol.CreateVariable("softmax_label")
    val conv1 = Symbol.Convolution()(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 20, "kernel" -> (5, 5)/*, "stride" -> (2, 2)*/))

    val act1 = Symbol.Activation()(Map("data" -> conv1, "name" -> "tanh1", "act_type" -> "tanh"))                       
    val mp1 = Symbol.Pooling()(Map("data" -> act1, "name" -> "mp1",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
                                 
    //second conv
    val conv2 = Symbol.Convolution()(Map("data" -> mp1, "name" -> "conv2", "num_filter" -> 50,
                                       "kernel" -> (5, 5), "stride" -> (2, 2)))
    val act2 = Symbol.Activation()(Map("data" -> conv2, "name" -> "tanh2", "act_type" -> "tanh"))
    val mp2 = Symbol.Pooling()(Map("data" -> act2, "name" -> "mp2",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
                              
    //first fullc
    val fl = Symbol.Flatten()(Map("data" -> mp2, "name" -> "flatten"))
    val fc1 = Symbol.FullyConnected()(Map("data" -> fl, "name" -> "fc1", "num_hidden" -> 500))
    val act3 = Symbol.Activation()(Map("data" -> fc1, "name" -> "tanh3", "act_type" -> "tanh"))
    
    //second fullc
    val fc2 = Symbol.FullyConnected()(Map("data" -> act3, "name" -> "fc2", "num_hidden" -> 10))
    
    //loss
    val sm = Symbol.SoftmaxOutput()(Map("data" -> fc2,"label"->label, "name" -> "sm"))
    val smce = Symbol.Softmax_cross_entropy(fc2, label)
 	val loss = Symbol.MakeLoss("makeloss")(Map("data"->smce))
 	loss
  }
  
  def lenet1:Symbol = {
	val data = Symbol.CreateVariable("data")
	val label = Symbol.CreateVariable("softmax_label")
    val conv1 = Symbol.Convolution()(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 20, "kernel" -> (5, 5)/*, "stride" -> (2, 2)*/))

    val act1 = Symbol.Activation()(Map("data" -> conv1, "name" -> "tanh1", "act_type" -> "tanh"))                       
    val mp1 = Symbol.Pooling()(Map("data" -> act1, "name" -> "mp1",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
                                 
    //second conv
    val conv2 = Symbol.Convolution()(Map("data" -> mp1, "name" -> "conv2", "num_filter" -> 50,
                                       "kernel" -> (5, 5), "stride" -> (2, 2)))
    val act2 = Symbol.Activation()(Map("data" -> conv2, "name" -> "tanh2", "act_type" -> "tanh"))
    val mp2 = Symbol.Pooling()(Map("data" -> act2, "name" -> "mp2",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))
                              
    //first fullc
    val fl = Symbol.Flatten()(Map("data" -> mp2, "name" -> "flatten"))
    val fc1 = Symbol.FullyConnected()(Map("data" -> fl, "name" -> "fc1", "num_hidden" -> 500))
    val act3 = Symbol.Activation()(Map("data" -> fc1, "name" -> "tanh3", "act_type" -> "tanh"))
    
    //second fullc
    val fc2 = Symbol.FullyConnected()(Map("data" -> act3, "name" -> "fc2", "num_hidden" -> 10))
    
    //loss
    val sm = Symbol.SoftmaxOutput()(Map("data" -> fc2,"label"->label))
    sm
  }
  
  
  def mlp():Symbol = {
	    val data = Symbol.CreateVariable("data")
	    val weight_1 = Symbol.CreateVariable("weight_1")
	    val weight_2 = Symbol.CreateVariable("weight_2")
	    val weight_3 = Symbol.CreateVariable("weight_3")
	    val label = Symbol.CreateVariable("softmax_label")
 		val fc1 = Symbol.FullyConnected()(Map("data" -> data, "name" -> "fc1", "weight"->weight_1,"no_bias"->true,"num_hidden" -> 128))
 		
	    val act1 = Symbol.Activation()(Map("data" -> fc1, "name" -> "relu1", "act_type" -> "relu"))
	    
	    val fc2 = Symbol.FullyConnected()(Map("data" -> act1, "name" -> "fc2", "num_hidden" -> 64))
 		val fc3 = Symbol.FullyConnected()(Map("data" -> data, "name" -> "fc3", "num_hidden" -> 10))
 		val sm = Symbol.SoftmaxOutput("sm")(Map("data" -> fc3))
 		val smce = Symbol.Softmax_cross_entropy(fc3, label)
 		
// 		val loss = Symbol.MakeLoss("makeloss")(Map("data"->(smce+Symbol.sum(Symbol.square(weight_1))*0.0003f)))
 		val loss = Symbol.MakeLoss("makeloss")(Map("data"->smce))
 		Symbol.Group(loss,sm)  
  	}
  	
  def mlp1():Symbol = {
	    val data = Symbol.CreateVariable("data")
	    val weight_1 = Symbol.CreateVariable("weight_1")
	    val weight_2 = Symbol.CreateVariable("weight_2")
	    val weight_3 = Symbol.CreateVariable("weight_3")
	    val label = Symbol.CreateVariable("softmax_label")
 		val fc1 = Symbol.FullyConnected()(Map("data" -> data, "name" -> "fc1", "weight"->weight_1,"no_bias"->true,"num_hidden" -> 128))
	    val act1 = Symbol.Activation()(Map("data" -> fc1, "name" -> "relu1", "act_type" -> "relu"))
	    
	    val fc2 = Symbol.FullyConnected()(Map("data" -> act1, "name" -> "fc2", "num_hidden" -> 64))
	    val act2 = Symbol.Activation()(Map("data" -> fc2, "name" -> "relu1", "act_type" -> "relu"))
	    
 		val fc3 = Symbol.FullyConnected()(Map("data" -> act2, "name" -> "fc3", "num_hidden" -> 10))
 		val sm = Symbol.SoftmaxOutput("sm")(Map("data" -> fc3,"label"->label))
 		val smce = Symbol.Softmax_cross_entropy(fc3, label)
 		val loss = Symbol.MakeLoss("makeloss")(Map("data"->smce))
 		Symbol.Group(loss,sm) 
  	}
  
}