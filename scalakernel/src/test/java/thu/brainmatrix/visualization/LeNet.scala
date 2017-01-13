package thu.brainmatrix.visualization

import thu.brainmatrix.Symbol

/**
 * @author Liu Xianggen
 */
object LeNet {

  def getSymbol(numClasses: Int = 10): Symbol = {
    val data = Symbol.CreateVariable("data")
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
    val fc2 = Symbol.FullyConnected()(Map("data" -> act3, "name" -> "fc2", "num_hidden" -> 30))
    
    //loss
    val softmax = Symbol.SoftmaxOutput()(Map("data" -> fc2, "name" -> "sm"))
    softmax
  }
}
