package thu.brainmatrix.synapse_symbol
import thu.brainmatrix.Shape
import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
object Config {
	final val NUMBER = 1000
    final val SHAPE = Shape(1,NUMBER)
    final val SPIKENUM = 10
    
    final val  one_s = Symbol.CreateVariable("one_s")
    final val  zero_s = Symbol.CreateVariable("zero_s")
    final val  spikes_ones_s = Symbol.CreateVariable("spikes_ones_s")
    final val CTX = Context.cpu(0)
    final val onenda = NDArray.ones(SHAPE, CTX)
    final val zerosnda = NDArray.zeros(SHAPE, CTX)
    final val spikes_ones_nda = NDArray.ones(Shape(SPIKENUM,1), CTX)
    final val MAP    = Map("one_s"->onenda,"zero_s"->zerosnda,"spikes_ones_s"->spikes_ones_nda)
    
}