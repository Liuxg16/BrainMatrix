package thu.brainmatrix.synapse
import thu.brainmatrix.util.RK4
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
class Engine(ctx:Context = Context.defaultCtx) {
  	
	def run(model:Model,t0:NDArray, y0:Array[NDArray], h:NDArray, stepSize:Int):(NDArray,Array[NDArray]) = {
		val rk4 = new RK4(model.update)
		val (t, y) = rk4.solve(t0, y0, h, stepSize)(ctx)
		(t,y)
	}
	
	
}