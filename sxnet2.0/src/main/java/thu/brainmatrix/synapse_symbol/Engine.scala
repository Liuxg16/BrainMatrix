package thu.brainmatrix.synapse_symbol
import thu.brainmatrix.util.RK4
import thu.brainmatrix.Executor
import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
class Engine(ctx:Context = Context.defaultCtx,val model:Model) {
  	var executor:Executor = null
  	var executor1:Executor = null
	
	val t_onehot :NDArray = NDArray.zeros(Shape(Config.NUMBER,Config.SPIKENUM), ctx)
	
	def run(t0:NDArray, y0:Array[NDArray], h:NDArray, stepSize:Int):(NDArray,Array[NDArray]) = {
  		
		val rk4 = new RK4(functions)
		val (t, y) = rk4.solve(t0, y0, h, stepSize)(ctx)
		(t,y)
	}
	
	
	def build(module:Module = null) {
		val in_args = Map("t_onehot"->t_onehot) ++ this.model.symbolMap ++ this.model.getInitialMap() ++ Config.MAP
		
//		model.update().listArguments().foreach{z => println(in_args(z).shape)}
//		in_args.keySet.foreach {println}
		
    	val arg_grad_store = Map("t_onehot"->NDArray.zeros(Shape(1), ctx))
	
		this.executor = this.model.update().easy_bind(ctx,in_args, arg_grad_store)
		if(module !=null)
			this.executor1 = module.getSymbol().easy_bind(ctx,in_args, arg_grad_store)
		
	}
	
	def functions(t: NDArray,y:Array[NDArray]):Array[NDArray] ={
		val t_1 = NDArray.array(t.toArray,Shape(Config.NUMBER),ctx)
		NDArray.onehotEncode(t_1, t_onehot)
		
		this.model.getInitialY() zip y map{case (a,b) =>{
			b.copyTo(a)
		}}
		
		
		
//		this.executor1.forward()
//		println(this.executor1.outputs(0))
		
		this.executor.forward()
		this.executor.outputs
		
	}
	
	
	def dispose(){
		this.executor.dispose()
		this.t_onehot.dispose()
	}
	
}