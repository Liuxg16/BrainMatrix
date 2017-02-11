package thu.brainmatrix.synapse_symbol
import thu.brainmatrix.util.RK4
import scala.util.parsing.json._
import thu.brainmatrix.Symbol
import thu.brainmatrix.Visualization
import thu.brainmatrix.Executor
import thu.brainmatrix.NDArray
import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
class Engine(ctx:Context = Context.defaultCtx,val model:Model) {
  	var executor:Executor = null
  	var executor1:Executor = null
	
	val t_onehot :NDArray = NDArray.zeros(Shape(Config.NUMBER,Config.SPIKENUM), ctx)
	var in_args  =  Map[String, NDArray]() 
	
	def run(t0:NDArray, y0:Array[NDArray], h:NDArray, stepSize:Int):(NDArray,Array[NDArray]) = {
  		
		val rk4 = new RK4(functions)
		val (t, y) = rk4.solve(t0, y0, h, stepSize)(ctx)
		
		(t,y)
	}
	
	
	def build(module:Module = null) {
		this.in_args = Map("t_onehot"->t_onehot) ++ this.model.symbolMap ++ this.model.getInitialMap() ++ Config.MAP
		
//		model.update().listArguments().foreach{z => println(in_args(z).shape)}
//		in_args.keySet.foreach {println}
		
    	val arg_grad_store = Map("t_onehot"->NDArray.zeros(Shape(1), ctx))
	
		this.executor = this.model.update().easy_bind(ctx,in_args, arg_grad_store)
		if(module !=null)
			this.executor1 = module.getSymbol().easy_bind(ctx,in_args, arg_grad_store)
		
	}
	
	def plot(){
		val netName = "synapse_net"
		val sym = this.model.update()
		val in_args = Map("t_onehot"->t_onehot) ++ this.model.symbolMap ++ this.model.getInitialMap() ++ Config.MAP
		val  shape_init = in_args.map(arg => (arg._1,arg._2.shape)) 
		
		val dot = Visualization.plotNetwork(symbol = sym,
          title =netName , shape = shape_init,
          nodeAttrs = Map("shape" -> "rect", "fixedsize" -> "false"))

          
        dot.render(engine = "dot", format = "pdf", fileName = netName, path = "output/")
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
		this.in_args.values.foreach { x => x.dispose() }
		
		this.t_onehot.dispose()
		
		this.executor.dispose()
		
	}
	
}