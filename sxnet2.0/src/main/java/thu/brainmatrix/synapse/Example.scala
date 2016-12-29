package thu.brainmatrix.synapse
import thu.brainmatrix.Base

import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.util.Draw
object Example {
  	def main(args:Array[String]){
  		Base.welcome
  		val starttime = System.currentTimeMillis()
  		test()
  		val elapsetime = System.currentTimeMillis() - starttime
  		println(elapsetime)
  	}
	
	def test(){
		val ctx = Context.cpu(0)
		val steps_num:Int =20000
		
		// create an input source
		// presynaptic spikes
		val xpreinput1 = new Input( steps_num)(ctx);  
		
		xpreinput1.initial(3)
//		print(NDArray.min(xpreinput1.current))
		
		// create an axon
		val xaxon1 = new Axon(ctx);
		xaxon1.addSpikeInput(xpreinput1);
		// create a dendrite
		val  xdendrite1 = new Dendrite(ctx);
		// create an synapse
		val  xsynapse1 =  new Synapse(ctx);
		
		xaxon1.addSynapse(xsynapse1);
		xdendrite1.addSynapse(xsynapse1);
		
		// input with higher input rates
		val xpreinput2 = new Input(steps_num)(ctx);
		xpreinput2.initial(5)
		
		val xaxon2 = new Axon(ctx);
		xaxon2.addSpikeInput(xpreinput2);
		
		
		
		val  xdendrite2 = new Dendrite(ctx);
		val  xsynapse2 = new Synapse(ctx);
		xaxon2.addSynapse(xsynapse2);
		xdendrite2.addSynapse(xsynapse2);
		
		// create an model
		val model = new Model(ctx);
//		
		model.addModule(xaxon1);
		model.addModule(xaxon2);
		model.addModule(xsynapse1);
		model.addModule(xsynapse2);
		model.addModule(xdendrite1);
		model.addModule(xdendrite2);
		
		val y0  = model.getInitial();
//      create a engine
		val engine  = new Engine()
		val t0 		=  NDArray.zeros(Config.SHAPE, ctx)
		val h 		=  NDArray.ones(Config.SHAPE, ctx)
		
		println(xsynapse1.getResindex("preCaBuff"))
		println(xsynapse2.getResindex("preAbeta"))
		
		val (t,y) = engine.run(model, t0, y0, h,steps_num-1);

		val draw = new Draw()
		
//		println(xpreinput1.current)
//		println(t)
//		println(y(xaxon1.getResindex("preVm")).slice(0))
//		
		val tslice0arr = NDArray.transpose(t).slice(0).toArray 
		t.dispose()
		val yrec = y.map { x => NDArray.transpose(x).slice(0).toArray }
		y.foreach { x => x.dispose() }
		
		draw.subplot(4,4,0)
		draw.add_line(tslice0arr, yrec(xaxon1.getResindex("preVm")))
		draw.add_line(tslice0arr, yrec(xaxon2.getResindex("preVm")))
		draw.addInfo("preVM", "time(ms)", "presynaptic Vm(mV)")
		
		draw.subplot(4,4,1)
		draw.add_line(tslice0arr, yrec(xdendrite1.getResindex("postVm")))
		draw.add_line(tslice0arr, yrec(xdendrite2.getResindex("postVm")))
		draw.addInfo("postVm", "time(ms)", "postsynaptic Vm(mV)")
		
		draw.subplot(4,4,2)
		val preCa1 = yrec(xsynapse1.getResindex("preCa"))
		val preCa2 = yrec(xsynapse2.getResindex("preCa"))
		
		draw.add_line(tslice0arr, preCa1)
		draw.add_line(tslice0arr, preCa2)
		draw.addInfo("presynaptic [Ca]i (uM)", "time(ms)", "presynaptic [Ca]i (uM)")
		
		draw.subplot(4,4,3)
		val Sensor1 = yrec(xsynapse1.getResindex("preSensor"))
		val Sensor2 = yrec(xsynapse2.getResindex("preSensor"))
		draw.add_line(tslice0arr, Sensor1)
		draw.add_line(tslice0arr, Sensor2)
		draw.addInfo("presynaptic [Sensor]i", "time(ms)", "presynaptic [Sensor]i")
		
		
		draw.subplot(4,4,4)
		val Pr1 = preCa1 zip Sensor1 map{x => x._1 * x._2}
		val Pr2 = preCa2 zip Sensor2 map{x => x._1 * x._2}
		draw.add_line(tslice0arr, Pr1)
		draw.add_line(tslice0arr, Pr2)
		draw.addInfo("probability of release", "time(ms)")
		
		draw.subplot(4,4,5)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("preCaBuff")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("preCaBuff")))
		draw.addInfo("presynaptic Ca buffer", "time(ms)")
		
		
		draw.subplot(4,4,6)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("aPreCDK")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("aPreCDK")))
		draw.addInfo("aPreCDK", "time(ms)")
		
		
		draw.subplot(4,4,7)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("aPreTrkB")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("aPreTrkB")))
		draw.addInfo("presynaptic TrkB", "time(ms)")
		
		draw.subplot(4,4,8)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("preNR2B")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("preNR2B")))
		draw.addInfo("presynaptic NR2B", "time(ms)")
		
		draw.subplot(4,4,9)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("preMg")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("preMg")))
		draw.addInfo("presynaptic [Mg]i (uM)", "time(ms)")
		
		draw.subplot(4,4,10)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("preAbeta")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("preAbeta")))
		draw.addInfo("presynaptic Abeta", "time(ms)")
		
		draw.subplot(4,4,11)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("qNMDAR")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("qNMDAR")))
		draw.addInfo("postsynaptic NMDAR", "time(ms)")
		
		draw.subplot(4,4,12)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("postCa")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("postCa")))
		draw.addInfo("postsynaptic [Ca]i", "time(ms)")
		
		draw.subplot(4,4,13)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("postCaBuff")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("postCaBuff")))
		draw.addInfo("postsynaptic Ca buffer", "time(ms)")
		
		draw.subplot(4,4,14)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("aPostCN")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("aPostCN")))
		draw.addInfo("postsynaptic CN", "time(ms)")
		
		draw.subplot(4,4,15)
		draw.add_line(tslice0arr, yrec(xsynapse1.getResindex("aPostTrkB")))
		draw.add_line(tslice0arr, yrec(xsynapse2.getResindex("aPostTrkB")))
		draw.addInfo("post TrkB", "time(ms)")
//				
		draw.draw()
//
	}
}