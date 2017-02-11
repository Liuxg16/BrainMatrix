package thu.brainmatrix.visualization

import thu.brainmatrix.synapse_symbol._
import thu.brainmatrix.Shape
import scala.util.parsing.json._
import thu.brainmatrix.Symbol
import thu.brainmatrix.Visualization
object SynapseVis {
  def main(args: Array[String]): Unit = {
    val leis = new ExampleVis
    leis.net
    
    val ctx = Config.CTX
    val xpreinput1 = new Input("input1")(ctx);  
		
		xpreinput1.initial(3)
		
		// create an axon
		val xaxon1 = new Axon(ctx,"axon1");
		xaxon1.addSpikeInput(xpreinput1);
		
		// create a dendrite
		val  xdendrite1 = new Dendrite(ctx,"Dendrite1");
		// create an synapse
		val  xsynapse1 =  new Synapse(ctx,"Synapse1");
		
		xaxon1.addSynapse(xsynapse1);
		xdendrite1.addSynapse(xsynapse1);
		
		// input with higher input rates
		val xpreinput2 = new Input("input2")(ctx);
		xpreinput2.initial(5)
		
		val xaxon2 = new Axon(ctx,"axon2");
		xaxon2.addSpikeInput(xpreinput2);
		
		val  xdendrite2 = new Dendrite(ctx,"Dendrite2");
		val  xsynapse2 = new Synapse(ctx,"Synapse2");
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
    
    
    val (sym, shape) = (model.update(),Shape(1, 1, 28, 28))
      
      
      val dot = Visualization.plotNetwork(symbol = sym,
          title = leis.net, shape = Map("data" -> shape),
          nodeAttrs = Map("shape" -> "rect", "fixedsize" -> "false"))

          
      dot.render(engine = "dot", format = "pdf", fileName = leis.net, path = leis.outDir)

  }

}