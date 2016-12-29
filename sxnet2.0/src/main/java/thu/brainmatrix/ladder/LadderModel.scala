package thu.brainmatrix.ladder
import thu.brainmatrix.Symbol
import scala.collection.mutable.ListBuffer
import thu.brainmatrix.NDArray
import thu.brainmatrix.Initializer
import  scala.collection.immutable.Range
import thu.brainmatrix.DataIter
import thu.brainmatrix.Optimizer
import thu.brainmatrix.MAE
import thu.brainmatrix.IO
import thu.brainmatrix.optimizer.SGD

import org.slf4j.LoggerFactory

  /*
   *  
   * by liuxianggen
   * 2016-05-11
   * @param data:the input symbol
   * @param dims:dimension of the network
   * 
   */
class LadderModel(val dims:Vector[Int],val sparse_penalty:Float=0,
      pt_dropout:Float=0,ft_dropout:Float=0,input_act:String=null,
      internal_act:String = "relu",output_act:String=null) extends AEModel {
    
    val N = dims.length-1
    val stacks = ListBuffer[Symbol]()
    val data = Symbol.CreateVariable("data")
    
    /*
     * config  each layer
     * 
     */
    
    
    var decoder_act:String = null
    var idropout = 0f
    var odropout = 0f
    var encoder_act:String = null
    for(i <- 0 until N){
        if(i==0){
            decoder_act = input_act
            idropout  = 0f
        }else{
            decoder_act = internal_act
            idropout = pt_dropout
        }
        if(this.N-1 == i){
            encoder_act = output_act
            odropout = 0f
        }else{
            encoder_act = internal_act
            odropout = pt_dropout
        }
        val (istack,iargs,iargs_grad,iargs_mult,iauxs) = make_stack(i,data,dims(i),dims(i+1),sparse_penalty,idropout,odropout,
                encoder_act,decoder_act)
        
        // the key symbol of each layer
        this.stacks.append(istack)
        this.args.++=(iargs)
        this.args_grad ++= iargs_grad
        this.args_mult ++=iargs_mult
        this.auxs ++=iauxs
        
    }
    
    /**
     * encoder: key symbol the forward network of this autoencoder network 
     * internals: each encoder in forward network
     */
    val (encoder,internals) = make_encoder(this.data,dims,sparse_penalty,ft_dropout,internal_act,output_act)
    val decoder = make_decoder(this.encoder,dims,sparse_penalty,ft_dropout,internal_act,input_act)
    if(input_act=="softmax"){
        this.loss = this.decoder
    }else{
        this.loss = Symbol.LinearRegressionOutput()(Map("data"->this.decoder,"label"->this.data))
    }
    
    def make_encoder(data:Symbol,dims:Vector[Int],sparse_penalty:Float=0f,dropout:Float = 0f,
            internal_act:String = "relu",output_act:String = null):(Symbol,ListBuffer[Symbol])={
        
        var x = data
        val internals = ListBuffer[Symbol]()
        val N = dims.length-1
        for(i<-0 until N){
            x = Symbol.FullyConnected(name="encoder_%d".format(i))(Map("data"->x,"num_hidden"->dims(i+1)))
            if(internal_act!=null && i<N-1){
                x = Symbol.Activation()(Map("data"->x,"act_type"->internal_act))
                if(internal_act=="sigmoid" && sparse_penalty!=0f){
                    x = Symbol.IdentityAttachKLSparseReg("sparse_encoder_%d".format(i))(Map("data"->x,"penalty"->sparse_penalty))
                }
            }else if(output_act!=null && i==N-1){
                    x = Symbol.Activation()(Map("data"->x,"act_type"->output_act))
                    if(output_act=="sigmoid" && sparse_penalty!=0f){
                        x = Symbol.IdentityAttachKLSparseReg("sparse_encoder_%d".format(i))(Map("data"->x,"penalty"->sparse_penalty))
                    }
            }
            
            if(dropout!=0){
                x = Symbol.Dropout()(Map("data"->x,"p"->dropout))
            }
            internals.append(x)      
        } 
//        internals.foreach { x => println(x.debugStr+"\n-------------------------\n") }
        (x,internals)
        
    }
    
    
    
    
    
     def make_decoder(feature:Symbol,dims:Vector[Int],sparse_penalty:Float=0f,dropout:Float = 0f,
            internal_act:String = "sigmoid",input_act:String = null):Symbol = {
        
        var x = feature
        val N = dims.length-1
        for(i<- Range(0,N).reverse){
            x = Symbol.FullyConnected(name="decoder_%d".format(i))(Map("data"->x,"num_hidden"->dims(i)))
            if(internal_act!=null && i>0){
                x = Symbol.Activation()(Map("data"->x,"act_type"->internal_act))
                if(internal_act=="sigmoid" && sparse_penalty!=0f){
                    x = Symbol.IdentityAttachKLSparseReg("sparse_decoder_%d".format(i))(Map("data"->x,"penalty"->sparse_penalty))
                }
            }else if(input_act!=null && i==0){
                    x = Symbol.Activation()(Map("data"->x,"act_type"->input_act))
                    if(input_act=="sigmoid" && sparse_penalty!=0f){
                        x = Symbol.IdentityAttachKLSparseReg("sparse_decoder_%d".format(i))(Map("data"->x,"penalty"->sparse_penalty))
                    }
            }
            
            
            var x_lat = 
              if(i==0)  Symbol.Activation()(Map("data"->this.data,"act_type"->"relu"))
              else Symbol.Activation()(Map("data"->this.internals(i-1),"act_type"->"relu"))
//            if(internal_act=="sigmoid" && sparse_penalty!=0f){
//                x_lat = Symbol.IdentityAttachKLSparseReg("sparse_lateral_%d"
//                        .format(i))(Map("data"->x_lat,"penalty"->sparse_penalty))
//            }
            x = x * 0.95 + x_lat * 0.05
            
            if(dropout!=0 && i>0){
                x = Symbol.Dropout()(Map("data"->x,"p"->dropout))
            }
                 
        } 
        x
        
    }
    
    
    def make_stack(istack:Int ,data:Symbol,num_input:Int,num_hidden:Int,
            sparse_penalty:Float=0f,idropout:Float = 0f,odropout:Float=0f,
            encoder_act:String = "relu",decoder_act:String = "relu"):(Symbol,ListBuffer[(String,NDArray)],
                    ListBuffer[(String,NDArray)],ListBuffer[(String,Float)],ListBuffer[(String,NDArray)]) = {
        var x = data
        if(0f!=idropout){
            x = Symbol.Dropout()(Map("data"->data,"p"->idropout))
        }
        x = Symbol.FullyConnected(name="encoder_%d".format(istack))(Map("data"->x,
                "num_hidden"->num_hidden))
        if(encoder_act!=null){
            x = Symbol.Activation()(Map("data"->x,"act_type"->encoder_act))
            if(encoder_act=="sigmoid" && sparse_penalty!=0f){
                x = Symbol.IdentityAttachKLSparseReg("sparse_encoder_%d"
                        .format(istack))(Map("data"->x,"penalty"->sparse_penalty))
            }
        }
        
        if(0f!=odropout){
            x = Symbol.Dropout()(Map("data"->x,"p"->idropout))
        }
        x = Symbol.FullyConnected(name="decoder_%d".format(istack))(Map("data"->x,
                "num_hidden"->num_input))
        
        if(decoder_act=="softmax"){
            x = Symbol.SoftmaxOutput()(Map("data"->x,"label"->data,"prob_label"->true,"act_type"->decoder_act))
        }else if(decoder_act != null){
            x = Symbol.Activation()(Map("data"->x,"act_type"->decoder_act))
            if(decoder_act=="sigmoid" && sparse_penalty!=0f){
                x = Symbol.IdentityAttachKLSparseReg("sparse_decoder_%d"
                        .format(istack))(Map("data"->x,"penalty"->sparse_penalty))
            }
        var x_lat = Symbol.Activation()(Map("data"->data,"act_type"->"relu"))
//        if(decoder_act=="sigmoid" && sparse_penalty!=0f){
//                x_lat = Symbol.IdentityAttachKLSparseReg("sparse_lateral_%d"
//                        .format(istack))(Map("data"->x_lat,"penalty"->sparse_penalty))
//            }
//        x = Symbol.ElementWiseSum(Array(x_lat,x), s"lateral_decoder_$istack")
        x = x * 0.95 + x_lat * 0.05
//        x = Symbol.Activation1(Map("data"->x,"act_type"->"sigmoid"))
            
        x  = Symbol.LinearRegressionOutput()(Map("data"->x,"label"->data))
        }else{
            x  = Symbol.LinearRegressionOutput()(Map("data"->x,"label"->data))
        }
        
        val args_t = ListBuffer(("encoder_%d_weight".format(istack),NDArray.empty(this.xpu,num_hidden,num_input)),
                ("encoder_%d_bias".format(istack),NDArray.empty(this.xpu, num_hidden)),
                ("decoder_%d_weight".format(istack), NDArray.empty(this.xpu,num_input, num_hidden)),
                ("decoder_%d_bias".format(istack),NDArray.empty(this.xpu,num_input)))
        
        val args_grad_t = ListBuffer(("encoder_%d_weight".format(istack),NDArray.zeros(this.xpu,num_hidden,num_input)),
                ("encoder_%d_bias".format(istack),NDArray.zeros(this.xpu, num_hidden)),
                ("decoder_%d_weight".format(istack), NDArray.zeros(this.xpu,num_input, num_hidden)),
                ("decoder_%d_bias".format(istack),NDArray.zeros(this.xpu,num_input)))
            
        val args_mult_t = ListBuffer(("encoder_%d_weight".format(istack),1.0f),
                ("encoder_%d_bias".format(istack),2.0f),
                ("decoder_%d_weight".format(istack),1.0f),
                ("decoder_%d_bias".format(istack),2.0f))
        
        val auxs_t = ListBuffer[(String,NDArray)]()
        if(encoder_act=="sigmoid" && sparse_penalty!=0f){
             auxs_t.append(("sparse_encoder_%d_moving_avg".format(istack),NDArray.ones(this.xpu,num_hidden)*0.5f))
            }
        if(decoder_act=="sigmoid" && sparse_penalty!=0f){
             auxs_t.append(("sparse_decoder_%d_moving_avg".format(istack),NDArray.ones(this.xpu,num_input)*0.5f))
           
            }
//        auxs_t.append(("sparse_lateral_%d_moving_avg".format(istack),NDArray.ones(this.xpu,num_input)*0.5f))
         val init_t = new thu.brainmatrix.Uniform(0.07f)
         for((k,v) <- args_t){
             init_t(k,v)  
//             println("------------------------")
//             System.err.println(s"param:$k \t\t stat(mean):$tf")
         }
//         println("------------------------")
//         println(auxs_t.toMap.keys)
         
         (x,args_t,args_grad_t,args_mult_t,auxs_t)
        
    }
  
  
    def layerwise_pretrain(data_iter:DataIter,batch_Size:Int,n_iter:Int,optimizer:Optimizer){
//        def l2_norm(){}
        val solver = new Solver(optimizer)
        solver.set_metric(new MAE())
        solver.set_monitor(new Monitor(3))
        for (i <- 0 until this.N){
            var data_iter_i:DataIter = null
            var X_i = ListBuffer[NDArray]()
            if(i==0){
                data_iter_i = data_iter
                println(s"Pre-training layer $i...")
               
                solver.solve_0(this.xpu, this.stacks(i), this.args, this.args_grad, this.auxs, data_iter_i, 0, n_iter, false)

            }else{
                X_i = AEModel.extract_feature(this.internals(i-1), this.args, this.auxs, data_iter, this.xpu).values.head
                println(s"Pre-training layer $i...")
                solver.solve(this.xpu, this.stacks(i), this.args, this.args_grad, this.auxs, X_i, 0, n_iter, false)

            }
        }
    }
    
    def finetune(data_iter:DataIter,batch_size:Int,n_iter:Int,optimizer:Optimizer){
        
        println("+++++++++++++++++++++++++++")
//        val (argShapes, _, auxShapes) = this.loss.inferShape(Map("data"->Vector(10,1,1,28)))
      
        val solver  = new Solver(optimizer)
        solver.set_metric(new MAE())
        solver.set_monitor(new Monitor(3))
        solver.solve_0(this.xpu,this.loss,this.args,this.args_grad,this.auxs,data_iter,0,n_iter,false)
        
    }
    
    def eval(data_iter:DataIter):Float = {
        val  X_data = AEModel.extract_feature(this.loss, this.args, this.auxs, data_iter, this.xpu).values.head
        data_iter.reset()
        var sum = 0f
        for(x_data<-X_data){
          val temp = NDArray.mean(NDArray.square(x_data-data_iter.next().data(0)))
          sum += temp.toScalar
        }
        
        sum/(X_data.length)
    }
    
}

object LadderModel{
    private val logger = LoggerFactory.getLogger(classOf[LadderModel])
    
    def main(args:Array[String]){
        println("-----------------------AutoEncoder--------------------------------")
        val  batchSize=100
        val iterNum = 60
        val lr_init = 1.4f
//        val ae = new LadderModel(dims = Vector(784,200,50,20,10),sparse_penalty = 0.8f,pt_dropout=0.8f,internal_act="sigmoid", output_act="sigmoid")
        val ae = new LadderModel(dims = Vector(784,200,50,20,10),pt_dropout=0.9f,internal_act="relu", output_act="relu")
        
        //get dataIter
        val trainDataIter = IO.MNISTIter(Map(
	      "image" -> "data/train-images-idx3-ubyte",
	      "label" -> "data/train-labels-idx1-ubyte",
	      "data_shape" -> "(784)",
	      "label_name" -> "sm_label",
	      "batch_size" -> batchSize.toString,
	      "shuffle" -> "1",
	      "flat" -> "1",
	      "silent" -> "0",
	      "seed" -> "10"))
        
        ae.layerwise_pretrain(trainDataIter, batchSize, iterNum, optimizer=new SGD(learningRate = lr_init, momentum = 0f, wd = 0f))
        println("Finetune ....")
        ae.finetune(trainDataIter, batchSize, iterNum, optimizer=new SGD(learningRate = lr_init, momentum = 0f, wd = 0f))
        
        println("Evaluation ......")
        val training_error = ae.eval(trainDataIter)
        println(s"training error:$training_error")
        
         //get dataIter
        val valDataIter = IO.MNISTIter(Map(
	      "image" -> "data/t10k-images-idx3-ubyte",
	      "label" -> "data/t10k-labels-idx1-ubyte",
	      "data_shape" -> "(784)",
	      "label_name" -> "sm_label",
	      "batch_size" -> batchSize.toString,
	      "shuffle" -> "1",
	      "flat" -> "1",
	      "silent" -> "0",
	      "seed" -> "10"))
        
	      val val_error = ae.eval(valDataIter)
        println(s"val error:$val_error")
        
        
        
        
        println("-----------------------AutoEncoder--------------------------------")
    }
}




