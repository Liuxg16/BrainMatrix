package thu.brainmatrix.sae
import scala.collection.mutable.ListBuffer
import org.slf4j.LoggerFactory
import thu.brainmatrix.Optimizer
import thu.brainmatrix.optimizer.SGD
import thu.brainmatrix.EvalMetric
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.Base
import thu.brainmatrix.Symbol
import thu.brainmatrix.DataIter
import thu.brainmatrix.MXKVStoreUpdater

class Solver(var optimizer:Optimizer = new SGD()) {
    private val logger = LoggerFactory.getLogger(classOf[Solver])
    var updater = Optimizer.getUpdater(this.optimizer)
    var metric :EvalMetric = null
    var monitor:Monitor = null
    
    def set_metric(metric:EvalMetric){
        this.metric = metric
    }
    def set_monitor(monitor:Monitor){
        this.monitor = monitor
    }
    
    def solve(xpu:Context,sym:Symbol,args:ListBuffer[(String,NDArray)],
            args_grad:ListBuffer[(String,NDArray)],auxs:ListBuffer[(String,NDArray)],
            X_i:ListBuffer[NDArray],begin_iter:Int=0,end_iter:Int,debug:Boolean=false){

            val input_desc:Map[String,Shape] = Map("data"->X_i.head.shape)
            val input_names = input_desc.keys
            val input_buffs = input_desc.map(x => NDArray.empty(x._2, xpu))
            
            val args_t = args.toMap ++ input_names.zip(input_buffs).toMap
            val output_names = sym.listOutputs()
            if(debug){
                logger.info("need to code in details")
            }
                        
            val exe = sym.easy_bind(xpu,args=args_t,argsGrad = args_grad.toMap,auxStates = auxs.toMap)
            
//            println("----------------------------")
//            println(sym.debugStr)
//            println(exe.debugStr)
            
            require(sym.listArguments().length==exe.gradArrays.length,"dismatch error in solve Solver.scala ")
            var update_dict = sym.listArguments().zip(exe.gradArrays).toMap
            
            update_dict = update_dict.-(sym.listArguments()(0))
//            sym.listArguments().foreach(println)
//            println(update_dict.length)
//            
            val batch_size = input_buffs.head.shape(0)
            this.optimizer.setRescaleGrad(1.0f/batch_size)
            
            /**
             * output_dict :output info (String,NDArray
             * output_buff : the new buffer refered to output_dict
             * internal_dict: internal nodes ,not the output
             */
            
            var output_dict = ListBuffer[(String,NDArray)]()
            var output_buff = ListBuffer[(String,NDArray)]()
            var internal_dict = input_names.zip(input_buffs).toMap
            for((key,arr)<-sym.listOutputs().zip(exe.outputs)){
                if(output_names.contains(key)){
                    output_dict :+= (key,arr)
                    output_buff :+= (key,NDArray.empty(arr.shape,Context.defaultCtx))
                }else{
                    internal_dict += (key->arr)
                 }
            }
            val output_buff_m = output_buff.toMap
            
            /**
             * training start....
             * 
             */
            for(i<- begin_iter until end_iter){
//            	println(s"------------------------$i-----")
                
                
                /**
                 * update the input training data
                 */
                X_i(i).copyTo(input_buffs.head)
                
                exe.forward(isTrain=true)
                
                /**
                 * internal node info: internal_dict
                 */
                if(this.monitor!=null){
                    this.monitor.forward_end(i, internal_dict)
                }
                
                /***
                 * backup the output info
                 */
                for(key<-output_dict){
                    key._2.copyTo(output_buff_m(key._1))
                }
                
                
                exe.backward()
//                println(s"------------------------$i-----")
                
//                println(sym.debugStr)
//                println(exe.debugStr)
                
                updateParams(args_t,update_dict,this.updater)
//                 println(s"------------------------$i-----")
                
                if(this.metric!=null){
//                	println(input_buffs.last.shape)
//                	println(output_buff_m(output_names(0)).shape)
                    this.metric.update(Array(input_buffs.last), Array(output_buff_m(output_names(0))))
                }    
            
                if(this.monitor !=null){
                    this.monitor.backward_end(i,args_t,update_dict,this.metric)
                }
                
                exe.outputs(0).waitToRead()
            }
            
//        
    }
    
     def solve_0(xpu:Context,sym:Symbol,arg:ListBuffer[(String,NDArray)],
            args_grad:ListBuffer[(String,NDArray)],auxs:ListBuffer[(String,NDArray)],
            data_iter:DataIter,begin_iter:Int=0,end_iter:Int,debug:Boolean=false){
//            if(this.monitor !=null){
//                    this.monitor.backward_end(0,arg.toMap,args_grad.toMap,this.metric)
//                }
       
//            val input_desc:Map[String,Shape] = data_iter.provideData ++ data_iter.provideLabel
            val input_desc:Map[String,Shape] = data_iter.provideData 
            val input_names = input_desc.keys
            val input_buffs = input_desc.map(x => NDArray.empty(x._2, xpu))
            
            val args_t = arg.toMap ++ input_names.zip(input_buffs).toMap
            val output_names = sym.listOutputs()
            if(debug){
                logger.info("need to code in details")
            }
                        
            val exe = sym.easy_bind(xpu,args=args_t,argsGrad = args_grad.toMap,auxStates = auxs.toMap)
            
//            println("----------------------------")
//            println(sym.debugStr)
//            println(exe.debugStr)
            
            require(sym.listArguments().length==exe.gradArrays.length,"dismatch error in solve Solver.scala ")
            var update_dict = sym.listArguments().zip(exe.gradArrays).toMap
            
            update_dict = update_dict.-(sym.listArguments()(0))
//            sym.listArguments().foreach(println)
//            println(update_dict.length)
//            
            val batch_size = input_buffs.head.shape(0)
            this.optimizer.setRescaleGrad(1.0f/batch_size)
            
            /**
             * output_dict :output info (String,NDArray
             * output_buff : the new buffer refered to output_dict
             * internal_dict: internal nodes ,not the output
             */
            
            var output_dict = ListBuffer[(String,NDArray)]()
            var output_buff = ListBuffer[(String,NDArray)]()
            var internal_dict = input_names.zip(input_buffs).toMap
            for((key,arr)<-sym.listOutputs().zip(exe.outputs)){
                if(output_names.contains(key)){
                    output_dict :+= (key,arr)
                    output_buff :+= (key,NDArray.empty(arr.shape,Context.defaultCtx))
                }else{
                    internal_dict += (key->arr)
                 }
            }
            val output_buff_m = output_buff.toMap
            
            data_iter.reset()
            
            /**
             * training start....
             * 
             */
            for(i<- begin_iter until end_iter){
//            	println(s"------------------------$i-----")
                val batch = data_iter.next()
                
                /**
                 * update the input training data
                 */
                for((data,buff)<-batch.data.zip(input_buffs)){
                    data.copyTo(buff)
                }    
                exe.forward(isTrain=true)
                
                /**
                 * internal node info: internal_dict
                 */
                if(this.monitor!=null){
                    this.monitor.forward_end(i, internal_dict)
                }
                
                /***
                 * backup the output info
                 */
                for(key<-output_dict){
                    key._2.copyTo(output_buff_m(key._1))
                }
                
                
                exe.backward()
//                println(s"------------------------$i-----")
                
//                println(sym.debugStr)
//                println(exe.debugStr)
                
                updateParams(args_t,update_dict,this.updater)
//                 println(s"------------------------$i-----")
                
                if(this.metric!=null){
//                	println(input_buffs.last.shape)
//                	println(output_buff_m(output_names(0)).shape)
                    this.metric.update(Array(input_buffs.last), Array(output_buff_m(output_names(0))))
                }    
            
                if(this.monitor !=null){
                    this.monitor.backward_end(i,args_t,update_dict,this.metric)
                }
                
                exe.outputs(0).waitToRead()
            }
    }
    
      // Perform update of param_arrays from grad_arrays not on kvstore
  private def updateParams(paramMap: Map[String, NDArray],
                           gradMap: Map[String, NDArray],
                           updater: MXKVStoreUpdater,
                           numDevice: Int=1) {
      var idx = 0
      for(key<-gradMap.keys){
          if(paramMap(key)!=null ){
        	 if(!key.equals("data") && !key.equals("input")){
        	 	  updater.update(numDevice+idx, gradMap(key), paramMap(key))
	              idx +=1
        	 }
          }else{
              throw new java.lang.UnknownError("dismatch error!!!")
          }
      }
      
    }
    
}


/**
 * a class to monitor the process
 * @param interval: interval for each print
 */
class Monitor(val interval:Int){
    private val logger = LoggerFactory.getLogger(classOf[Monitor])
    def stat(x:NDArray):Float = {
        NDArray.mean(NDArray.abs(x)).toScalar
    }
    
    def forward_end(i:Int,internals:Map[String,NDArray]){
        if(i%this.interval==0){
            for(key<- internals.keys){
                val arr = internals(key)
                val mean = this.stat(arr)
                logger.info(s"Iter:$i  param:$key \t\t stat(mean):$mean")
                System.err.println(s"Iter:$i  param:$key \t\t stat(mean):$mean")
            }
        }
    }
    
    def backward_end(i:Int,args:Map[String,NDArray],grads:Map[String,NDArray],metric:EvalMetric){
        if(i%this.interval==0){
            for(key<- grads.keys){
                val arr = grads(key)
                val mean_args = this.stat(args(key))
                val mean_grad = this.stat(arr)
                System.err.println(s"Iter:$i  param:$key \t\t stat(mean):$mean_args \t\t grad_stat:$mean_grad")
            }
        }
        if(i%this.interval==0 && metric !=null){
        	val metricValue = (metric.get._2)
            System.err.println(s"Iter:$i \tmetric:$metricValue")
            metric.reset()
        }
    }
}

























