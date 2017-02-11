package thu.brainmatrix.ladder
import thu.brainmatrix.Symbol
import thu.brainmatrix.NDArray
import thu.brainmatrix.Base._
import thu.brainmatrix.Context
import java.io.FileNotFoundException
import org.slf4j.LoggerFactory
import scala.collection.mutable.ListBuffer
import thu.brainmatrix.DataIter
class AEModel(val xpu: Context = Context.defaultCtx) {
    var loss:Symbol = null
    
    /**
     * the following four items is array of tuples(key,value),containing 
     * description + value
     */
    var args = ListBuffer[(String,NDArray)]()
    var args_grad = ListBuffer[(String,NDArray)]()
    var args_mult = ListBuffer[(String,Float)]()
    var auxs = ListBuffer[(String,NDArray)]()
    
    def save(fname:String){
      AEModel.logger.info("save model!")
    } 
    
    def load(fname:String){
      AEModel.logger.info("load model!")
    }  
}

object AEModel{
   private val logger = LoggerFactory.getLogger(classOf[AEModel])
   
    def extract_feature(sym:Symbol, args:ListBuffer[(String,NDArray)],auxs:ListBuffer[(String,NDArray)],data_iter:DataIter,xpu:Context =Context.cpu())
       :Map[String,ListBuffer[NDArray]] = {
       val input_buffs = data_iter.provideData.map{
           x => NDArray.empty(x._2,xpu)
       }
       
       val input_names = data_iter.provideData.map(_._1)
       val args_ef = args.toMap ++  input_names.zip(input_buffs).toMap
       val exe = sym.easy_bind(xpu, args = args_ef, auxStates = auxs.toMap)
       
       var output_buffs:Array[NDArray] =null
       var outputs  = Array.fill[ListBuffer[NDArray]](exe.outputs.length)(ListBuffer[NDArray]())
       data_iter.reset()
       var dataBatch = data_iter.next()
        
       while (dataBatch != null) {
            for ((data,buff)<- dataBatch.data.zip(input_buffs)){
                data.copyTo(buff)
            }
            exe.forward(isTrain=false)
            if(output_buffs==null){
                output_buffs = exe.outputs.map(x => {
                    NDArray.empty(x.shape, ctx=Context.defaultCtx)
                })
            }else{
                for((out,buff)<-outputs.zip(output_buffs)){
                    out.append(buff)
                }
            }
            for((out,buff)<-exe.outputs.zip(output_buffs)){
                out.copyTo(buff)
            } 
            if(data_iter.hasNext)
            	dataBatch = data_iter.next()
            else {
              dataBatch = null
            }
       }
       for((out,buff)<-outputs.zip(output_buffs)){
                    out.append(buff)
                }
       
      sym.listOutputs().zip(outputs).toMap
    }
    
    def main(args:Array[String]){
      AEModel.logger.warn("FileNotFoundException ?")
      throw new FileNotFoundException("FileNotFoundException!")
      
      println("test!")
    }
}

//
//# pylint: skip-file
//import mxnet as mx
//import numpy as np
//import logging
//from solver import Solver, Monitor
//try:
//   import cPickle as pickle
//except:
//   import pickle
//
//
//def extract_feature(sym, args, auxs, data_iter, N, xpu=mx.cpu()):
//    input_buffs = [mx.nd.empty(shape, ctx=xpu) for k, shape in data_iter.provide_data]
//    input_names = [k for k, shape in data_iter.provide_data]
//    args = dict(args, **dict(zip(input_names, input_buffs)))
//    exe = sym.bind(xpu, args=args, aux_states=auxs)
//    outputs = [[] for i in exe.outputs]
//    output_buffs = None
//
//    data_iter.hard_reset()
//    for batch in data_iter:
//        for data, buff in zip(batch.data, input_buffs):
//            data.copyto(buff)
//        exe.forward(is_train=False)
//        if output_buffs is None:
//            output_buffs = [mx.nd.empty(i.shape, ctx=mx.cpu()) for i in exe.outputs]
//        else:
//            for out, buff in zip(outputs, output_buffs):
//                out.append(buff.asnumpy())
//        for out, buff in zip(exe.outputs, output_buffs):
//            out.copyto(buff)
//    for out, buff in zip(outputs, output_buffs):
//        out.append(buff.asnumpy())
//    outputs = [np.concatenate(i, axis=0)[:N] for i in outputs]
//    return dict(zip(sym.list_outputs(), outputs))
//
//class MXModel(object):
//    def __init__(self, xpu=mx.cpu(), *args, **kwargs):
//        self.xpu = xpu
//        self.loss = None
//        self.args = {}
//        self.args_grad = {}
//        self.args_mult = {}
//        self.auxs = {}
//        self.setup(*args, **kwargs)
//
//    def save(self, fname):
//        args_save = {key: v.asnumpy() for key, v in self.args.items()}
//        with open(fname, 'w') as fout:
//            pickle.dump(args_save, fout)
//
//    def load(self, fname):
//        with open(fname) as fin:
//            args_save = pickle.load(fin)
//            for key, v in args_save.items():
//                if key in self.args:
//                    self.args[key][:] = v
//
//    def setup(self, *args, **kwargs):
//        raise NotImplementedError("must override this")
