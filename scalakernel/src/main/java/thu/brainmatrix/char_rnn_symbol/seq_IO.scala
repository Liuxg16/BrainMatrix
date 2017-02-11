package thu.brainmatrix.char_rnn_symbol
import thu.brainmatrix.NDArray
import thu.brainmatrix.Context
import thu.brainmatrix.io.NDArrayLSTMIter
import thu.brainmatrix.Shape
import scala.io.Source
import scala.math
import java.io.File
import java.io.PrintWriter

object seq_IO {
    /**
     * @author liuxianggen
     * @date 20160718
     * @brief there is the encoder of INPUT_FILE,make each char have a id,
     * 		　　which increase as the frequency decrease. For example：
     * 	　　　　　　input file:
     * 			I love you
     * 		　　vocab:O->1,I->2,l->3...
     * @param　inputFileName
     * @return: vocab_final a map which the max length is 10000
     * @example
     * @note
     */
    def build_vocabulary(inputFileName:String,vocabFileName:String,max_vocab:Int=10000):Map[Char,Int] = {
        val vocabfile = new File(vocabFileName)
        var vocab_final = Map[Char,Int]() 
        if(vocabfile.isFile()){
//            println(s"INFO:Using $vocabFileName,while vocabulary already exists")
            val source  = Source.fromFile(vocabfile)
            val line = source.mkString
            (line.zipWithIndex).map(x=>{
                vocab_final = vocab_final ++ Map(x._1->(x._2))
            })
        }else{
            //if the vocabFile is not existed, now we generate one
            var dict = Map[Char,Int]()
            val source  = Source.fromFile(inputFileName)
            val lineIter = source.mkString
            lineIter.map(w => {
                     dict = dict.updated(w, dict.getOrElse(w,0)+1)
                })
            
            var vocab = dict.toList sortBy(_._2)
//            println("------------------")
//            println(vocab)
            vocab = vocab.take(math.min(max_vocab,vocab.length)).reverse //with decreased order 
            //write to the vocabfile
            val out = new PrintWriter(vocabfile)
            vocab.map(x => {
                out.print(x._1)
            })
            out.close()
            
            (vocab.zipWithIndex) map(x=>{
                vocab_final = vocab_final ++ Map(x._1._1->(x._2))
            })
        }
        vocab_final = vocab_final ++ Map(Config.UNKNOW_CHAR->vocab_final.size)
        vocab_final   
    }
    
    def char_idx(vocab:Map[Char,Int],c:Char){
        if(vocab.contains(c))
            vocab.get(c)
        else {
          vocab.get(Config.UNKNOW_CHAR)
        }
    }
    
    def Str2Char_NDArrayIterator(text:String,labelName:String = "label",vocab:Map[Char,Int],batch_size:Int,seq_len:Int,ctx:Context = Context.defaultCtx):NDArrayLSTMIter = {
        //culculate the number of sequence after delete the first char
        val num_seq_len = math.floor((text.length()-1)/seq_len).toInt
        //map to index of the char
        var array_train = text.map {vocab(_).toFloat}.toArray
        var array_label = array_train.drop(1).take(num_seq_len*seq_len)
        array_train = array_train.take(num_seq_len*seq_len)
        val NDA_train = NDArray.array(array_train, Shape(num_seq_len,seq_len),ctx)
        val NDA_label = NDArray.array(array_label,Shape(num_seq_len,seq_len),ctx)
        val dataIter = new NDArrayLSTMIter(IndexedSeq(NDA_train),"data",IndexedSeq(NDA_label),labelName, batch_size, false, "discard")//the rest will discard
//        println(s"length:${dataIter}")
//        println(s"provideData:${dataIter.provideData}")//(32,24)
//            println(s"provideData:${dataIter.provideLabel}")//(32,24)
        dataIter
    }
    
    def lstmDataIter(text:String,inputName:String = "data",labelName:String = "label",vocab:Map[Char,Int],batch_size:Int,seq_len:Int,ctx:Context = Context.defaultCtx):NDArrayLSTMIter = {
        //culculate the number of sequence after delete the first char
        val num_seq_len_temp = math.floor((text.length()-1)/seq_len).toInt
        val num_batch = math.floor(num_seq_len_temp/batch_size).toInt 
        val num_seq = num_batch*batch_size
        val num_char = num_batch*batch_size*seq_len
        //map to index of the char
        var array_train = text.map {vocab(_).toFloat}.toArray
        array_train = array_train.take(num_char+1)
        
        
        val map_train = (0 until seq_len).map(x => Array.fill[Float](num_seq)(0f)).toArray
        val map_label = (0 until seq_len).map(x => Array.fill[Float](num_seq)(0f)).toArray
        (0 until num_char).map(x =>{
            val id = x%seq_len
            map_train(id)(x/seq_len) = array_train(x)
            map_label(id)(x/seq_len) = array_train(x+1)
        }    
        )
        
//        val init_state_map = Map("_l0_init_h"->NDArray.zeros(Shape(32,64),ctx),"_l0_init_c"->NDArray.zeros(Shape(32,64),ctx),"_l1_init_h"->NDArray.zeros(Shape(32,64),ctx),"_l1_init_c"->NDArray.zeros(Shape(32,64),ctx))
//        val NDA_train = NDArray.array(array_train, Shape(num_seq_len,seq_len),ctx)
//        val NDA_label = NDArray.array(array_label,Shape(num_seq_len,seq_len),ctx)
        val dataIter = new NDArrayLSTMIter(map_train.map(NDArray.array(_,Shape(num_seq,1))).toIndexedSeq,inputName,map_label.map(NDArray.array(_,Shape(num_seq))).toIndexedSeq,labelName, batch_size, false, "discard")//the rest will discard
//        println(s"provideData:${dataIter.provideLabel}")//(32,24)
        dataIter
    }
    
    
    def RNN_OneHot_DataIter(text:String,inputName:String = "data",labelName:String = "label",vocab:Map[Char,Int],batch_size:Int,seq_len:Int,ctx:Context = Context.defaultCtx):NDArrayLSTMIter = {
        //culculate the number of sequence after delete the first char
        val num_seq_len_temp = math.floor((text.length()-1)/seq_len).toInt
        val num_batch = math.floor(num_seq_len_temp/batch_size).toInt 
        val num_seq = num_batch*batch_size
        val num_char = num_batch*batch_size*seq_len
        //map to index of the char
        var array_train = text.map {vocab(_).toFloat}.toArray
        val label_arr = NDArray.array(array_train.take(num_char+1).drop(1),Shape(num_seq,seq_len))
        array_train = array_train.take(num_char)
        
        val tarin_arr = NDArray.zeros(Shape(num_seq,seq_len,vocab.size), ctx)
        (0 until num_char).map(x =>{
            val id = x%seq_len
            tarin_arr(x/seq_len,id,array_train(x).toInt) = 1 
        }    
        )
        val dataIter = new NDArrayLSTMIter(IndexedSeq(tarin_arr),inputName,IndexedSeq(label_arr),labelName, batch_size, false, "discard")//the rest will discard
//        println(s"provideData:${dataIter.provideData}")//(32,24)
        dataIter
    }
    
    
      def lstm_vec_DataIter(text:String,inputName:String = "data",labelName:String = "label",vocab:Map[Char,Int],batch_size:Int,seq_len:Int,vocab_len:Int,ctx:Context = Context.defaultCtx):NDArrayLSTMIter = {
        //culculate the number of sequence after delete the first char
        val num_seq_len_temp = math.floor((text.length()-1)/seq_len).toInt
        val num_batch = math.floor(num_seq_len_temp/batch_size).toInt 
        val num_seq = num_batch*batch_size
        val num_char = num_batch*batch_size*seq_len
        //map to index of the char
        var array_train = text.map {vocab(_)}.toArray
        array_train = array_train.take(num_char+1)
        
        
        val map_train = (0 until seq_len).map(x => NDArray.zeros(Shape(num_seq,vocab_len), ctx)).toArray
        val map_label = (0 until seq_len).map(x => NDArray.zeros(Shape(num_seq), ctx)).toArray
        (0 until num_char).map(x =>{
            val id = x%seq_len
            map_train(id)(x/seq_len,array_train(x)) = 1 
            map_label(id)(x/seq_len) = array_train(x+1)
        }    
        )
        
//        val init_state_map = Map("_l0_init_h"->NDArray.zeros(Shape(32,64),ctx),"_l0_init_c"->NDArray.zeros(Shape(32,64),ctx),"_l1_init_h"->NDArray.zeros(Shape(32,64),ctx),"_l1_init_c"->NDArray.zeros(Shape(32,64),ctx))
//        val NDA_train = NDArray.array(array_train, Shape(num_seq_len,seq_len),ctx)
//        val NDA_label = NDArray.array(array_label,Shape(num_seq_len,seq_len),ctx)
        val dataIter = new NDArrayLSTMIter(map_train.toIndexedSeq,inputName,map_label.toIndexedSeq,labelName, batch_size, false, "discard")//the rest will discard
//        println(s"provideData:${dataIter.provideLabel}")//(32,24)
        dataIter
    }
    
    
    
    
    def SampleDataIter(text:String,inputName:String = "data",labelName:String = "label",vocab:Map[Char,Int],batch_size:Int,seq_len:Int,ctx:Context = Context.defaultCtx):NDArrayLSTMIter = {
        //culculate the number of sequence after delete the first char
        val num_seq_len_temp = math.floor((text.length()-1)/seq_len).toInt
        val num_batch = 1
        val num_seq = num_batch*batch_size
        val num_char = num_batch*batch_size*seq_len
        //map to index of the char
        var array_train = text.map {vocab(_).toFloat}.toArray
        array_train = array_train.take(num_char+1)
        val map_train = (0 until seq_len).map(x => (s"${inputName}_$x",Array.fill[Float](num_seq)(0f))).toMap
        val map_label = (0 until seq_len).map(x => (s"${labelName}_$x",Array.fill[Float](num_seq)(0f))).toMap
        (0 until num_char).map(x =>{
            val id = x%seq_len
            val arr_train = map_train.getOrElse(s"${inputName}_$id",Array.fill[Float](num_seq)(0f))
            val arr_label = map_label.getOrElse(s"${labelName}_$id",Array.fill[Float](num_seq)(0f))
            arr_train(x/seq_len) = array_train(x)
            arr_label(x/seq_len) = array_train(x+1)
        }    
        )
        
//        val init_state_map = Map("_l0_init_h"->NDArray.zeros(Shape(32,64),ctx),"_l0_init_c"->NDArray.zeros(Shape(32,64),ctx),"_l1_init_h"->NDArray.zeros(Shape(32,64),ctx),"_l1_init_c"->NDArray.zeros(Shape(32,64),ctx))
//        val NDA_train = NDArray.array(array_train, Shape(num_seq_len,seq_len),ctx)
//        val NDA_label = NDArray.array(array_label,Shape(num_seq_len,seq_len),ctx)
        val dataIter = new NDArrayLSTMIter(map_train.values.map(NDArray.array(_,Shape(num_seq,1))).toIndexedSeq,inputName,map_label.values.map(NDArray.array(_,Shape(num_seq))).toIndexedSeq,labelName, batch_size, false, "pad")//the rest will discard
//        println(s"provideData:${dataIter.provideData}")//(32,24)
        dataIter
    }
    
    
    
    def main(args:Array[String]){
        //test build_vocabulary
        val vocab = build_vocabulary("./seqData/input.txt","./seqData/vocab.txt")
//        vocab.foreach(println)
//        println(vocab.values)
        
    }
}