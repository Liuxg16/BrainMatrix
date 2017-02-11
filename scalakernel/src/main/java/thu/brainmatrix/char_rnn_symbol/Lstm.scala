package thu.brainmatrix.char_rnn_symbol
import thu.brainmatrix.Symbol
import thu.brainmatrix.Shape
import Config._
object Lstm {
  
    /**
     * @author liuxianggen
     * @date 20160718
     * @brief 
     * @param n_layer
     * @param seq_len:the length of sequence
     * @param n_layer
     * @param n_layer
     * @return
     * @example
     * @note
     */
    def LSTM(n_layer:Int,seq_len:Int,dim_hidden:Int,dim_embed:Int,
             n_alphabet:Int,dropout:Float=0, output_states:Boolean=false):Symbol = {
        val embed_W = Symbol.CreateVariable("_embed_weight")
        val pred_W = Symbol.CreateVariable("_pred_weight")
        val pred_b = Symbol.CreateVariable("_pred_bias")
        var layer_param_states_ = for(i<- 0 until n_layer) yield{
            val param = new LSTMParam(Symbol.CreateVariable(s"_l${i}_i2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_h2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_i2h_bias"),
                    Symbol.CreateVariable(s"_l${i}_h2h_bias"))
            val state = new LSTMState(Symbol.CreateVariable(s"_l${i}_init_c"),
                    Symbol.CreateVariable(s"_l${i}_init_h"))
            (param,state)
        }
        val layer_param_states  = layer_param_states_.toArray
        var data = Symbol.CreateVariable("data")
        var label = Symbol.CreateVariable("label")
//        val embed = Symbol.Embedding("embed")(Map("data" -> data, "input_dim" -> inputSize,
//                                           "weight" -> embedWeight, "output_dim" -> numEmbed))
        val inputs = Symbol.SliceChannel()(Array(data),Map("num_outputs" -> seq_len, "squeeze_axis" -> true))
        var dpRatio = 0f
        var hiddens = Array[Symbol]()
        var hidden: Symbol = null
        var input :Symbol = null
        for(t<-0 until seq_len){
        	input = inputs.get(t)
        	hidden = Symbol.FullyConnected(s"fully_$t")(Map("data"->input,"weight"->embed_W,"no_bias"->"true","num_hidden"->dim_embed))    
            
            //stack LSTM cells
            for(i<-0 until n_layer){
            	if (i == 0) dpRatio = 0f else dpRatio = dropout
                val (l_param,l_state) = layer_param_states(i)
                //val dp = if(i==1) 1 else dropout //not do dropout for input layer
                val next_state = lstmCell(s"_lstm_$t",hidden, l_state,l_param, num_hidden=dim_hidden, dropout=dpRatio)
                hidden = next_state.h
                layer_param_states(i) = (l_param,next_state)
            }
        	if (dropout > 0f) hidden = Symbol.Dropout()(Map("data" -> hidden, "p" -> dropout))
            hiddens = hiddens :+ hidden
        }
        val hiddenConcat = Symbol.Concat()(hiddens, Map("dim" -> 0))
        val pred = Symbol.FullyConnected("pred")(Map("data"->hiddenConcat,"weight"->pred_W,"bias"->pred_b,"num_hidden"->n_alphabet))
        val label1 = Symbol.Reshape()(Map("data" -> label, "target_shape" -> "(0,)"))
        val smax = Symbol.SoftmaxOutput("softmax")(Map("data" -> pred,"label"->label1))    
        smax
    }
    
    //test
     def LSTMNet(n_layer:Int,seq_len:Int,dim_hidden:Int,dim_embed:Int,
             n_alphabet:Int,dropout:Float=0, output_states:Boolean=false):Symbol = {
        
    	val embed_W = Symbol.CreateVariable("_embed_weight")
        val pred_W = Symbol.CreateVariable("_pred_weight")
        val pred_b = Symbol.CreateVariable("_pred_bias")
        var layer_param_states_ = for(i<- 0 until n_layer) yield{
            val param = new LSTMParam(Symbol.CreateVariable(s"_l${i}_i2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_h2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_i2h_bias"),
                    Symbol.CreateVariable(s"_l${i}_h2h_bias"))
            val state = new LSTMState(Symbol.CreateVariable(s"_l${i}_init_c"),
                    Symbol.CreateVariable(s"_l${i}_init_h"))
            (param,state)
        }
        val layer_param_states  = layer_param_states_.toArray
        
        
        var outputs = Array[Symbol]()
        for(t<-0 until seq_len){
        	
            var data_i = Symbol.CreateVariable(s"data_$t")
            var label_i = Symbol.CreateVariable(s"label_$t")
            var hidden = Symbol.FullyConnected(s"fully_$t")(Map("data"->data_i,"weight"->embed_W,"no_bias"->"true","num_hidden"->dim_embed))
            //stack LSTM cells
            for(i<-0 until n_layer){
                val (l_param,l_state) = layer_param_states(i)
                //val dp = if(i==1) 1 else dropout //not do dropout for input layer
                val next_state = lstmCell(s"_lstm_$t",hidden, l_state,l_param, num_hidden=dim_hidden, dropout=dropout)
                hidden = next_state.h
                layer_param_states(i) = (l_param,next_state)
                
            }
  
            val pred = Symbol.FullyConnected(s"_pred_$t")(Map("data"->hidden,"weight"->pred_W,"bias"->pred_b,"num_hidden"->n_alphabet))
            val smax = Symbol.SoftmaxOutput(s"_softmax_$t")(Map("data" -> pred,"label"->label_i))    
            
            outputs = outputs :+ smax
        }
        Symbol.Group(outputs: _*)
     }
    
    def lstmGenerator(n_layer:Int,seq_len:Int,dim_hidden:Int,dim_embed:Int,
             n_alphabet:Int,dropout:Float=0, output_states:Boolean=false):Symbol = {
     
        var layer_param_states_ = for(i<- 0 until n_layer) yield{
            val param = new LSTMParam(Symbol.CreateVariable(s"_l${i}_i2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_h2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_i2h_bias"),
                    Symbol.CreateVariable(s"_l${i}_h2h_bias"))
            val state = new LSTMState(Symbol.CreateVariable(s"_l${i}_init_c"),
                    Symbol.CreateVariable(s"_l${i}_init_h"))
            (param,state)
        }
        
        val layer_param_states  = layer_param_states_.toArray
        
        var outputs = Array[Symbol]()
        for(t<-0 until seq_len){
            var data_i = Symbol.CreateVariable(s"data_$t")
            var hidden = Symbol.FullyConnected(s"fully_$t")(Map("data"->data_i,"num_hidden"->dim_embed))
            //stack LSTM cells
            for(i<-0 until n_layer){
                val (l_param,l_state) = layer_param_states(i)
                val next_state = lstmCell(s"_lstm_$t",hidden, l_state,l_param, num_hidden=dim_hidden, dropout=dropout)
                hidden = next_state.h
                layer_param_states(i) = (l_param,next_state)
            }
            if(dropout>0)                
                hidden = Symbol.Dropout()(Map("data"->hidden,"p"->dropout))
            val pred = Symbol.FullyConnected(s"_pred_$t")(Map("data"->hidden,"num_hidden"->n_alphabet))
            val smax = Symbol.SoftmaxOutput(s"_softmax_$t")(Map("data" -> pred))    
           
            outputs = outputs :+ smax
        }
        Symbol.Group(outputs: _*)
    }
        
     
     
     
    def lstmCell(name:String,input:Symbol, prev_state:LSTMState, param:LSTMParam,
                   num_hidden:Int=512, dropout:Float=0):LSTMState = {
        
        var x = {
      	if (dropout > 0f) Symbol.Dropout()(Map("data" -> input, "p" -> dropout))
      		else input
    	}
        val i2h = Symbol.FullyConnected(s"${name}_i2h")(Map("data"->x,"weight"->param.i2h_W,"num_hidden"->num_hidden*4,"bias"->param.i2h_b))
        val h2h = Symbol.FullyConnected(s"${name}_h2h")(Map("data"->prev_state.h,"weight"->param.h2h_W,"num_hidden"->num_hidden*4,"bias"->param.h2h_b))
        val gates = Symbol.SliceChannel(s"${name}_gates")(Array(i2h+h2h),Map("num_outputs"->4))
        val in_gate = Symbol.Activation()(Map("data"->gates.get(0),"name" -> "sig_in_gate", "act_type" -> "sigmoid"))
        val in_trans = Symbol.Activation()(Map("data"->gates.get(1),"name" -> "sig_in_trans", "act_type" -> "tanh"))
        val forget_gate = Symbol.Activation()(Map("data"->gates.get(2),"name" -> "sig_f_gate", "act_type" -> "sigmoid"))
        val out_gate = Symbol.Activation()(Map("data"->gates.get(3),"name" -> "sig_out_gate", "act_type" -> "sigmoid"))
        
        val next_c =  (forget_gate * prev_state.c) + (in_gate * in_trans)
        val next_h =  out_gate * Symbol.Activation()(Map("data"->next_c, "act_type"->"tanh"))
//        val (a,b,_) = next_c.inferShape(Map("data_1"-> Vector(32,24),"_l0_init_h"->Vector(32,64),"_l0_init_c"->Vector(32,64)))
//        val (a1,b1,_) = (next_c).inferShape(Map("data_1"-> Vector(32,24),"_l0_init_c"->Vector(32,64),"_l0_init_h"->Vector(32,64)))
        new LSTMState(next_c,next_h)
    }
    
    
    /**
     * @author liuxianggen
     * @date 20160718
     * @brief 
     * @param n_layer
     * @param seq_len:the length of sequence
     * @param n_layer
     * @param n_layer
     * @return
     * @example
     * @note
     */
    def LSTM_forward(n_layer:Int,seq_len:Int,dim_hidden:Int,dim_embed:Int,
             n_alphabet:Int,dropout:Float=0, output_states:Boolean=false):Symbol = {
        val embed_W = Symbol.CreateVariable("_embed_weight")
        val pred_W = Symbol.CreateVariable("_pred_weight")
        val pred_b = Symbol.CreateVariable("_pred_bias")
        var layer_param_states_ = for(i<- 0 until n_layer) yield{
            val param = new LSTMParam(Symbol.CreateVariable(s"_l${i}_i2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_h2h_weight"),
                    Symbol.CreateVariable(s"_l${i}_i2h_bias"),
                    Symbol.CreateVariable(s"_l${i}_h2h_bias"))
            val state = new LSTMState(Symbol.CreateVariable(s"_l${i}_init_c"),
                    Symbol.CreateVariable(s"_l${i}_init_h"))
            (param,state)
        }
        val layer_param_states  = layer_param_states_.toArray
        var data = Symbol.CreateVariable("data")
        var label = Symbol.CreateVariable("label")
//        val embed = Symbol.Embedding("embed")(Map("data" -> data, "input_dim" -> inputSize,
//                                           "weight" -> embedWeight, "output_dim" -> numEmbed))
        var dpRatio = 0f
        var hiddens = Array[Symbol]()
        var hidden = Symbol.FullyConnected(s"fully_0")(Map("data"->data,"weight"->embed_W,"no_bias"->"true","num_hidden"->dim_embed))    
            
            //stack LSTM cells
            for(i<-0 until n_layer){
            	if (i == 0) dpRatio = 0f else dpRatio = dropout
                val (l_param,l_state) = layer_param_states(i)
                //val dp = if(i==1) 1 else dropout //not do dropout for input layer
                val next_state = lstmCell("_lstm_0",hidden, l_state,l_param, num_hidden=dim_hidden, dropout=dpRatio)
                hidden = next_state.h
                layer_param_states(i) = (l_param,next_state)
            }
        	if (dropout > 0f) hidden = Symbol.Dropout()(Map("data" -> hidden, "p" -> dropout))
            hiddens = hiddens :+ hidden
        val hiddenConcat = Symbol.Concat()(hiddens, Map("dim" -> 0))
        val pred = Symbol.FullyConnected("pred")(Map("data"->hiddenConcat,"weight"->pred_W,"bias"->pred_b,"num_hidden"->n_alphabet))
        val label1 = Symbol.Reshape()(Map("data" -> label, "target_shape" -> "(0,)"))
        val smax = Symbol.SoftmaxOutput("softmax")(Map("data" -> pred,"label"->label1))    
        smax
    }
    
    
    
    
    
    
    def main(args:Array[String]){
       
//          var layer_param_states = for(i<- 0 until 2) yield{
//            val param = new LSTMParam(Symbol.CreateVariable(s"_l${i}_i2h_weight"),
//                    Symbol.CreateVariable(s"_l${i}_h2h_weight"),
//                    Symbol.CreateVariable(s"_l${i}_i2h_bias"),
//                    Symbol.CreateVariable(s"_l${i}_h2h_bias"))
//            val state = new LSTMState(Symbol.CreateVariable(s"_l${i}_init_c"),
//                    Symbol.CreateVariable(s"_l${i}_init_h"))
//            (param,state)
//        }
//          
        val train_data_shape_map = (0 until SEQ_LENGTH).map(x => {
            (s"data_$x",Shape(BATCH_SIZE,1)) 
        }).toMap
        val label_data_shape_map = (0 until SEQ_LENGTH).map(x => {
            (s"label_$x",Shape(BATCH_SIZE)) 
        }).toMap
        val init_state_map = Map("_l0_init_h"->Shape(16,64),"_l0_init_c"->Shape(16,64),"_l1_init_h"->Shape(16,64),"_l1_init_c"->Shape(16,64))
        val input_shape =  train_data_shape_map ++  label_data_shape_map ++  init_state_map    

       
//        
        val lstm = Lstm.LSTMNet(LSTM_N_LAYER, SEQ_LENGTH, DIM_HIDDEN, DIM_EMBED, 87, DROPOUT)
        lstm.listArguments().foreach(println)
        val (a,b,_) = lstm.inferShape(input_shape)
        println("**********LSTM**************")
//        b.foreach {println}
//        var data_i = Symbol.CreateVariable(s"data_1")
//        var label_i = Symbol.CreateVariable(s"label_1")
//        val h = Symbol.CreateVariable("h")
//        
//        
//        var i2h_W = Symbol.CreateVariable("i2h_W")
//        var h2h_W = Symbol.CreateVariable("h2h_W")
//        var h2h_b = Symbol.CreateVariable("h2h_b")
//        var i2h_b = Symbol.CreateVariable("i2h_b")
//        
//        var x = data_i
//        val name = "1"
//        val num_hidden  = 64
//        val i2h = Symbol.FullyConnected(s"${name}_i2h")(Map("data"->x,"weight"->i2h_W,"num_hidden"->num_hidden*4,"bias"->i2h_b))
//        val h2h = Symbol.FullyConnected(s"${name}_h2h")(Map("data"->h,"weight"->h2h_W,"num_hidden"->num_hidden*4,"bias"->h2h_b))
////        val (a,b,_) = i2h.inferShape(Map("data_12"-> Vector(32,24)))
////        b.foreach(println)
//        val gates = Symbol.SliceChannel(s"${name}_gates")(Array(i2h+h2h),Map("num_outputs"->4))
//        val in_gate = Symbol.Activation()(Map("data"->gates.get(0),"name" -> "sig_in_gate", "act_type" -> "sigmoid"))
//        val in_trans = Symbol.Activation()(Map("data"->gates.get(1),"name" -> "sig_in_trans", "act_type" -> "tanh"))
//        val forget_gate = Symbol.Activation()(Map("data"->gates.get(2),"name" -> "sig_f_gate", "act_type" -> "sigmoid"))
//        val out_gate = Symbol.Activation()(Map("data"->gates.get(3),"name" -> "sig_out_gate", "act_type" -> "sigmoid"))
//        val fg = Symbol.FullyConnected(s"${name}_h22h")(Map("data"->h2h,"num_hidden"->64))
//        val c = Symbol.CreateVariable("c")
//        val next_c =  (forget_gate *c )
//        println(next_c.debug())
//        val (a,b,_) = (next_c).inferShape(Map("data_1"-> Vector(32,24),"c"->Vector(32,64),"h"->Vector(32,64)))
//        b.foreach {println}
//        
//        val c = lstmCell(s"_lstm_1",data_i, l_state,l_param, num_hidden=64, dropout=0)
        
        
    }
}

class LSTMState(val c:Symbol,val h:Symbol)
class LSTMParam(val i2h_W:Symbol,val h2h_W:Symbol,val i2h_b:Symbol,val h2h_b:Symbol)
class LSTMFEDState(val input:Symbol)
