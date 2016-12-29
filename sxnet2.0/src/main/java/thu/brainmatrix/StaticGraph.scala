package thu.brainmatrix

import thu.brainmatrix.Base._
import scala.collection.mutable.{ArrayBuffer,ListBuffer}
import scala.collection.mutable.LinkedHashMap
import scala.Vector


/**
 * 2016-3-14
 * by liuxianggen
 * its function is the same as in mxnet c++ part
 * brief a struct needing to be converted to mxnet c++ part
 * 
 * note:
 * 1.need to add finalize function
 *
 */
class StaticGraph(){
	
	private var disposed = false
	override protected def finalize(): Unit = {
    	dispose()
  }

  /**
   * Release the native memory.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      _LIB.mxStaticGraphFree(handle)
      disposed = true
    }
  }
  var arg_nodes:Vector[Int]  = Vector()
  var heads:Vector[DataEntry] = Vector()
  var nodes:Vector[Node] = Vector()
  var handle: StaticGraphHandle = _
  
  def reset{
  	this.arg_nodes = Vector()
  	this.heads = Vector()
  	this.nodes = Vector()
  	this.handle = 0
  }
  
  def debug:String = {
  	var s = "-----------StaticGraph debug information ----------------------------\n"
  	s += "arg_nodes:\n"
//  	s= "length:" + this.arg_nodes.length
  	this.arg_nodes.foreach { n => s += " " + n }
  	s += "\nheads:\n"
  	this.heads.foreach { x => s += x.Info }
  	s += "\nnodes:\n"
  	this.nodes.foreach{
  		x => {
  		val sourceid = x.inputs.map(_.source_id.toString()+" ")
  		s += "\nname:" + x.name+"\n\t" +"is_backward:" + x.backward_source_id+ "\n\tinputs_e_source_id" +sourceid.foldLeft(" ")(_+_)
  			}
  	}

  	s += "\n-----------StaticGraph debug information ----------------------------\n"
  	s
  	
  	
  }
  
  /**
   * 2016-3-25
   * by liuxianggen 
   */
  def ToStaticGraph:Int = {
//  	println("-----------------------StaticGraph　Info--------------------------------")
  	/**
  	 * MXNET:
  	 * DataEntry:source_id, index
  	 * 
  	 *   struct Node {
				    /*! \brief wrapped operator property */
				    std::unique_ptr<OperatorProperty> op;
				    /*! \brief name of the node */
				    std::string name;
				    /*! \brief inputs (node_id, index) for of the nodes*/
				    std::vector<DataEntry> inputs;
				    /*!
				     * \brief If this field is nonnegative, this indicates this
				     *  Node is corresponds to a Backward Operation of Operator.
				     *  backward_source_id will points to the corresponding Forward Node.
				     *
				     *  For normal node, this field is -1.
				     *  When the node is a Backward node, the op field will be nullptr
				     */
				    int32_t backward_source_id;
				    /*! \brief additional attributes about the node */
				    std::map<std::string, std::string> attr;
  	 * 
  	 */
  	
  	//for arg_nodes
  	val arg_node_sg:Array[Int] = this.arg_nodes.toArray//
  	
  	
  	//for heads
  	val heads1:Vector[(Int,Int)] = heads.map { x => (x.source_id,x.index)}
  	val (heads_source_V:Vector[Int],heads_index_V:Vector[Int]) = heads1.unzip
  	val heads_source :Array[Int]= heads_source_V.toArray//
  	val heads_index :Array[Int]= heads_index_V.toArray//
	

  	
  	//for nodes
  	val nods3:Vector[(OperatorPropertyRef,String,Vector[DataEntry])] = nodes.map{x => (x.opRef, x.name, x.inputs)}
  	val nods45 = nodes.map { x => (x.backward_source_id,x.attr ) }
  	
  	val (nods_opRef,nods_name_V,nods_inputs):(Vector[OperatorPropertyRef],Vector[String],Vector[Vector[DataEntry]])= nods3.unzip3//
//  	val nods_opHandles :Array[OperatorPropertyHandle]= nods_opRef.map({_.value.handle}).toArray
  	val OperatorPropertyHandleref = new OperatorPropertyHandleRef
  	var nods_opHandles_V :Vector[OperatorPropertyHandle]= nods_opRef.map( x => {
  		if(x.value==null)
//  			x.value.handle
  			OperatorPropertyHandleref.value
  		else
  			x.value.handle
  			})
  	nods_opHandles_V :+= OperatorPropertyHandleref.value
  	val nods_opHandles = nods_opHandles_V.toArray//
  	
  	val nods_name :Array[String]= nods_name_V.toArray//
//  	println("nods_name:")
//  	println(nods_name.length)
//  	nods_name.foreach {println}
  	
  	/**
  	 * 
  	 * for nods_inputs, actually, it's a Vector[Vector[DataEntry]]
  	 * so complicated for convert to c++ by JNI,
  	 * make it to two matrixes, like matrix1(source_id) and matrix2(index)
  	 * nods_inputs(i) = matrix1(i,:),matrix1(i,:)
  	 * 
  	 */
  	
//  	println("-------------------------------------------------------")
//  	println("inputs:")
  	val nods_inputs_len_arr :Array[Int] = nods_inputs.map { _ .length}.toArray// 
//  	nods_inputs.foreach(x => {	
//  		print("len:"+x.length+" ")
//  		x.foreach(y => print("\nindex:"+y.index + " source_id:"+y.source_id))
//  		println
//  		})
  	
 	 	val nods_inputsM = nods_inputs.flatten
 	 	val nods_inputs_source_ids:Array[Int] = nods_inputsM.map { _.source_id}.toArray//
  	val nods_inputs_indexs:Array[Int] = nods_inputsM.map { _.index }.toArray//
  	
  	/**
  	 * nods_atts:Array[Map[String,String]]
  	 */
  	val (nods_backward_source_ids_V:Vector[Int],nods_attrs) = nods45.unzip//
  	val nods_backward_source_ids = nods_backward_source_ids_V.toArray
  	val nods_attr_len_arr:Array[Int] = nods_attrs.map( _.size).toArray//
  	val nods_attr_len_arr_len = nods_attr_len_arr.foldLeft(0)(_ + _)
  	val nods_attrs_keys:Array[String] = (nods_attrs.map(x => { x.keys}).flatten).toArray//
  	val nods_attrs_values:Array[String] = (nods_attrs.map(x => { x.values}).flatten).toArray//
  	
  	
  	/**
  	 * 
  	 * (Array[Int],Array[Int]，Array[Int]，Array[OperatorPropertyHandle]，Int,Array[String],
  	 * Array[Int] ,Array[Int]，Array[Int]，Array[Int]，Array[Int]，Array[String]，Array[String])
  	 * 
  	 */
  	 val handleref:StaticGraphHandleRef = new StaticGraphHandleRef
  	 val ret = _LIB.mxScalaToStaticGraph(handleref,arg_node_sg,heads_source,heads_index,nods_opHandles,nods_name.length,nods_name,nods_inputs_len_arr,nods_inputs_source_ids,nods_inputs_indexs,
  			nods_backward_source_ids,nods_attr_len_arr,nods_attr_len_arr_len,nods_attrs_keys,nods_attrs_values)
  	 this.handle = handleref.value
//  	println("-----------------------StaticGraph　Info--------------------------------")
  	ret
  }
  
    /**
     * @author liuxianggen
     * @date 20160724
     * @brief check the truth variable and returns the kv:name and its shape, keys_arr: the index of arg_node order
     *        there is something important:the index is the order of arg_node, not the normal node
     *        example:
     *        nodes:1,2,3,4,5,6
     *        args_nodes:1,3,5,6
     *        kwargs:Map("data"->Vector(2,3))  where "data" is the node(3)'s name. however, node(3) is the 2th node in the arg_node
     *        so, return:
     *         kv = Map("data"->Vector(2,3))
     *         key_arr = 2
     * @param
     * @return
     * @example
     * @note
     */  
  def identifyVar(kwargs: Map[String, Shape]):(LinkedHashMap[String, Shape],ArrayBuffer[Int])= {
  	val keys_arr = ArrayBuffer.empty[Int]
  	val kv =scala.collection.mutable.LinkedHashMap[String,Shape]()
  	val varNodeName = this.arg_nodes.map{ this.nodes(_) }.map {_.name}
  	for(i <- 0 until varNodeName.length){
  		if(kwargs.contains(varNodeName(i))){
  			keys_arr += i
  			kv(varNodeName(i)) = kwargs.getOrElse(varNodeName(i),Shape()) 
  		}
  		
  	}
//  	val v = kwargs.filter(kv => {
//  		varNodeName.contains(kv._1)})
//    v
  	(kv,keys_arr)
  }
  

    /**
     * @author liuxianggen
     * @date 20160724
     * @brief transform the kwargs to the data structure which can recognized by jni, the following comments1 works when needed
     * @param
     * @return
     * @example
     * @note
     */
  def inferShape(kwargs:Map[String,Shape],inShapeData: ListBuffer[Array[Int]],outShapeData: ListBuffer[Array[Int]],auxShapeData: ListBuffer[Array[Int]],complete: Base.RefInt){
  	  this.ToStaticGraph
  	  val (kv,keys_arr) = this.identifyVar(kwargs)
      val indPtr = ArrayBuffer(0)
      var sdata = ArrayBuffer.empty[Int]
      kv.foreach { case (key, shape) =>
//        keys += key
          sdata = sdata ++ shape.toVector
          indPtr += sdata.size
      }
//		comments1
//  	println("----------------------parameter--------------------------------")
//  	println(indPtr.size-1)
//  	println(indPtr)
//  	println(keys_arr)
//  	println(sdata)
//  	kv.foreach(println)
//  	println("---------------------------------------------------------------")
  	
  	  _LIB.mxScalaSGInferShape(this.handle, this.arg_nodes.size, indPtr.size - 1,keys_arr.toArray, indPtr.toArray, sdata.toArray, inShapeData, outShapeData, auxShapeData, complete)
  	
  }
  
  
  
  def printOperator{
  	this.nodes.foreach { x => {
  		if(x.opRef.value!=null){
  			println(x.name+" operator name:")
  			println(x.opRef.value.opName)
  			(x.opRef.value.printParam())
  		}
  	} }
  }
  
  
  
  def bind(in_args:Array[NDArray],arg_grad_store:Array[NDArray],grad_req_type:Array[Int],
  		auxNDArrays:Array[NDArray] = new Array[NDArray](0)):ExecutorHandleRef = {
  	val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]

  	val execHandle = new ExecutorHandleRef
  	if(this.handle == 0){
  		System.err.println("bind error! handle == 0")
  	}else{
  		
    	checkCall(_LIB.mxScalaExecutorBindX(this.handle,
                                   1,//1
                                   0,//0
                                   ctxMapKeys.size,//0
                                   ctxMapKeys.toArray,//null
                                   ctxMapDevTypes.toArray,//null
                                   ctxMapDevIDs.toArray,//null
                                   in_args.size,
                                   in_args.map(_.handle),
                                   arg_grad_store.map(_.handle),
                                   grad_req_type,
                                   auxNDArrays.map(_.handle),
//                                   new Array[NDArrayHandle](0),
                                   execHandle))
  	}
  	execHandle
  	
  }
  
  def bind(in_argsh:Array[NDArrayHandle],arg_grad_storeh:Array[NDArrayHandle],
                                   grad_req_type:Array[Int]):Executor = {
  	val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]

  	val execHandle = new ExecutorHandleRef
  	println("---------------------binding-----------------------")
  	if(this.handle == 0){
  		System.err.println("bind error! handle == 0")
  	}else{
//  		in_args.foreach{x => println(x.shape)}
//  		println("---------------------------------------------------------")
//  		arg_grad_store.foreach{x => println(x.shape)}
  		
    	checkCall(_LIB.mxScalaExecutorBindX(this.handle,
                                   1,//1
                                   0,//0
                                   ctxMapKeys.size,//0
                                   ctxMapKeys.toArray,//null
                                   ctxMapDevTypes.toArray,//null
                                   ctxMapDevIDs.toArray,//null
                                   in_argsh.size,
                                   in_argsh,
                                   arg_grad_storeh,
                                   grad_req_type,
                                   new Array[NDArrayHandle](0),
                                   execHandle))
  	}
  	println("---------------------binding succeed!-----------------------")
  	new Executor(execHandle.value,null)
  }
  
  def bind(deviceTypeid:Int,
            deviceID:Int,
            numCtx: Int,
            ctxMapKeys: Array[String],
            ctxMapDevTypes: Array[Int],
            ctxMapDevIDs: Array[Int],
            numArgs: Int,
            argsHandle: Array[NDArrayHandle],
            argsGradHandle: Array[NDArrayHandle],
            reqsArray: Array[Int],
            auxArgsHandle: Array[NDArrayHandle]):ExecutorHandleRef = {

    	val execHandle = new ExecutorHandleRef
    	if(this.handle == 0){
    		throw new java.lang.Error("bind error! handle == 0")
    	}else{
  //  		in_args.foreach{x => println(x.shape)}
  //  		println("---------------------------------------------------------")
  //  		arg_grad_store.foreach{x => println(x.shape)}
    		
      	  checkCall(_LIB.mxScalaExecutorBindX(this.handle,
                                     deviceTypeid,//1
                                     deviceID,//0
                                     numCtx,//0
                                     ctxMapKeys,//null
                                     ctxMapDevTypes,//null
                                     ctxMapDevIDs,//null
                                     numArgs,
                                     argsHandle,
                                     argsGradHandle,
                                     reqsArray,
                                     auxArgsHandle,
                                     execHandle))
    	}
      execHandle
  }
  
  	def saveToFile(fname: String){
  		this.ToStaticGraph
  		checkCall(_LIB.mxScalaSymbolSaveToFile(this.handle,fname))
  	}
  
}