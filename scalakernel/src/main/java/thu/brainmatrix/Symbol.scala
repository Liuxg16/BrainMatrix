package thu.brainmatrix

import thu.brainmatrix.Base._
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.Stack
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Stack
import scala.Vector

/**
 * Symbolic configuration API of brainmatrix. <br />
 * <b>
 * WARNING: it is your responsibility to clear this object through dispose().
 * NEVER rely on the GC strategy
 * </b>
 * @author Yizhi Liu
 */
// scalastyle:off finalize
class Symbol private(private[brainmatrix] val handle: SymbolHandle) {
  private val logger: Logger = LoggerFactory.getLogger(classOf[Symbol])
  private var disposed = false

  override protected def finalize(): Unit = {
		this.staticGraph.dispose()    
  }

  
	
	//global variable for symbol graph
	var heads_ : Vector[DataEntry] = Vector()
	var staticGraph = new StaticGraph()
  /**
   * Release the native memory.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      this.staticGraph.dispose()  
      disposed = true
    }
  }

	// 2015-3-6
	/**
	 * by liuxianggen
	 * depth first search algorithm
	 */
	private[brainmatrix] def DFSVisit(fvisit: (NodeRef) => Unit): Unit = {
		var res: Vector[NodeRef] = Vector()
		var stack: Stack[(NodeRef, Int)] = Stack()
		var visited: Set[NodeRef] = Set()
		heads_.map(head => {
			val ptr = head.source
			if (!visited.contains(ptr)) {
				stack.push((head.source,0))
				visited += (ptr)
			}
		//	stack.foreach(println)
			while (!stack.isEmpty) {
				var back: (NodeRef, Int) = stack.top
//				println("back:"+back._1.value.name)
				//find its inputs whether have visited all
				if (back._2 == (back._1.value.inputs.length)) {
					res = res :+ back._1
					fvisit(back._1)
					stack.pop
				} else {
					//find its inputs whether have visited all(not)
					var inputs: Vector[DataEntry] = back._1.value.inputs
					var input: DataEntry = inputs(back._2)
					stack.update(0, (back._1,back._2+1))
//					back = (back._1,back._2+1) 
					val ptr = input.source
					//add  un-visited node to stack and visited 
					if (!visited.contains(ptr)) {
						stack.push((input.source, 0))
						visited += ptr
					}
				}
			}

		})
	}

	def is_atomic(): Boolean = {
		return heads_(0).source.value.is_atomic
	}

	def NumVisibleOutputs(): Int = {
		1
	}
	
	def NumOutputs():Int = {
	  heads_.length
	}
	
		/**
	 * 2016-3-15
	 * by liuxianggen
	 * find each variable and link them with inputs
	 */
	def Compose(kwargs: Map[String, Symbol], name: String) {
		// the name of this
		heads_(0).source.value.name = name
		var nmatched: Int = 0

		// atomic symbol do not have place holder for all the arguments
		if (this.is_atomic()) {

//			println(heads_(0).source.value.opRef.value.opName)
			val req_args: Vector[String] = heads_(0).source.value.opRef.value.ListArguments
//			println(" && ")
//			req_args.foreach {println}
			(0 until req_args.length).map(i => {
				val iter: Symbol = kwargs.getOrElse(req_args(i), null)
				if (iter != null) {
//					added by liuxianggen,which is none in brainmatrix c++
//					iter.heads_(0).source.value.backward_source_node = heads_(0).source
					heads_(0).source.value.inputs :+= iter.heads_(0)
					
					nmatched += 1
				} else {
					val noderef = new NodeRef()
					val node = new Node(new OperatorPropertyRef, name+"_"+req_args(i))
//					added by liuxianggen,which is none in brainmatrix c++
//					node.backward_source_node = heads_(0).source
					noderef.value = node
					heads_(0).source.value.inputs :+= new DataEntry(noderef, 0)
					if (heads_(0).source.value.attr.size != 0)
						heads_(0).source.value.inputs(i).source.value.attr = heads_(0).source.value.attr
				}
			})
			if (nmatched != kwargs.size)
				heads_(0).source.value.inputs = Vector()
				
		} else {
			
			System.err.println("should not execute in there of compose! ")
			// find all the arguments positions
			var (dup_args, max_dup) = this.FindDuplicateArgs
			if (max_dup > 1) {
				/**
				 * operations for kvstores
				 */
			}
			this.DFSVisit { noderef =>
				{
					/**
					 * this part is for the in-place algorithm
					 * need complete
					 *
					 */
					//        		(0 until noderef.value.inputs.size).map(i =>{
					//        			val e:DataEntry = noderef.value.inputs(i)
					//        			if(e.source.value.is_variable()){
					//        				/*
					//        				 * translate from:
					//        				 *  auto iter = kwargs.find(e->source->name);
					//            		 *  if (iter != kwargs.end()) {...
					//        				 */
					//        				if(kwargs.contains(e.source.value.name)){
					//        					var target = kwargs(e.source.value.name).heads_(0)
					//        				}
					//        			}
					//        		})

				}
			}
		}

	}

	
	/**
	 * by liuxianggen
	 * 2016-7-2
	 * 
	 * for the operations:
	 * arithmetric
	 * 
	 */
	
	def Compose(args: Array[Symbol],name: String) {
	    require(!heads_(0).source.value.is_variable(),"Variable cannot be composed!")  
	    heads_(0).source.value.name = name
	    for(i <- 0 until args.length){
	      require(args(i).NumOutputs()==1,s"Argument $i is a tuple with one more elements,scalar is required")
	    }
	    
	    if(this.is_atomic()){
	      val req_args :Vector[String]= heads_(0).source.value.opRef.value.ListArguments
//	      println("--------------------------")
//	      println(req_args)
//	      println("--------------------------")
	      require(args.length==req_args.length,"dismatch of arguments,requires:"+req_args.length+",provided:"+args.length)
	      heads_(0).source.value.reset_inputs()
	      for(i <- 0 until args.length){
	          heads_(0).source.value.inputs :+= args(i).heads_(0)  
	      }
	      for(i<-args.length until req_args.length){
  	        val noderef = new NodeRef()
  					val node = new Node(new OperatorPropertyRef, Symbol.DefaultVarName(name,req_args(i)))
  //					added by liuxianggen,which is none in brainmatrix c++
  //					node.backward_source_node = heads_(0).source
  					noderef.value = node
  					heads_(0).source.value.inputs :+= new DataEntry(noderef, 0)
  	        if (heads_(0).source.value.attr.size != 0)
						  heads_(0).source.value.inputs(i).source.value.attr = heads_(0).source.value.attr
	      }
	       
	    
	    }
	 
	  
	  }
	
	


	 /**
     * @author lxg
     * @date 20160706
     * @brief get the index-th symbol from this group which is from symbol
     * @param index
     * @return symbol
     * @note 
     */
	def get(index:Int):Symbol = {
	    require(index<this.heads_.length,"the index overcome the length of group size!!")
	    val s = new Symbol((new SymbolHandleRef).value)
	    s.heads_ :+= this.heads_(index)
	    s
	    
	}
	
	
	
	
	

	/**
	 * 2016-3-15
	 * by liuxianggen
	 * find the most number of duplicate arguments
	 *
	 */
	private def FindDuplicateArgs: (Map[String, Int], Int) = {
	  import scala.collection.mutable.Map
		var out = Map[String, Int]()
		var max_dup: Int = 1;
		this.DFSVisit { noderef =>
			{
				if (noderef.value.is_variable)
					if (out.contains(noderef.value.name)) {
						out(noderef.value.name) += 1
						max_dup = Math.max(max_dup, out(noderef.value.name))
					} else
						out(noderef.value.name) = 1
			}
		}
		(out.toMap, max_dup)
	}

	/**
	 * 2016-3-14
	 * by liuxianggen
	 * the key function to convert graph from symbol graph
	 */
	def ToStaticGraph() {
		var node_order: Vector[NodeRef] = Vector()
		var node_index: Map[NodeRef, Int] = Map()
//		this.staticGraph.arg_nodes = Vector()
//		this.staticGraph.nodes = Vector()
		this.staticGraph.reset
		this.DFSVisit { noderef =>
			{
				var nid: Int = node_index.size
				node_index += (noderef -> nid)
				if (noderef.value.is_variable()) {
					this.staticGraph.arg_nodes :+= nid
				}
				node_order :+= noderef
			}
		}
		//setup nodes
		/**
		 * which is different with c++, new the node first in scala
		 */
		(0 until node_order.size).map(nid => {
			val ophandle = new OperatorPropertyRef
			var node: Node = new Node(ophandle)
			if (node_order(nid).value.opRef.value != null) {

				node.opRef.value = node_order(nid).value.opRef.value.Copy()
				this.staticGraph.nodes :+= node
			} else {
				this.staticGraph.nodes :+= node
			}
			if (node_order(nid).value.backward_source_node.value != null) {
				this.staticGraph.nodes(nid).backward_source_id = node_index(node_order(nid).value.backward_source_node)
			} else {
				this.staticGraph.nodes(nid).backward_source_id = -1
			}
			if (node_order(nid).value.attr != null) {
				this.staticGraph.nodes(nid).attr = node_order(nid).value.attr
			}
			this.staticGraph.nodes(nid).name = node_order(nid).value.name
			/*
  			 * out_graph.nodes(nid).inputs.clear
  			 */
			this.staticGraph.nodes(nid).inputs = Vector()
			(node_order(nid).value.inputs).map(src => {
				var e: DataEntry = new DataEntry(new NodeRef, src.index)
				e.source_id = node_index(src.source)
				this.staticGraph.nodes(nid).inputs :+= e
			})
		})
		this.staticGraph.heads = Vector()
		this.heads_.foreach { head =>
			{
				var e: DataEntry = new DataEntry(new NodeRef, head.index)
				e.source_id = node_index(head.source)
				this.staticGraph.heads :+= e
			}
		}

	}
	
	def debug():String = {
		this.ToStaticGraph()
		this.staticGraph.debug
	}
	
	
    /**
     * @author liuxianggen
     * @date 20160708
     * @brief given the shape of inputs,get total shape info with this special symbol graph
     * the shape info include:
     *  inShapeData:shapes of all the args symbol,in order
     *  outShapeData:shapes of all the head in heads_ of symbol,when the length of head>1,means this symbol is a group
     *  auxShapeData:need to clearfy 
     * @param kwargs:map the name of inputs to it's shape, such as Map("data" -> Vector(1, 3, 4, 5))
     * @return as described aboved mentioned
     * @note:when this is a group, this function will find all the head, and return all the info from whatever head
     * @example:
     * 			sym.inferShape(Map("data" -> Vector(1, 3, inputSize._1, inputSize._2)))
     */
    def inferShape(kwargs:Map[String,Shape]): (Seq[Shape], Seq[Shape], Seq[Shape]) = {
    	val inShapeData = ListBuffer.empty[Array[Int]]
        val outShapeData = ListBuffer.empty[Array[Int]]
        val auxShapeData = ListBuffer.empty[Array[Int]]
        val complete = new RefInt
    	this.ToStaticGraph()
    	this.staticGraph.inferShape(kwargs,inShapeData, outShapeData, auxShapeData, complete)
        	
        if (complete.value != 0) {
          (inShapeData.map(Shape(_)), outShapeData.map(Shape(_)), auxShapeData.map(Shape(_)))
        } else {
          (null, null, null)
        }
    	}
    /**
	 * 2016-3-23
	 * by liuxianggen
	 */
	 def SetAttr(key:String,value:String){
		val node:NodeRef = heads_(0).source
		heads_.foreach { e => {
			require(node == e.source,"error")
//			if(node == e.source)
//				println("True")
			} 
		}
		if(node.value.attr.size == 0){
			node.value.attr = scala.collection.mutable.Map[String,String]() 
		}
		node.value.attr(key)  = value
	}

	  // Set the attribute of the symbol.
   def setAttr(attr: Map[String, String]): Unit = {
    attr.foreach { case (key, value) =>
      SetAttr(key,value)
    }
  }
	
	
	
	
	
  def +(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Plus")(Array(this, other))
  def +[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_PlusScalar")(Array(this), Map("scalar" -> other.toString))
  }

  def -(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Minus")(Array(this, other))
  def -[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_MinusScalar")(Array(this), Map("scalar" -> other.toString))
  }

  def *(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Mul")(Array(this, other))
  def *[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_MulScalar")(Array(this), Map("scalar" -> other.toString))
  }

  def /(other: Symbol): Symbol = Symbol.createFromListedSymbols("_Div")(Array(this, other))
  def /[@specialized(Int, Float, Double) V](other: V): Symbol = {
    Symbol.createFromListedSymbols("_DivScalar")(Array(this), Map("scalar" -> other.toString))
  }

  //need to change
  override def clone(): Symbol = {
    val clonedHandle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCopy(handle, clonedHandle))
    new Symbol(clonedHandle.value)
  }

//  def get(index: Int): Symbol = {
//    val newHandle = new SymbolHandleRef
//    checkCall(_LIB.mxSymbolGetOutput(handle, index, newHandle))
//    new Symbol(handle = newHandle.value)
//  }

  def get(name: String): Symbol = {
    var index: Int = -1
    for ((output, i) <- listOutputs().view.zipWithIndex) {
      if (output == name) {
        require(index == -1, s"There are multiple outputs with name $name")
        index = i
      }
    }
    require(index >= 0, s"Cannot find output that matches name $name")
    get(index)
  }

  
  
  

  /**
   * Infer the type of outputs and arguments of given known types of arguments.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known types passed in.
   * @param args Provide type of arguments in a positional way. Unknown type can be marked as null
   * @return
   * argTypes : list of numpy.dtype or None
   *            List of types of arguments.
   *            The order is in the same order as list_arguments()
   * outTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_outputs()
   * auxTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_auxiliary()
   */
  def inferType(args: Class[_ >: Float with Int with Double]*)
    : (Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]]) = {
    val sdata: Array[Int] = args.map(NDArray.DTYPE_NATIVE_TO_MX.getOrElse(_, -1)).toArray
    inferType(null, sdata)
  }

  /**
   * Infer the type of outputs and arguments of given known types of arguments.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known types passed in.
   * @param kwargs Provide keyword arguments of known types.
   * @return
   * argTypes : list of numpy.dtype or None
   *            List of types of arguments.
   *            The order is in the same order as list_arguments()
   * outTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_outputs()
   * auxTypes : list of numpy.dtype or None
   *            List of types of outputs.
   *            The order is in the same order as list_auxiliary()
   */
  def inferType(kwargs: Map[String, Class[_ >: Float with Int with Double]])
    : (Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]]) = {
    val filteredArgs = kwargs.filter { case (key, value) =>
      NDArray.DTYPE_NATIVE_TO_MX.contains(value)
    }
    val keys = filteredArgs.keys.toArray
    val sdata = filteredArgs.values.map(NDArray.DTYPE_NATIVE_TO_MX(_)).toArray
    inferType(keys, sdata)
  }

  private def inferType(keys: Array[String], values: Array[Int])
    : (Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]],
       Seq[Class[_ >: Float with Int with Double]]) = {
    val argTypeData = ListBuffer.empty[Int]
    val outTypeData = ListBuffer.empty[Int]
    val auxTypeData = ListBuffer.empty[Int]
    val complete = new RefInt
    checkCall(_LIB.mxSymbolInferType(
      handle, keys, values, argTypeData, outTypeData, auxTypeData, complete))
    if (complete.value != 0) {
      (argTypeData.map(NDArray.DTYPE_MX_TO_NATIVE),
        outTypeData.map(NDArray.DTYPE_MX_TO_NATIVE),
        auxTypeData.map(NDArray.DTYPE_MX_TO_NATIVE))
    } else {
      (null, null, null)
    }
  }

  /**
   * Infer the shape of outputs and arguments of given known shapes of arguments.
   * User can either pass in the known shapes in positional way or keyword argument way.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known shapes passed in.
   * @param args Provide shape of arguments in a positional way.
   *             Unknown shape can be marked as None
   * @return
   * argShapes List of shapes of arguments. The order is in the same order as list_arguments()
   * outShapes List of shapes of outputs. The order is in the same order as list_outputs()
   * auxShapes List of shapes of outputs. The order is in the same order as list_auxiliary()
   */
//  def inferShape(args: Shape*): (Seq[Shape], Seq[Shape], Seq[Shape]) = {
//    val keys: Array[String] = null
//    val indPtr = ArrayBuffer(0)
//    val sdata = ArrayBuffer.empty[Int]
//    args.foreach { shape =>
//      if (shape != null) {
//        sdata ++= shape.toVector
//        indPtr += sdata.size
//      }
//    }
//    inferShape(keys, indPtr.toArray, sdata.toArray)
//  }

  /**
   * Infer the shape of outputs and arguments of given known shapes of arguments.
   * User can either pass in the known shapes in positional way or keyword argument way.
   * Tuple of Nones is returned if there is not enough information passed in.
   * An error will be raised if there is inconsistency found in the known shapes passed in.
   * @param kwargs Provide keyword arguments of known shapes.
   * @return
   * argShapes List of shapes of arguments. The order is in the same order as list_arguments()
   * outShapes List of shapes of outputs. The order is in the same order as list_outputs()
   * auxShapes List of shapes of outputs. The order is in the same order as list_auxiliary()
   */
//  def inferShape(kwargs: Map[String, Shape]): (Seq[Shape], Seq[Shape], Seq[Shape]) = {
//    val keys = ArrayBuffer.empty[String]
//    val indPtr = ArrayBuffer(0)
//    val sdata = ArrayBuffer.empty[Int]
//    kwargs.foreach { case (key, shape) =>
//      keys += key
//      sdata ++= shape.toVector
//      indPtr += sdata.size
//    }
//    inferShape(keys.toArray, indPtr.toArray, sdata.toArray)
//  }
//
//  def inferShape(keys: Array[String], indPtr: Array[Int], values: Array[Int])
//    : (Seq[Shape], Seq[Shape], Seq[Shape]) = {
//    val argShapeData = ListBuffer.empty[Array[Int]]
//    val outShapeData = ListBuffer.empty[Array[Int]]
//    val auxShapeData = ListBuffer.empty[Array[Int]]
//    val complete = new RefInt
//
//    checkCall(_LIB.mxSymbolInferShape(handle, indPtr.size - 1, keys, indPtr, values,
//      argShapeData, outShapeData, auxShapeData, complete))
//    if (complete.value != 0) {
//      (argShapeData.map(s => Shape(s)),
//       outShapeData.map(s => Shape(s)),
//       auxShapeData.map(s => Shape(s)))
//    } else {
//      (null, null, null)
//    }
//  }

  /**
   * Get attribute string from the symbol, this function only works for non-grouped symbol.
   * @param key  The key to get attribute from.
   * @return value The attribute value of the key, returns None if attribute do not exist.
   */
//  def attr(key: String): Option[String] = {
//    val ret = new RefString
//    val success = new RefInt
//    checkCall(_LIB.mxSymbolGetAttr(handle, key, ret, success))
//    if (success.value != 0) {
//      Option(ret.value)
//    } else {
//      None
//    }
//  }

  /**
   * Invoke symbol as function on inputs.
   * @param name resulting symbol name
   * @param symbols provide named symbols
   * @return the resulting symbol
   */
  def apply(name: String, symbols: Map[String, Symbol]): Symbol = {
    val s = clone()
    s.compose(name, symbols)
    s
  }

  /**
   * Get a debug string.
   * @return Debug string of the symbol.
   */
  def debugStr: String = {
    val str = new RefString
    checkCall(_LIB.mxSymbolPrint(handle, str))
    str.value
  }

  // Set the attribute of the symbol.
//  private def setAttr(attr: Map[String, String]): Unit = {
//    attr.foreach { case (key, value) =>
//      checkCall(_LIB.mxSymbolSetAttr(handle, key, value))
//    }
//  }

 /**
   * Save symbol into file.
   * You can also use pickle to do the job if you only work on python.
   * The advantage of load/save is the file is language agnostic.
   * This means the file saved using save can be loaded by other language binding of mxnet.
   * You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
   *
   * @param fname The name of the file
   *        - s3://my-bucket/path/my-s3-symbol
   *        - hdfs://my-bucket/path/my-hdfs-symbol
   *        - /path-to/my-local-symbol
   * @see Symbol.load : Used to load symbol from file.
   */
  def save(fname: String): Unit = {
	  this.ToStaticGraph()
	  this.staticGraph.saveToFile(fname)
  }

  /**
   * Compose symbol on inputs.
   * This call mutates the current symbol.
   * @param name resulting symbol name
   * @param symbols provide positional arguments
   * @return the resulting symbol
   */
  private def compose(name: String, symbols: Array[Symbol]): Unit = {
    val args = symbols.map(_.handle)
    checkCall(_LIB.mxSymbolCompose(handle, name, null, args))
  }

  private def compose(name: String, symbols: Map[String, Symbol]): Unit = {
    val keys = symbols.keys.toArray
    val args = symbols.values.map(_.handle).toArray
    checkCall(_LIB.mxSymbolCompose(handle, name, keys, args))
  }

  /**
   * Bind current symbol to get an executor, allocate all the ndarrays needed.
   * Allows specifying data types.
   * This function will ask user to pass in ndarray of position
   * they like to bind to, and it will automatically allocate the ndarray
   * for arguments and auxiliary states that user did not specify explicitly.
   *
   * @param ctx The device context the generated executor to run on.
   * @param gradReq {'write', 'add', 'null'}, or list of str or dict of str to str, optional
   *                Specifies how we should update the gradient to the args_grad.
   *                - 'write' means everytime gradient is write to specified args_grad NDArray.
   *                - 'add' means everytime gradient is add to the specified NDArray.
   *                - 'null' means no action is taken, the gradient may not be calculated.
   * @param typeDict Input type dictionary, name->dtype
   * @param shapeDict Input shape dictionary, name->shape
   * @return The generated Executor
   */
  def simpleBind(ctx: Context, gradReq: String = "write",
                 shapeDict: Map[String, Shape],
                 typeDict: Map[String, Class[_ >: Float with Int with Double]] = null): Executor = {
    val types =
      if (typeDict == null) listArguments().map((_, classOf[Float])).toMap
      else typeDict
      
    val (argShapes, _, auxShapes) = inferShape(shapeDict)
//    val (argTypes, _, auxTypes) = inferType(types)
//    require(argShapes != null && argTypes != null, "Input node is not complete")
    require(argShapes != null, "Input node is not complete")
    // alloc space
    val argNDArrays = (argShapes) map { case (shape) =>
      // TODO: NDArray dtype
      NDArray.zeros(shape, ctx)
    }
    val gradNDArrays =
      if (gradReq != "null") {
        ((listArguments() zip argShapes) flatMap { case (name, shape) =>
          if (!(name.endsWith("data") || name.endsWith("label"))) {
            // TODO: NDArray dtype
            Map(name -> NDArray.zeros(shape, ctx))
          } else {
            Map.empty[String, NDArray]
          }
        }).toMap
      } else {
        null
      }
    val auxNDArrays = (auxShapes) map { case (shape) =>
      // TODO: NDArray dtype
      NDArray.zeros(shape, ctx)
    }
    bind(ctx, argNDArrays, gradNDArrays, gradReq, auxNDArrays, null, null)
  }

  /**
   * Bind current symbol to get an executor.
   *
   * @param ctx Context The device context the generated executor to run on.
   * @param args Input arguments to the symbol.
   *             - If type is list of NDArray, the position is in the same order of list_arguments.
   *             - If type is dict of str to NDArray, then it maps the name of arguments
   *               to the corresponding NDArray.
   *             - In either case, all the arguments must be provided.
   * @param argsGrad When specified, args_grad provide NDArrays to hold
   *                 the result of gradient value in backward.
   *                 - If type is list of NDArray,
   *                   the position is in the same order of list_arguments.
   *                 - If type is dict of str to NDArray, then it maps the name of arguments
   *                   to the corresponding NDArray.
   *                 - When the type is dict of str to NDArray, users only need to provide the dict
   *                   for needed argument gradient.
   *                   Only the specified argument gradient will be calculated.
   * @param gradReq {'write', 'add', 'null'}, or list of str or dict of str to str, optional
   *                Specifies how we should update the gradient to the args_grad.
   *                - 'write' means everytime gradient is write to specified args_grad NDArray.
   *                - 'add' means everytime gradient is add to the specified NDArray.
   *                - 'null' means no action is taken, the gradient may not be calculated.
   * @param auxStates Input auxiliary states to the symbol, only need to specify when
   *                  list_auxiliary_states is not empty.
   *                  - If type is list of NDArray,
   *                    the position is in the same order of listAuxiliaryStates
   *                  - If type is dict of str to NDArray, then it maps the name of auxiliary_states
   *                    to the corresponding NDArray,
   *                  - In either case, all the auxiliary_states need to be provided.
   * @param group2ctx The dict mapping the ``ctx_group`` attribute to the context assignment.
   * @param sharedExec Executor to share memory with.
   *                 - This is intended for runtime reshaping, variable length sequences, etc.
   *                 - The returned executor shares state with shared_exec,
   *                   and should not be used in parallel with it.
   * @return The generated Executor
   * @note
   * Auxiliary states are special states of symbols that do not corresponds to an argument,
   * and do not have gradient. But still be useful for the specific operations.
   * A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   * Most operators do not have auxiliary states and this parameter can be safely ignored.
   *
   * User can give up gradient by using a dict in args_grad and only specify
   * gradient they interested in.
   */
  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradReq: String, auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad,
               Seq.fill(symbolArguments.size)(gradReq), auxStates, group2ctx, sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Seq[String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Seq[NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray],
           gradsReq: Map[String, String], auxStates: Map[String, NDArray],
           group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, argsGrad, gradsReq, auxStates, group2ctx,
      sharedExec)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Seq[NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Map[String, NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Map[String, NDArray], argsGrad: Seq[NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Seq[NDArray], argsGrad: Map[String, NDArray]): Executor = {
    bind(ctx, args, argsGrad, "write", Nil, null, null)
  }

  def bind(ctx: Context, args: Seq[NDArray]): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, null,
               Seq.fill(symbolArguments.size)("write"), Nil, null, null)
  }

  def bind(ctx: Context, args: Map[String, NDArray]): Executor = {
    val symbolArguments = listArguments()
    bindHelper(ctx, symbolArguments, args, null,
      Seq.fill(symbolArguments.size)("write"), Nil, null, null)
  }

  private def bindHelper(ctx: Context, symbolArguments: Seq[String],
                         args: Iterable[_], argsGrad: Iterable[_],
                         gradsReq: Iterable[_], auxStates: Iterable[_],
                         group2ctx: Map[String, Context], sharedExec: Executor): Executor = {
    require(args != null && !args.isInstanceOf[Set[_]])
    require(argsGrad == null || !argsGrad.isInstanceOf[Set[_]])
    require(auxStates == null || !auxStates.isInstanceOf[Set[_]])
    require(gradsReq != null && !gradsReq.isInstanceOf[Set[_]])

    val (argsHandle, argsNDArray) =
      if (args.isInstanceOf[Seq[_]]) {
        Symbol.getNDArrayInputs("args", args.asInstanceOf[Seq[NDArray]],
                                symbolArguments, allowMissing = false)
      } else {
        Symbol.getNDArrayInputs("args", args.asInstanceOf[Map[String, NDArray]],
                                symbolArguments, allowMissing = false)
      }

    // setup args gradient
    val (argsGradHandle, argsGradNDArray) =
      if (argsGrad == null) {
        (Array.fill[NDArrayHandle](args.size)(0L), null)
      } else if (argsGrad.isInstanceOf[Seq[_]]) {
        Symbol.getNDArrayInputs("args_grad", argsGrad.asInstanceOf[Seq[NDArray]],
                                symbolArguments, allowMissing = true)
      } else {
        Symbol.getNDArrayInputs("args_grad", argsGrad.asInstanceOf[Map[String, NDArray]],
                                symbolArguments, allowMissing = true)
      }

    val (auxArgsHandle, auxStatesNDArray) =
      if (auxStates == null) {
        Symbol.getNDArrayInputs("aux_states", Nil, listAuxiliaryStates(), allowMissing = false)
      } else if (auxStates.isInstanceOf[Seq[_]]) {
        Symbol.getNDArrayInputs("aux_states", auxStates.asInstanceOf[Seq[NDArray]],
                                listAuxiliaryStates(), allowMissing = false)
      } else {
        Symbol.getNDArrayInputs("aux_states", auxStates.asInstanceOf[Map[String, NDArray]],
                                listAuxiliaryStates(), allowMissing = false)
      }

    // setup requirements
    val reqsArray =
      if (gradsReq.isInstanceOf[Seq[_]]) {
        gradsReq.asInstanceOf[Seq[String]].map { req =>
          require(Symbol.bindReqMap.contains(req), s"grad_req must be in ${Symbol.bindReqMap}")
          Symbol.bindReqMap(req)
        }.toArray
      } else {
        val gradsReqMap = gradsReq.asInstanceOf[Map[String, String]]
        symbolArguments.map { req =>
          val value = gradsReqMap.getOrElse(req, "null")
          require(Symbol.bindReqMap.contains(value), s"grad_req must be in ${Symbol.bindReqMap}")
          Symbol.bindReqMap(value)
        }.toArray
      }

    val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]

    if (group2ctx != null) {
      group2ctx.foreach { case (key, value) =>
        ctxMapKeys += key
        ctxMapDevTypes += value.deviceTypeid
        ctxMapDevIDs += value.deviceId
      }
    }

    val sharedHadle = if (sharedExec != null) sharedExec.handle else 0L
    
//     println("*********************************")
//    println("args:")
//    println(argsHandle.length)
//    println("size:")
//    argsHandle.foreach(x => {
//    	println(new NDArray(x).shape)
//    			})
//    			
//    println("argsGrad:")
//    println(auxArgsHandle.length)
////    println("size:")
////    gradNDArraysHandles.foreach(x => {
////    	println(new NDArray(x).shape)
////    			})
//    auxArgsHandle.foreach(println)
//    			
//    println("!!!")
//    reqsArray.foreach(println)
//    
//    if(auxArgsHandle!=null){
//	    println("auxArgs:")
//	    println(auxArgsHandle.length)
//	    println("size:")
//	    auxArgsHandle.foreach(x => {
//	    	println(new NDArray(x).shape)
//	    			})
//    }
    
    
    
    
    this.ToStaticGraph()
 	this.staticGraph.ToStaticGraph
// 	println("---------------")
 	val execRef = this.staticGraph.bind(ctx.deviceTypeid,//1
                                         ctx.deviceId,//0
                                         ctxMapKeys.size,//0
                                         ctxMapKeys.toArray,//null
                                         ctxMapDevTypes.toArray,//null
                                         ctxMapDevIDs.toArray,//null
                                         args.size,
                                         argsHandle,
                                         argsGradHandle,
                                         reqsArray,
                                         auxArgsHandle)
//    println("---------------")
//    checkCall(_LIB.mxExecutorBindEX(handle,
//                                   ctx.deviceTypeid,
//                                   ctx.deviceId,
//                                   ctxMapKeys.size,
//                                   ctxMapKeys.toArray,
//                                   ctxMapDevTypes.toArray,
//                                   ctxMapDevIDs.toArray,
//                                   args.size,
//                                   argsHandle,
//                                   argsGradHandle,
//                                   reqsArray,
//                                   auxArgsHandle,
//                                   sharedHadle,
//                                   execHandle))
//    
//    val executor = new Executor(execHandle.value, this.clone())//vital code!!!
//    
    
    val executor = new Executor(execRef.value, this)
    executor.argArrays = argsNDArray
    executor.gradArrays = argsGradNDArray
    executor.auxArrays = auxStatesNDArray
    executor._ctx = new Context(ctx.deviceType, ctx.deviceId)
    executor._gradsReq = gradsReq
    executor._group2ctx =
      if (group2ctx == null) null
      else group2ctx.map { case (key, value) =>
        (key -> new Context(value.deviceType, value.deviceId))
      }.toMap
    executor
  }

  
   def easy_bind(ctx: Context = Context.defaultCtx, args: Map[String,NDArray], argsGrad: Map[String,NDArray]=null,
           auxStates: Map[String,NDArray]=null,group2ctx: Map[String, Context]=null, gradReq: String = "write"): Executor = {
	
    val (argHandle,argNDArrays) = Symbol.getNDArrayInputs("args",args,listArguments(),false)
    val gradMap =
    	if(argsGrad==null){
	        ((listArguments() zip argNDArrays)  flatMap { case (name, argArr) =>
	          if (!(name.endsWith("data") || name.endsWith("label"))) {
	            // TODO: NDArray dtype
	            Map(name -> NDArray.zeros(argArr.shape, ctx))
	          } else {
	            Map.empty[String, NDArray]
	          }
	        }).toMap
	    }else
	    	argsGrad
	    	
	 
    var (gradNDArraysHandles,gradNDArrays) = Symbol.getNDArrayInputs("args_grad",gradMap,listArguments(),true)
  
    /**
     * aux
     */
    
    val (auxArgsHandle, auxStatesNDArray) =
      if (auxStates == null) {
        Symbol.getNDArrayInputs("aux_states", Nil, listAuxiliaryStates(), allowMissing = false)
      } else if (auxStates.isInstanceOf[Seq[_]]) {
        Symbol.getNDArrayInputs("aux_states", auxStates.asInstanceOf[Seq[NDArray]],
                                listAuxiliaryStates(), allowMissing = false)
      } else {
        Symbol.getNDArrayInputs("aux_states", auxStates.asInstanceOf[Map[String, NDArray]],
                                listAuxiliaryStates(), allowMissing = false)
      }
    
    /**
     * reqArray
     * 
     */
    var  gradReqArrays = Array[String]()
    if(gradReq.equals("write")){
//    	 gradReqArrays= Symbol.getNDArrayInputsPlus("aux_states",auxStates,this.listAuxiliaryStates(),false)._3	
         gradReqArrays  = Array.fill(gradNDArrays.length)("write")
    }else
    	gradReqArrays  = Array.fill[String](gradNDArrays.length)("null") 
    val reqsArray: Array[Int] = gradReqArrays.map(Symbol.bindReqMap(_))
    val ctxMapKeys = ArrayBuffer.empty[String]
    val ctxMapDevTypes = ArrayBuffer.empty[Int]
    val ctxMapDevIDs = ArrayBuffer.empty[Int]

    if (group2ctx != null) {
      group2ctx.foreach { case (key, value) =>
        ctxMapKeys += key
        ctxMapDevTypes += value.deviceTypeid
        ctxMapDevIDs += value.deviceId
      }
    }

        
//    println("*********************************")
//    println("args:")
//    println(argHandle.length)
//    println("size:")
//    argHandle.foreach(x => {
//    	println(new NDArray(x).shape)
//    			})
//    			
//    println("argsGrad:")
//    println(gradNDArraysHandles.length)
////    println("size:")
////    gradNDArraysHandles.foreach(x => {
////    	println(new NDArray(x).shape)
////    			})
//    gradNDArraysHandles.foreach(println)
//    			
//    println("!!!")
//    reqsArray.foreach(println)
//    
//    if(auxArgsHandle!=null){
//	    println("auxArgs:")
//	    println(auxArgsHandle.length)
//	    println("size:")
//	    auxArgsHandle.foreach(x => {
//	    	println(new NDArray(x).shape)
//	    			})
//    }
    
    this.ToStaticGraph()
 		this.staticGraph.ToStaticGraph
 		val execRef = this.staticGraph.bind(ctx.deviceTypeid,//1
                                         ctx.deviceId,//0
                                         ctxMapKeys.size,//0
                                         ctxMapKeys.toArray,//null
                                         ctxMapDevTypes.toArray,//null
                                         ctxMapDevIDs.toArray,//null
                                         argNDArrays.size,
                                         argHandle,
                                         gradNDArraysHandles,
                                         reqsArray,
                                         auxArgsHandle)
                                   
    val executor = new Executor(execRef.value, this)
    executor.argArrays = argNDArrays
    executor.gradArrays = gradNDArrays
    executor.auxArrays = auxStatesNDArray
    executor
  }
  
  
  
  /**
   * Save symbol into a JSON string.
   * See Also
   * symbol.loadJson : Used to load symbol from JSON string.
   */
  def toJson: String = {
    val jsonStr = new RefString
    this.ToStaticGraph()
    this.staticGraph.ToStaticGraph
    checkCall(_LIB.mxStaticGraphSaveToJSON(this.staticGraph.handle,jsonStr))
    jsonStr.value
  }
  
  /**
	 * list all the arguments of this symbol
	 */
	def listArguments():Array[String] = {
		val arr = Stack[String]()
		if(this.is_atomic()){
			heads_(0).source.value.opRef.value.ListArguments.toArray
		}else{
			this.DFSVisit { x => {
				if(x.value.is_variable()){
					arr.push(x.value.name)
				}
			} }
			arr.reverse.toArray	
		}
    
	}
	
	 /**
   * List all auxiliary states in the symbol.
   * @return The names of the auxiliary states.
   * @note
   * Auxiliary states are special states of symbols that do not corresponds to an argument,
   * and do not have gradient. But still be useful for the specific operations.
   * A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   * Most operators do not have Auxiliary states.
   */
  def listAuxiliaryStates(): Seq[String] = {
    val aarr = Stack[String]()
    if(this.is_atomic()){
    	heads_(0).source.value.opRef.value.ListAuxiliaryStates()
    }else{
    	this.DFSVisit { x => {
    		if(x.value.opRef.value!=null){
    			val aux_args = x.value.opRef.value.ListAuxiliaryStates()
    			if(aux_args.length>0){
    				val hname = x.value.name
    				aux_args.foreach(x => aarr.push(hname + "_" + x))
    			}
    		}
    	} }
    }
    
    aarr.reverse
  }
	
  
  	/**
     * @author lxg
     * @date 20161230
     * @brief get the all nodes
     * @
     * @return symbol
     * @note 
     * 
   	 * Get a new grouped symbol whose output contains all the internal outputs of this symbol.
     * @return The internal of the symbol.
     */
     
	def getInternals():Symbol = {
		val s = new Symbol((new SymbolHandleRef).value)
		this.heads_.foreach { s.heads_ :+= _ }
		var nout  =  0
		this.DFSVisit { nodeRef => {
			val node = nodeRef.value
			if(node.is_variable()){
				nout = 1
			} 
			else if(node.is_backward()){
				nout = node.backward_source_node.value.inputs.size
			}
			else {
				nout = node.opRef.value.NumVisibleOutputs()
			}
			for(i <- 0 until nout){
				s.heads_ :+= new DataEntry(nodeRef, i)
			}}
		}
	  
	    s
	}  
  
  
  
  
    /**
     * @author liuxianggen
     * @date 20160708
     * @brief get the name of all the symbols in the group
     * @param 
     * @return Vector[String]：the name of all the symbols in the group
     * @example 
     * @note
     */
	def listOutputs(): Vector[String] = {
		var res: Vector[String] = Vector[String]()
		this.heads_.map { head =>
			{
				if (head.source.value.is_variable()) {
					res :+= head.source.value.name
				} else {
					var rname: String = null
					// the output of node is the corresponding input of its backward node, so,,,  
					if (head.source.value.is_backward()) {
						rname = head.source.value.backward_source_node.value.opRef.value.ListArguments(head.index)
					} else {
						rname = head.source.value.opRef.value.ListOutputs(head.index)
					}
					val hname = head.source.value.name
					if (hname.length() == 0)
						res :+= rname
					else
						res :+= (hname + "_" + rname)
				}
			}
		}
		res
	}
//  def listOutputs() : Seq[String] = {
//    val arr = ArrayBuffer.empty[String]
//    val outputs_arr = Stack[String]()
//    for(head <- heads_){
//      if(head.source.value.is_variable()){
//        outputs_arr.push(head.source.value.name)
//      }else{
//        val hname = head.source.value.name
//        var rname:String =null
//        if(head.source.value.is_backward()){
//          rname = head.source.value.backward_source_node.value.opRef.value.ListArguments(head.index)
//        }else{
//          rname = head.source.value.opRef.value.ListOutputs(head.index)
//        }
//        if(head.source.value.name.length()==0){
//            outputs_arr.push(rname)
//        }else{
//            outputs_arr.push(head.source.value.name + "_" +rname)
//        }
//        
//      }
//    }
//    
//    
//    checkCall(_LIB.mxSymbolListOutputs(handle, arr))
//    arr
//  }
  
}
// scalastyle:on finalize

object Symbol {
  private type SymbolCreateNamedFunc = Map[String, Any] => Symbol
  private val logger = LoggerFactory.getLogger(classOf[Symbol])
  private val functions: Map[String, SymbolFunction] = initSymbolModule()
  private val bindReqMap = Map("null" -> 0, "write" -> 1, "add" -> 3)

  // TODO: _CrossDeviceCopy

  def pow(sym1: Symbol, sym2: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_Power")(Array(sym1, sym2))
  }

  def pow[@specialized(Int, Float, Double) V](sym: Symbol, number: V): Symbol = {
    Symbol.createFromListedSymbols("_PowerScalar")(Array(sym), Map("scalar" -> number.toString))
  }

  def pow[@specialized(Int, Float, Double) V](number: V, sym: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_RPowerScalar")(Array(sym), Map("scalar" -> number.toString))
  }

  /**
   * Take absolute value of the src
   * @param src Source symbolic input to the function
   */
  def abs(src: Symbol): Symbol = {
    createFromListedSymbols("abs")(Array(src))
  }

  /**
   * Take sign value of the src
   * @param src Source symbolic input to the function
   */
  def sign(src: Symbol): Symbol = {
    createFromListedSymbols("sign")(Array(src))
  }

  /**
   * Take round value of the src
   * @param src Source input to the function
   */
  def round(src: Symbol): Symbol = {
    createFromListedSymbols("round")(Array(src))
  }

  /**
   * Take ceil value of the src
   * src Source input to the function
   */
  def ceil(src: Symbol): Symbol = {
    createFromListedSymbols("ceil")(Array(src))
  }

  /**
   * Take floor value of the src
   * @param src Source input to the function
   */
  def floor(src: Symbol): Symbol = {
    createFromListedSymbols("floor")(Array(src))
  }

  /**
   * Take square of the src
   * @param src Source symbolic input to the function
   */
  def square(src: Symbol): Symbol = {
    createFromListedSymbols("square")(Array(src))
  }

  /**
   * Take sum of the src
   * @param src Source symbolic input to the function
   */
  def sum(src: Symbol): Symbol = {
    createFromListedSymbols("sum")(Array(src))
  }

  /**
   * Take sqrt of the src
   * src Source symbolic input to the function
   */
  def sqrt(src: Symbol): Symbol = {
    createFromListedSymbols("sqrt")(Array(src))
  }

  /**
   * Take rsqrt of the src
   * @param src Source symbolic input to the function
   */
  def rsqrt(src: Symbol): Symbol = {
    createFromListedSymbols("rsqrt")(Array(src))
  }

  /**
   * Take exp of the src
   * @param src Source symbolic input to the function
   */
  def exp(src: Symbol): Symbol = {
    createFromListedSymbols("exp")(Array(src))
  }

  /**
   * Take log of the src
   * @param src Source symbolic input to the function
   */
  def log(src: Symbol): Symbol = {
    createFromListedSymbols("log")(Array(src))
  }

  /**
   * Take cos of the src
   * @param src Source symbolic input to the function
   */
  def cos(src: Symbol): Symbol = {
    createFromListedSymbols("cos")(Array(src))
  }

  /**
   * Take sin of the src
   * @param src Source symbolic input to the function
   */
  def sin(src: Symbol): Symbol = {
    createFromListedSymbols("sin")(Array(src))
  }

  /**
   * Return transpose of the src
   * @param src Source symbolic input to the function
   */
  def transpose(src: Symbol): Symbol = {
    createFromListedSymbols("transpose")(Array(src))
  }

  def max(left: Symbol, right: Symbol): Symbol = {
    createFromListedSymbols("_Maximum")(Array(left, right))
  }

  def max[@specialized(Int, Float, Double) V](left: Symbol, right: V): Symbol = {
    createFromListedSymbols("_MaximumScalar")(Array(left), Map("scalar" -> right.toString))
  }

  def max[@specialized(Int, Float, Double) V](left: V, right: Symbol): Symbol = {
    createFromListedSymbols("_MaximumScalar")(Array(right), Map("scalar" -> left.toString))
  }

  def min(left: Symbol, right: Symbol): Symbol = {
    createFromListedSymbols("_Minimum")(Array(left, right))
  }

  def min[@specialized(Int, Float, Double) V](left: Symbol, right: V): Symbol = {
    createFromListedSymbols("_MinimumScalar")(Array(left), Map("scalar" -> right.toString))
  }

  def min[@specialized(Int, Float, Double) V](left: V, right: Symbol): Symbol = {
    createFromListedSymbols("_MinimumScalar")(Array(right), Map("scalar" -> left.toString))
  }

  
  
  def Dot(lhs:Symbol,rhs:Symbol,hiddenSize:Int):Symbol = {
	  
	  Symbol.FullyConnected("Dot")(Map("data"->lhs,"weight"->Symbol.transpose(rhs),"num_hidden" -> hiddenSize,"no_bias"->true))
  }
   
  
   /**
	lhs add rhs with broadcast
	
	Parameters
	----------
	lhs : Symbol
	Left symbolic input to the function
	rhs : Symbol
	Right symbolic input to the function
   */
  def broadcast_plus(left: Symbol, right: Symbol): Symbol = {
	   createFromListedSymbols("broadcast_plus")(Array(left, right))
  }
   
   /**
	lhs minus rhs with broadcast
	
	Parameters
	----------
	lhs : Symbol
	Left symbolic input to the function
	rhs : Symbol
	Right symbolic input to the function
   */
   def broadcast_minus(left: Symbol, right: Symbol): Symbol = {
    createFromListedSymbols("broadcast_minus")(Array(left, right))
   }
   
   
   /**
	lhs multiple rhs with broadcast
	
	Parameters
	----------
	lhs : Symbol
	Left symbolic input to the function
	rhs : Symbol
	Right symbolic input to the function
   */
   def broadcast_mul(left: Symbol, right: Symbol): Symbol = {
    	createFromListedSymbols("broadcast_mul")(Array(left, right))
   }
   
   /**
	lhs divide rhs with broadcast
	
	Parameters
	----------
	lhs : Symbol
	Left symbolic input to the function
	rhs : Symbol
	Right symbolic input to the function
   */
   def broadcast_div(left: Symbol, right: Symbol): Symbol = {
    	createFromListedSymbols("broadcast_div")(Array(left, right))
  }
   
   /**
	lhs power rhs with broadcast
	
	Parameters
	----------
	lhs : Symbol
	Left symbolic input to the function
	rhs : Symbol
	Right symbolic input to the function
   */
   def broadcast_power(left: Symbol, right: Symbol): Symbol = {
    	createFromListedSymbols("broadcast_div")(Array(left, right))
  }
  
    
  
  /**
	Take sum of the src in the given axis and returns a NDArray. Follows numpy semantics.
	
	Parameters
	----------
	src : Symbol
	Left symbolic input to the function
	axis : Shape(tuple), optional, default=()
	Same as Numpy. The axes to perform the reduction.If left empty, a global reduction will be performed.
	keepdims : boolean, optional, default=False
	Same as Numpy. If keepdims is set to true, the axis which is reduced is left in the result as dimension with size one.
	@example:
	val sum = Symbol.Sum("sum")(Map("data"->lhs,"axis"->2))
	if lhs.shape = (10,3,4)
	no axis   => (1)
	axis = 0  => (3,4)
	axis = 1  => (10,4)
	axis = 2  => (10,3)
	axis = 3  => error:src/operator/././broadcast_reduce_op_common.h:26: Check failed: param_axis[i] < max_ndim axes must be within the range, ndim of the source=3axis=(3,)
   */
   def Sum(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("sum", name, attr)
   }
  
  
   
	/**
	 * 2016-2-29
	 * there are two tasks:
	 * 1. create symbolBase
	 * 2. initialize the op
	 */
	def Create(operator: String,kwargs:Map[String,String] = null): Symbol = {
		val opref = new OperatorPropertyRef
        val op = OperatorProperty(operator)
//    if (kwargs != null){
//	    val paramkeys = (kwargs - "name").keys.toArray
//	    val paramvals = (kwargs - "name").values.toArray
//	    op.Init(paramkeys, paramvals)
//    }else
//    	System.err.println(s"the Symbol: $operator has no type to set, may be wrong")
//    	
    if (kwargs == null){
    		System.err.println(s"the Symbol: $operator has no type to set, may be wrong")
    }
	  val paramkeys = (kwargs - "name").keys.toArray
	  val paramvals = (kwargs - "name").values.toArray
	    op.Init(paramkeys, paramvals)
    
        opref.value = op  
//		
//		if(op.value == null){
//			System.err.println("error:op is not be initialized!")
//			null
//		}
		val node = new Node(opref, "")
		val nret: Int = op.NumVisibleOutputs()
		val sb: Symbol = new Symbol((new SymbolHandleRef).value)
		val noderef = new NodeRef()
		noderef.value = node
		(0 until nret).map(i => {
			sb.heads_ :+= new DataEntry(noderef, i)
		})
		
		sb
	}

	def Variable(name: String): Symbol = {
		val sb: Symbol = new Symbol((new SymbolHandleRef).value)
		val opref  = new OperatorPropertyRef
		val node = new Node(opref, name)
		val noderef = new NodeRef()
		noderef.value = node
		sb.heads_ :+= new DataEntry(noderef, 0);
		sb
	}

	
	def CreateVariable(name: String): Symbol = {
		val sb: Symbol = new Symbol((new SymbolHandleRef).value)
		val opref  = new OperatorPropertyRef
		val node = new Node(opref, name)
		val noderef = new NodeRef()
		noderef.value = node
		sb.heads_ :+= new DataEntry(noderef, 0);
		sb
	}
	
	
	/**
	 * 
	 * 
	 * 
	 */
	def Group(symbols:Symbol*):Symbol = {
	    val ret = new Symbol((new SymbolHandleRef).value)
	    symbols.foreach { s => ret.heads_ = ret.heads_ ++ s.heads_ }
	    ret
	}
	
	
	
	/**
	 * by liuxianggen
	 * 2016-3-9
	 */
	def CreateAtomicSymbol_mx(opName: String):OperatorProperty = {
		val op = OperatorProperty(opName)
		op
	}

	/**
	 * 2016-3-15
	 */
	private def DefaultVarName(op_name: String, arg_name: String): String = {
		if (op_name.size == 0)
			arg_name
		else
			op_name + "_" + arg_name
	}

  /**
   * Get output from a symbol and pass 0 gradient back
   *
   * Parameters
   * ----------
   * data : Symbol. Input data.
   */
  def BlockGrad(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("BlockGrad", name, attr)
  }

  /**
   * Crop the 2th and 3th dim of input data, with the corresponding size of w_h or with width
   * and height of the second input symbol
   *
   * Parameters
   * ----------
   * num_args : int, required.
   *            Number of inputs for crop,
   *            if equals one, then we will use the h_w for crop height and width,
   *            else if equals two,
   *            then we will use the height and width of the second input symbol,
   *            we name crop_like here
   * offset : Shape(tuple), optional, default=(0, 0), corp offset coordinate: (y, x)
   * h_w : Shape(tuple), optional, default=(0, 0), corp height and weight: (h, w)
   * center_crop : boolean, optional, default=False.
   *               If set to true, then it will use be the center_crop,
   *               or it will crop using the shape of crop_like
   */
  def Crop(name: String = null, attr: Map[String, String] = null)(
           inputs: Array[Symbol], params: Map[String, Any] = null): Symbol = {
    createFromListedSymbolsNoCheck("Crop", name, attr)(inputs, params)
  }

  /**
   * Apply dropout to input
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to dropout.
   * p : float, optional, default=0.5. Fraction of the input that gets dropped out at training time
   */
  def Dropout(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Dropout", name, attr)
  }

  /**
   * Apply a sparse regularization to the output a sigmoid activation function.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data.
   * sparseness_target : float, optional, default=0.1. The sparseness target
   * penalty : float, optional, default=0.001. The tradeoff parameter for the sparseness penalty
   * momentum : float, optional, default=0.9. The momentum for running average
   */
  def IdentityAttachKLSparseReg(name: String = null,
                                attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("IdentityAttachKLSparseReg", name, attr)
  }

  /**
   * Apply activation function to input.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to activation function.
   * act_type : {'elu', 'leaky', 'prelu', 'rrelu'},optional, default='leaky'
   *            Activation function to be applied.
   * slope : float, optional, default=0.25. Init slope for the activation. (For leaky and elu only)
   * lower_bound : float, optional, default=0.125. Lower bound of random slope. (For rrelu only)
   * upper_bound : float, optional, default=0.334. Upper bound of random slope. (For rrelu only)
   */
  def LeakyReLU(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("LeakyReLU", name, attr)
  }

  /**
   * Apply convolution to input then add a bias.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to the ConvolutionOp.
   * alpha : float, optional, default=0.0001,
   *         value of the alpha variance scaling parameter in the normalization formula
   * beta : float, optional, default=0.75,
   *        value of the beta power parameter in the normalization formula
   * knorm : float, optional, default=2, value of the k parameter in normalization formula
   * nsize : int (non-negative), required, normalization window width in elements.
   */
  def LRN(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("LRN", name, attr)
  }

  /**
   * Use mean absolute error regression for final output, this is used on final output of a net.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to function.
   * label : Symbol. Input label to function.
   * grad_scale : float, optional, default=1. Scale the gradient by a float factor
   */
  def MAERegressionOutput(name: String = null,
                          attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("MAERegressionOutput", name, attr)
  }

  /**
   * Reshape input to target shape
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to  reshape.
   * target_shape : Shape(tuple), required. Target new shape. One and only one dim can be 0,
   *                in which case it will be infered from the rest of dims
   * note
   * ---------
   * (neg_idx) < (0) One and only one dim can be inferenced,such as -1
   *    * example
   * val inputs = Symbol.Reshape()(Map("data" -> label, "shape" -> "(-1,-1,6)"))
   * if the shape of lhs and rhs are both (10,3,2)
   * dim = -1  => auto set this dimension
	dim = 0  => delete this dimension
	dim = 1  => set 1
	dim = 2  => set 2
   * 
   */
  def Reshape(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Reshape", name, attr)
  }

  /**
   * Slice channel into many outputs with equally divided channel
   *
   * Parameters
   * ----------
   * num_outputs : int, required. Number of outputs to be sliced.
   * 
   * note:
   *  when error come with that:new and old shape do not match total elements,please add "axis" 
   * data3_slice = mx.symbol.SliceChannel(data = data_sym3, num_outputs=5, axis=0)
   */
  def SliceChannel(name: String = null, attr: Map[String, String] = null)(
                   inputs: Array[Symbol], params: Map[String, Any] = null): Symbol = {
    createFromListedSymbolsNoCheck("SliceChannel", name, attr)(inputs, params)
  }

  /**
   * Apply softmax activation to input.
   * This is intended for internal layers. For output (loss layer) please use SoftmaxOutput.
   * If type=instance,
   * this operator will compute a softmax for each instance in the batch; this is the default mode.
   * If type=channel,
   * this operator will compute a num_channel-class softmax at each position of each instance;
   * this can be used for fully convolutional network, image segmentation, etc.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to activation function.
   * type : {'channel', 'instance'},optional, default='instance'. Softmax Mode.
   *        If set to instance,
   *        this operator will compute a softmax for each instance in the batch;
   *        this is the default mode.
   *        If set to channel,
   *        this operator will compute a num_channel-class softmax
   *        at each position of each instance;
   *        this can be used for fully convolutional network, image segmentation, etc.
   */
  def SoftmaxActivation(name: String = null,
                        attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("SoftmaxActivation", name, attr)
  }

  /**
   * Apply matrix multiplication to input then add a bias.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to the FullyConnectedOp.
   * weight : Symbol. Weight matrix.
   * bias : Symbol. Bias parameter.
   * num_hidden : int, required. Number of hidden nodes of the output.
   * no_bias : boolean, optional, default=False. Whether to disable bias parameter.
   */
  def FullyConnected(name: String = null,
                     attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("FullyConnected", name, attr)
  }

  /**
   * Apply activation function to input.
   * Softmax Activation is only available with CUDNN on GPUand will be computed
   * at each location across channel if input is 4D.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to activation function.
   * act_type : {'relu', 'sigmoid', 'softrelu', 'tanh'}, required.
   *            Activation function to be applied.
   */
  def Activation(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Activation", name, attr)
  }

  /**
   * Apply convolution to input then add a bias.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to the ConvolutionOp.
   * weight : Symbol. Weight matrix.
   * bias : Symbol. Bias parameter.
   * kernel : Shape(tuple), required. Convolution kernel size: (y, x)
   * stride : Shape(tuple), optional, default=(1, 1). Convolution stride: (y, x)
   * dilate : Shape(tuple), optional, default=(1, 1). Convolution dilate: (y, x)
   * pad : Shape(tuple), optional, default=(0, 0). Pad for convolution: (y, x)
   * num_filter : int (non-negative), required. Convolution filter(channel) number
   * num_group : int (non-negative), optional, default=1
   *             Number of groups partition.
   *             This option is not supported by CuDNN,
   *             you can use SliceChannel to num_group,
   *             apply convolution and concat instead to achieve the same need.
   * workspace : long (non-negative), optional, default=512. Tmp workspace for convolution (MB).
   * no_bias : boolean, optional, default=False. Whether to disable bias parameter.
   * 
   * 
   */
  def Convolution(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Convolution", name, attr)
  }

  /**
   * Apply deconvolution to input then add a bias.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to the DeconvolutionOp.
   * weight : Symbol. Weight matrix.
   * bias : Symbol. Bias parameter.
   * kernel : Shape(tuple), required, deconvolution kernel size: (y, x)
   * stride : Shape(tuple), optional, default=(1, 1), deconvolution stride: (y, x)
   * pad : Shape(tuple), optional, default=(0, 0), pad for deconvolution: (y, x)
   * num_filter : int (non-negative), required, deconvolution filter(channel) number
   * num_group : int (non-negative), optional, default=1, number of groups partition
   * workspace : long (non-negative), optional, default=512. Tmp workspace for deconvolution (MB)
   * no_bias : boolean, optional, default=True. Whether to disable bias parameter.
   */
  def Deconvolution(name: String = null,
                    attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Deconvolution", name, attr)
  }

  /**
   * Perform spatial pooling on inputs.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to the pooling operator.
   * kernel : Shape(tuple), required, pooling kernel size: (y, x)
   * pool_type : {'avg', 'max', 'sum'}, required. Pooling type to be applied.
   * stride : Shape(tuple), optional, default=(1, 1), stride for pooling (y, x)
   * pad : Shape(tuple), optional, default=(0, 0), pad for pooling: (y, x)
   */
  def Pooling(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Pooling", name, attr)
  }

  /**
   * Flatten input
   * Parameters
   * ----------
   * data : Symbol. Input data to flatten.
   * 
   * example: if input(batchSize,a,b,c)
   * output:  (batchSize,a*b*c)
   * 
   */
  def Flatten(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Flatten", name, attr)
  }

  /**
   * Perform a softmax transformation on input, backprop with logloss.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to softmax.
   * label : Symbol. Label data.
   * grad_scale : float, optional, default=1. Scale the gradient by a float factor
   * ignore_label : float, optional, default=-1.
   *                the ignore_label will not work in backward,
   *                and this onlybe used when multi_output=true
   * multi_output : boolean, optional, default=False.
   *                If set to true, for a (n,k,x_1,..,x_n) dimensionalinput tensor,
   *                softmax will generate n*x_1*...*x_n output, eachhas k classes
   * use_ignore : boolean, optional, default=False.
   *              If set to true,
   *              the ignore_label value will not contributorto the backward gradient
   */
  def SoftmaxOutput(name: String = null,
                    attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("SoftmaxOutput", name, attr)
  }

  /**
   * Cast array to a different data type.
   * Parameters
   * ----------
   * data : Symbol, Input data to cast function.
   * dtype : {Int, Double, Short, Float}, required, Target data type.
   */
  def Cast(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Cast", name, attr)
  }

  /**
   * Perform an elementwise sum over all the inputs.
   *
   * Parameters
   * ----------
   * num_args : int, required. Number of inputs to be sum.
   */
  def ElementWiseSum(name: String = null,
                     attr: Map[String, String] = null)(
                     symbols: Array[Symbol], params: Map[String, Any] = null): Symbol = {
    createFromListedSymbolsNoCheck("ElementWiseSum", name, attr)(symbols, params)
  }


   
  /**
   * Apply batch normalization to input.
   *
   * Parameters
   * ----------
   * data : Symbol, Input data to batch normalization
   * eps : float, optional, default=0.001, Epsilon to prevent div 0
   * momentum : float, optional, default=0.9, Momentum for moving average
   * fix_gamma : boolean, optional, default=True, Fix gamma while training
   */
  def BatchNorm(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("BatchNorm", name, attr)
  }

  /**
   * Perform nearest neighbor/bilinear up sampling to inputs
   *
   * Parameters
   * ----------
   * data : Symbol[]. Array of tensors to upsample
   * scale : int (non-negative), required. Up sampling scale
   * num_filter : int (non-negative), optional, default=0.
   *              Input filter. Only used by nearest sample_type.
   * sample_type : {'bilinear', 'nearest'}, required, upsampling method
   * multi_input_mode : {'concat', 'sum'},optional, default='concat'
   *                    How to handle multiple input.
   *                    concat means concatenate upsampled images along the channel dimension.
   *                    sum means add all images together,
   *                    only available for nearest neighbor upsampling.
   * num_args : int, required. Number of inputs to be upsampled.
   *            For nearest neighbor upsampling, this can be 1-N;
   *            the size of output will be(scale*h_0,scale*w_0)
   *            and all other inputs will be upsampled to thesame size.
   *            For bilinear upsampling this must be 2; 1 input and 1 weight.
   */
  def UpSampling(name: String = null, attr: Map[String, String] = null)(
                 inputs: Array[Symbol], params: Map[String, Any] = null): Symbol = {
    createFromListedSymbolsNoCheck("UpSampling", name, attr)(inputs, params)
  }

  /**
   * Perform an feature concat on channel dim (dim 1) over all the inputs.
   *
   * Parameters
   * ----------
   * data : Symbol[]. List of tensors to concatenate
   * num_args : int, required. Number of inputs to be concated.
   * dim : int, optional, default='1'. the dimension to be concated.
   * 
   * example
   * val concat0=Symbol.Concat("concat0")(Array(lhs,rhs),Map("dim"->0))
   * if the shape of lhs and rhs are both (10,3,2)
	dim = 0  => (20,3,2)
	dim = 1  => (10,6,2)
	dim = 2  => (10,3,4)
   * 
   * 
   */
  def Concat(name: String = null, attr: Map[String, String] = null)(
             inputs: Array[Symbol], params: Map[String, Any] = null): Symbol = {
    createFromListedSymbolsNoCheck("Concat", name, attr)(inputs, params)
  }

  /**
   * Use Logistic regression for final output, this is used on final output of a net.
   * Logistic regression is suitable for binary classification or probability prediction tasks.
   * Parameters
   * ----------
   * data : Symbol. Input data to function.
   * label : Symbol. Input label to function.
   * grad_scale : float, optional, default=1. Scale the gradient by a float factor
   */
  def LogisticRegressionOutput(name: String = null,
                               attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("LogisticRegressionOutput", name, attr)
  }

  /**
   * Use linear regression for final output, this is used on final output of a net.
   * Parameters
   * ----------
   * data : Symbol. Input data to function.
   * label : Symbol. Input label to function.
   * grad_scale : float, optional, default=1. Scale the gradient by a float factor
   * 
   * note:
   * E = \frac{1}{2N}*\sum_{i,j}(x_{i,j}-label_{i,j})
   * 
   */
  def LinearRegressionOutput(name: String = null,
                             attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("LinearRegressionOutput", name, attr)
  }

  /**
   * Apply swapaxis to input.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to the SwapAxisOp.
   * dim1 : int (non-negative), default=0, the first axis to be swapped.
   * dim2 : int (non-negative), default=0, the second axis to be swapped.
   */
  def SwapAxis(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("SwapAxis", name, attr)
  }

  /**
   * Get embedding for one-hot input
   *
   * Parameters
   * ----------
   * data : Symbol, Input data to the EmbeddingOp.
   * weight : Symbol, Embedding weight matrix.
   * input_dim : int, input dim of one-hot encoding
   * output_dim : int, output dim of embedding
   */
  def Embedding(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("Embedding", name, attr)
  }

  /**
   * Perform Smooth L1 on inputs.
   *
   * Parameters
   * ----------
   * data : Symbol. Input data to the smooth_l1 operator.
   * scalar : Float, required.
   */
  def SmoothL1(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("smooth_l1", name, attr)
  }

  /**
   * Special layer for propagating loss
   *
   * Parameters
   * ----------
   * data : Symbol, Input data to the MakeLossOp.
   * grad_scale : float, optional, default=1.
   *            Gradient scale as a supplement to unary and binary operators
   */
  def MakeLoss(name: String = null, attr: Map[String, String] = null): SymbolCreateNamedFunc = {
    createFromNamedSymbolsNoCheck("MakeLoss", name, attr)
  }

  /**
   * by liuxianggen
   * 20160825
   * there are to steps:
   * 1.softmax
   * 2.sum{log(p{label(i)})}
   * Calculate cross_entropy(lhs, one_hot(rhs))
	Parameters
	----------
	lhs : Symbol
	Left symbolic input to the function
	rhs : Symbol
	Right symbolic input to the function
   */
  def Softmax_cross_entropy(left: Symbol, right: Symbol): Symbol = {
    createFromListedSymbols("softmax_cross_entropy")(Array(left, right))
  }
  
  /**
   * Create a symbol that groups symbols together.
   * @param symbols List of symbols to be grouped.
   * @return The created group symbol.
   */
//  def Group(symbols: Symbol*): Symbol = {
//    val ihandles = symbols.map(_.handle).toArray
//    val handle = new SymbolHandleRef
//    checkCall(_LIB.mxSymbolCreateGroup(ihandles, handle))
//    new Symbol(handle.value)
//  }

  // List and add all the atomic symbol functions to current module.
  private def initSymbolModule(): Map[String, SymbolFunction] = {
    val symbolList = ListBuffer.empty[SymbolHandle]
    checkCall(_LIB.mxSymbolListAtomicSymbolCreators(symbolList))
    symbolList.map(makeAtomicSymbolFunction).toMap
  }

  // Create an atomic symbol function by handle and function name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle): (String, SymbolFunction) = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new MXUintRef
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    checkCall(_LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs))
    val paramStr = ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
//    println("Atomic Symbol function defination:\n{}", docStr)
    (name.value, new SymbolFunction(handle, keyVarNumArgs.value))
  }

  /**
   * Activation Operator of Neural Net.
   * The parameters listed below can be passed in as keyword arguments.
   * @param symbols Symbol parameters passed to create the resulting symbol
   * @param paramKwargs Key-value parameters passed to create the resulting symbol
   * @param attr Attributes set to the resulting symbol
   * @return the resulting symbol
   */
  
  def createFromListedSymbols(
      operator: String, name: String = null, attr: Map[String, String] = null)(
      symbols: Array[Symbol], paramKwargs: Map[String, String] = null): Symbol = {
    
	  val function = functions(operator)
      require(function != null, s"invalid operator name $operator")
  
      val params = if (paramKwargs == null) Map.empty[String, String] else paramKwargs
      
      //the group of operational functions contains the special operator =>1
      val addkeyVarNumArgs = (function.keyVarNumArgs != null
        && !function.keyVarNumArgs.isEmpty
        && !params.contains(function.keyVarNumArgs))
  
     val params1: scala.collection.mutable.Map[String, String] = (
          if (addkeyVarNumArgs) scala.collection.mutable.Map[String,String](function.keyVarNumArgs->symbols.length.toString)
          else scala.collection.mutable.Map[String,String]()
        ) ++ params
        
        
      val s = Create(operator,params1.toMap)
      val attrAll = AttrScope.current.get(Option(attr))
      s.setAttr(attrAll)
      val hint = operator.toLowerCase
      val managedName = NameManager.current.get(Option(name), hint)
      s.Compose(symbols,managedName)
      s
  }

  /**
   * Activation Operator of Neural Net.
   * The parameters listed below can be passed in as keyword arguments.
   * @param symbols Named symbol parameters passed to create the resulting symbol
   * @param paramKwargs Key-value parameters passed to create the resulting symbol
   * @param attr Attributes set to the resulting symbol
   * @return the resulting symbol
   */
  def createFromNamedSymbols(
      operator: String, name: String = null, attr: Map[String, String] = null)(
      symbols: Map[String, Symbol], paramKwargs: Map[String, String] = null): Symbol = {
    val function = functions(operator)
    require(function != null, s"invalid operator name $operator")
    
    //check the keyVarNumArgs, if not null, get wrong
    require(function.keyVarNumArgs == null || function.keyVarNumArgs.isEmpty,
      "This function support variable length of Symbol arguments.\n" +
      "Please pass all the input Symbols via positional arguments instead of keyword arguments.")
   
   	val params = if (paramKwargs == null) Map.empty[String, String] else paramKwargs
   	val s = Create(operator,params)
    val attrAll = AttrScope.current.get(Option(attr))
    s.setAttr(attrAll)
    val hint = operator.toLowerCase
    val managedName = NameManager.current.get(Option(name), hint)
    s.Compose(symbols,managedName)
    s
  }

  // a more friendly interface for creating symbols
  // all values except symbols in kwargs will be cast to String using its toString() method
  def createFromNamedSymbolsNoCheck(
      operator: String, name: String = null, attr: Map[String, String] = null)(
      kwargs: Map[String, Any]): Symbol = {
    val symbolArgs = kwargs.filter { case (key, value) =>
      value.isInstanceOf[Symbol]
    }.map { case (key, value) =>
      (key, value.asInstanceOf[Symbol])
    }
    val strArgs = kwargs.filter { case (key, value) =>
      !value.isInstanceOf[Symbol]
    }.map { case (key, value) =>
      (key, value.toString)
    }
    createFromNamedSymbols(operator, name, attr)(symbolArgs, strArgs)
  }

  // a more friendly interface for creating symbols
  // all values except symbols in kwargs will be cast to String using its toString() method
  def createFromListedSymbolsNoCheck(
       operator: String, name: String = null, attr: Map[String, String] = null)(
       symbols: Array[Symbol], kwargs: Map[String, Any] = null): Symbol = {
    val args =
      if (kwargs == null) null
      else kwargs.map { case (key, value) => (key, value.toString) }
    createFromListedSymbols(operator, name, attr)(symbols, args)
  }

  /**
   * Helper function to get ndarray lists handles from various inputs.
   * @param argKey The name of argument, used for error message.
   * @param args list of NDArray or dict of str to NDArray
   *             Input arguments to the symbols.
   *             If type is list of NDArray, the position is in the same order of arg_names.
   *             If type is dict of str to NDArray, then it maps the name of arguments
   *             to the corresponding NDArray
   * @param argNames List of argument names.
   * @param allowMissing Whether missing argument is allowed.
   *                     When allowed, the missing handle will be set to None(null)
   * @return The positional list of NDArrayHandles generated from input.
   */
  private def getNDArrayInputs(argKey: String, args: Seq[NDArray], argNames: Seq[String],
                               allowMissing: Boolean): (Array[NDArrayHandle], Array[NDArray]) = {
    require(args.length == argNames.length, s"Length of $argKey do not match number of arguments")
    val argHandles = args.map(_.handle)
    (argHandles.toArray, args.toArray)
  }

  private def getNDArrayInputs(argKey: String, args: Map[String, NDArray], argNames: Seq[String],
                               allowMissing: Boolean): (Array[NDArrayHandle], Array[NDArray]) = {
    val argArrays = ArrayBuffer.empty[NDArray]
    val argHandles = ArrayBuffer.empty[NDArrayHandle]
    argNames.foreach { name =>
      args.get(name) match {
        case narr: Some[NDArray] =>
          argArrays += narr.get
          argHandles += narr.get.handle
        case None =>
          require(allowMissing, s"Must specify all the arguments in $argKey")
          argArrays += null
          argHandles += 0L
      }
    }
    (argHandles.toArray, argArrays.toArray)
  }

  /**
   * Load symbol from a JSON file.
   *
   * You can also use pickle to do the job if you only work on python.
   * The advantage of load/save is the file is language agnostic.
   * This means the file saved using save can be loaded by other language binding of brainmatrix.
   * You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
   *
   * @param fname The name of the file, examples:
   *        - `s3://my-bucket/path/my-s3-symbol`
   *        - `hdfs://my-bucket/path/my-hdfs-symbol`
   *        - `/path-to/my-local-symbol`
   * @return The loaded symbol.
   * @see Symbol.save : Used to save symbol into file.
   */
  def load(fname: String): Symbol = {
    val handle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateFromFile(fname, handle))
    new Symbol(handle.value)
  }

  /**
   * Load symbol from json string.
   * @param json A json string.
   * @return The loaded symbol.
   * @see Symbol.tojson : Used to save symbol into json string.
   */
  def loadJson(json: String): Symbol = {
    val handle = new SymbolHandleRef
    checkCall(_LIB.mxSymbolCreateFromJSON(json, handle))
    new Symbol(handle.value)
  }
}

private case class SymbolFunction(handle: SymbolHandle, keyVarNumArgs: String)

object SymbolConversions {
  implicit def int2Scalar(x: Int): SymbolConversions[Int] = new SymbolConversions(x)
  implicit def double2Scalar(x: Double): SymbolConversions[Double] = new SymbolConversions(x)
  implicit def float2Scalar(x: Float): SymbolConversions[Float] = new SymbolConversions(x)
}

class SymbolConversions[@specialized(Int, Float, Double) V](val value: V) {
  def +(other: Symbol): Symbol = {
    other + value
  }

  def -(other: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_RMinusScalar")(
      Array(other), Map("scalar" -> value.toString))
  }

  def *(other: Symbol): Symbol = {
    other + value
  }

  def /(other: Symbol): Symbol = {
    Symbol.createFromListedSymbols("_RDivScalar")(
      Array(other), Map("scalar" -> value.toString))
  }
}


class NodeRef { var value: Node = null }
class DataEntryRef { var value: DataEntry = null }
class OperatorPropertyRef { var value: OperatorProperty = null }
class MapRef { var value: Map[String, String] = null }


class Node(val opRef: OperatorPropertyRef, var name: String = null) {
	// brief Operator of this node
	//var op:OperatorProperty
	// brief name of the node
	//var name:String
	// brief inputs to this node

	/**
	 * as a struct, initialization is very important
	 */
	var inputs: Vector[DataEntry] = Vector[DataEntry]()
	var backward_source_node: NodeRef = new NodeRef()
	var attr: scala.collection.mutable.Map[String,String] = scala.collection.mutable.Map()
	var backward_source_id: Int = -1

	def is_atomic: Boolean = {
		return (inputs.length == 0 && opRef.value != null)
	}

	def is_variable(): Boolean = {
//		println(this.name)
//		println("1"+opRef.value)
//		println("2"+backward_source_node.value)
		return (opRef.value == null && this.backward_source_node.value == null)
	}

	def is_backward(): Boolean = {
		//if there is backward node
		return (backward_source_node.value != null)
	}
	
	def reset_inputs(){
	  this.inputs = Vector[DataEntry]()
	}

}

class DataEntry(var source: NodeRef, var index: Int) {
	var source_id: Int = -1

	//brief the source of the node of this data
	//      val source:NodeRef
	//brief index of output from the source
	//      val index:Int
	
	def Info:String = {
		var s = "\tDataEntry:"+index 
		if(source.value != null)
			s += "\n node name:" + source.value.name
		if(this.source_id != -1)
			s += "\n source_id:" + this.source_id
		s
	}
}

