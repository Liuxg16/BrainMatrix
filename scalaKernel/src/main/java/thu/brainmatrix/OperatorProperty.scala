package thu.brainmatrix
import thu.brainmatrix.Base._
import scala.collection.mutable.{ListBuffer,ArrayBuffer}
import scala.Vector

/**
 * by liuxianggen
 * 2015-3-3
 * bref like  OperatorProperty in mxnet,provide Operator class function
 * bref OperatorPropertyHandle is the same as SymbolHandle, equals to the atomicSymbol
 */
class OperatorProperty(val handle:OperatorPropertyHandle,val opName:String) {
  
	/*!
	 *  by liuxianggen
	 *  2016-3-9
   *  \brief Initialize the Operator by setting the parameters
   *  This function need to be called before all other functions.
   *  \param kwargs the keyword arguments parameters
   */
	def Init(paramKeys: Array[String],paramVals: Array[String]){
		// call jni operator
		checkCall(_LIB.mxScalaOpInit(handle, paramKeys, paramVals))
	}
	
	/*!
	 * by liuxianggen
	 * 2016-3-9
   * \brief Get a map representation of internal parameters.
   *  This can be used by Init to recover the state of OperatorProperty.
   */
	def printParam(){
		// call jni operator
		checkCall(_LIB.mxScalaOpPrintParam(handle))
	}

	/**
	 * by liuxianggen
	 * 2015-3-4
	 * brief  according to the operator, return the inputs name
	 */
	def ListArguments: Vector[String] = {
		// call jni operator 
		val arr = ArrayBuffer.empty[String]
		checkCall(_LIB.mxScalaOpListArguments(handle, arr))
		val arr_vec = arr.toVector
		arr_vec
	}

	def ListOutputs: Vector[String] = {
		Vector("output")
	}

	def ListAuxiliaryStates():Vector[String] = {
		// call jni operator 
		val arr = ArrayBuffer.empty[String]
		checkCall(_LIB.mxScalaOpListAuxiliaryStates(handle, arr))
		val arr_vec = arr.toVector
		arr_vec
	}
	
	//    def Forward{
	//    	
	//    }

	def Copy(): OperatorProperty = {
		val opHandleRef = new OperatorPropertyHandleRef
		checkCall(_LIB.mxScalaOPCopy(handle, opHandleRef))
		new OperatorProperty(opHandleRef.value,this.opName)
		
	}

	def NumVisibleOutputs(): Int = {
	    val intref= new MXUintRef
	    checkCall(_LIB.mxScalaOpNumVisibleOutputs(this.handle, intref))
	    intref.value
	}
  
}

/*
 * by liuxianggen 
 * 2016-3-20
 * 
 */
object OperatorProperty{
	
	def apply(name:String):OperatorProperty = {
		  val opHandleRef = new OperatorPropertyHandleRef
		  val function = OperatorProperty.initSymbolModule()(name)
      require(function != null, s"invalid operator name opName")
      /*require(function.keyVarNumArgs == null || function.keyVarNumArgs.isEmpty,
      "This function support variable length of Symbol arguments.\n" +
      "Please pass all the input Symbols via positional arguments instead of keyword arguments.")*/
      checkCall(_LIB.mxScalaCreateOperatorProperty(function.handle, opHandleRef))
      new OperatorProperty(opHandleRef.value,name)
	}
      // List and add all the atomic symbol functions to current module.
  private def initSymbolModule(): Map[String, ScalaSymbolFunction] = {
    val symbolList = ListBuffer.empty[SymbolHandle]
    checkCall(_LIB.mxSymbolListAtomicSymbolCreators(symbolList))
    symbolList.map(makeAtomicSymbolFunction).toMap
  }
  
  
    // Create an atomic symbol function by handle and function name.
  private def makeAtomicSymbolFunction(handle: SymbolHandle): (String, ScalaSymbolFunction) = {
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
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n keyVarNumArgs:${keyVarNumArgs.value}"
//    println(docStr)
    (name.value, new ScalaSymbolFunction(handle, keyVarNumArgs.value))
  }
}

private case class ScalaSymbolFunction(handle: ScalaSymbolHandle, keyVarNumArgs: String)