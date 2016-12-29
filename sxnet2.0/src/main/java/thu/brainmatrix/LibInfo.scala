package thu.brainmatrix

import thu.brainmatrix.Base._

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
 * JNI functions
 * @author Yizhi Liu
 */
class LibInfo {
  @native def nativeLibInit(): Int
  // NDArray
  @native def mxNDArrayFree(handle: NDArrayHandle): Int
  @native def mxGetLastError(): String
  @native def mxNDArrayCreateNone(out: NDArrayHandleRef): Int
  @native def mxNDArrayCreate(shape: Array[Int],
                              ndim: Int,
                              devType: Int,
                              devId: Int,
                              delayAlloc: Int,
                              out: NDArrayHandleRef): Int
  @native def mxNDArrayWaitAll(): Int
  @native def mxNDArrayWaitToRead(handle: NDArrayHandle): Int
  @native def mxListFunctions(functions: ListBuffer[FunctionHandle]): Int
  @native def mxFuncDescribe(handle: FunctionHandle,
                             nUsedVars: MXUintRef,
                             nScalars: MXUintRef,
                             nMutateVars: MXUintRef,
                             typeMask: Base.RefInt): Int
  @native def mxFuncGetInfo(handle: FunctionHandle,
                            name: RefString,
                            desc: RefString,
                            numArgs: MXUintRef,
                            argNames: ListBuffer[String],
                            argTypes: ListBuffer[String],
                            argDescs: ListBuffer[String]): Int
  @native def mxFuncInvoke(function: FunctionHandle,
                           useVars: Array[NDArrayHandle],
                           scalarArgs: Array[MXFloat],
                           mutateVars: Array[NDArrayHandle]): Int
  @native def mxFuncInvokeEx(function: FunctionHandle,
                             useVars: Array[NDArrayHandle],
                             scalarArgs: Array[MXFloat],
                             mutateVars: Array[NDArrayHandle],
                             numParams: Int,
                             paramKeys: Array[Array[Byte]],
                             paramVals: Array[Array[Byte]]): Int
  @native def mxNDArrayGetShape(handle: NDArrayHandle,
                                ndim: MXUintRef,
                                data: ArrayBuffer[Int]): Int
  @native def mxNDArraySyncCopyToCPU(handle: NDArrayHandle,
                                     data: Array[MXFloat],
                                     size: Int): Int
  @native def mxNDArraySlice(handle: NDArrayHandle,
                             start: MXUint,
                             end: MXUint,
                             sliceHandle: NDArrayHandleRef): Int
  @native def mxNDArrayReshape(handle: NDArrayHandle,
                               nDim: Int,
                               dims: Array[Int],
                               reshapeHandle: NDArrayHandleRef): Int
  @native def mxNDArraySyncCopyFromCPU(handle: NDArrayHandle,
                                       source: Array[MXFloat],
                                       size: Int): Int
  @native def mxNDArrayLoad(fname: String,
                            outSize: MXUintRef,
                            handles: ArrayBuffer[NDArrayHandle],
                            outNameSize: MXUintRef,
                            names: ArrayBuffer[String]): Int
  @native def mxNDArraySave(fname: String,
                            handles: Array[NDArrayHandle],
                            keys: Array[String]): Int
  @native def mxNDArrayGetContext(handle: NDArrayHandle, devTypeId: Base.RefInt, devId: Base.RefInt): Int
  @native def mxNDArraySaveRawBytes(handle: NDArrayHandle, buf: ArrayBuffer[Byte]): Int
  @native def mxNDArrayLoadFromRawBytes(bytes: Array[Byte], handle: NDArrayHandleRef): Int

  // KVStore Server
  @native def mxInitPSEnv(keys: Array[String], values: Array[String]): Int
  @native def mxKVStoreRunServer(handle: KVStoreHandle, controller: KVServerControllerCallback): Int

  // KVStore
  @native def mxKVStoreCreate(name: String, handle: KVStoreHandleRef): Int
  @native def mxKVStoreInit(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            values: Array[NDArrayHandle]): Int
  @native def mxKVStorePush(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            values: Array[NDArrayHandle],
                            priority: Int): Int
  @native def mxKVStorePull(handle: KVStoreHandle,
                            len: MXUint,
                            keys: Array[Int],
                            outs: Array[NDArrayHandle],
                            priority: Int): Int
  @native def mxKVStoreSetUpdater(handle: KVStoreHandle, updaterFunc: MXKVStoreUpdater): Int
  @native def mxKVStoreIsWorkerNode(isWorker: RefInt): Int
  @native def mxKVStoreGetType(handle: KVStoreHandle, kvType: RefString): Int
  @native def mxKVStoreSendCommmandToServers(handle: KVStoreHandle,
                                             head: Int, body: String): Int
  @native def mxKVStoreBarrier(handle: KVStoreHandle): Int
  @native def mxKVStoreGetGroupSize(handle: KVStoreHandle, size: RefInt): Int
  @native def mxKVStoreGetRank(handle: KVStoreHandle, size: RefInt): Int
  @native def mxKVStoreFree(handle: KVStoreHandle): Int

  // DataIter Funcs
  @native def mxListDataIters(handles: ListBuffer[DataIterCreator]): Int
  @native def mxDataIterCreateIter(handle: DataIterCreator,
                                   keys: Array[String],
                                   vals: Array[String],
                                   out: DataIterHandleRef): Int
  @native def mxDataIterGetIterInfo(creator: DataIterCreator,
                                    name: RefString,
                                    description: RefString,
                                    argNames: ListBuffer[String],
                                    argTypeInfos: ListBuffer[String],
                                    argDescriptions: ListBuffer[String]): Int
  @native def mxDataIterFree(handle: DataIterHandle): Int
  @native def mxDataIterBeforeFirst(handle: DataIterHandle): Int
  @native def mxDataIterNext(handle: DataIterHandle, out: RefInt): Int
  @native def mxDataIterGetLabel(handle: DataIterHandle,
                                 out: NDArrayHandleRef): Int
  @native def mxDataIterGetData(handle: DataIterHandle,
                                out: NDArrayHandleRef): Int
  @native def mxDataIterGetIndex(handle: DataIterHandle,
                                outIndex: ListBuffer[Long],
                                outSize: RefLong): Int
  @native def mxDataIterGetPadNum(handle: DataIterHandle,
                                  out: MXUintRef): Int
  // Executors
  @native def mxExecutorOutputs(handle: ExecutorHandle, outputs: ArrayBuffer[NDArrayHandle]): Int
  @native def mxExecutorFree(handle: ExecutorHandle): Int
  @native def mxExecutorForward(handle: ExecutorHandle, isTrain: Int): Int
  @native def mxExecutorBackward(handle: ExecutorHandle,
                                 grads: Array[NDArrayHandle]): Int
  @native def mxExecutorPrint(handle: ExecutorHandle, debugStr: RefString): Int
  @native def mxExecutorSetMonitorCallback(handle: ExecutorHandle, callback: MXMonitorCallback): Int

  // Symbols
  @native def mxSymbolListAtomicSymbolCreators(symbolList: ListBuffer[SymbolHandle]): Int
  @native def mxSymbolGetAtomicSymbolInfo(handle: SymbolHandle,
                                          name: RefString,
                                          desc: RefString,
                                          numArgs: MXUintRef,
                                          argNames: ListBuffer[String],
                                          argTypes: ListBuffer[String],
                                          argDescs: ListBuffer[String],
                                          keyVarNumArgs: RefString): Int
  @native def mxSymbolCreateAtomicSymbol(handle: SymbolHandle,
                                         paramKeys: Array[String],
                                         paramVals: Array[String],
                                         symHandleRef: SymbolHandleRef): Int
  @native def mxSymbolSetAttr(handle: SymbolHandle, key: String, value: String): Int
  @native def mxSymbolCompose(handle: SymbolHandle,
                              name: String,
                              keys: Array[String],
                              args: Array[SymbolHandle]): Int
  @native def mxSymbolCreateVariable(name: String, out: SymbolHandleRef): Int
  @native def mxSymbolGetAttr(handle: SymbolHandle,
                              key: String,
                              ret: RefString,
                              success: RefInt): Int
  @native def mxSymbolListArguments(handle: SymbolHandle,
                                    arguments: ArrayBuffer[String]): Int
  @native def mxSymbolCopy(handle: SymbolHandle, clonedHandle: SymbolHandleRef): Int
  @native def mxSymbolListAuxiliaryStates(handle: SymbolHandle,
                                          arguments: ArrayBuffer[String]): Int
  @native def mxSymbolListOutputs(handle: SymbolHandle,
                                  outputs: ArrayBuffer[String]): Int
  @native def mxSymbolCreateGroup(handles: Array[SymbolHandle], out: SymbolHandleRef): Int
  @native def mxSymbolPrint(handle: SymbolHandle, str: RefString): Int
  @native def mxSymbolGetInternals(handle: SymbolHandle, out: SymbolHandleRef): Int
  @native def mxSymbolInferType(handle: SymbolHandle,
                                keys: Array[String],
                                sdata: Array[Int],
                                argTypeData: ListBuffer[Int],
                                outTypeData: ListBuffer[Int],
                                auxTypeData: ListBuffer[Int],
                                complete: RefInt): Int
  @native def mxSymbolInferShape(handle: SymbolHandle,
                                 numArgs: MXUint,
                                 keys: Array[String],
                                 argIndPtr: Array[MXUint],
                                 argShapeData: Array[MXUint],
                                 inShapeData: ListBuffer[Array[Int]],
                                 outShapeData: ListBuffer[Array[Int]],
                                 auxShapeData: ListBuffer[Array[Int]],
                                 complete: RefInt): Int
  @native def mxSymbolGetOutput(handle: SymbolHandle, index: Int, out: SymbolHandleRef): Int
  @native def mxSymbolSaveToJSON(handle: SymbolHandle, out: RefString): Int
  @native def mxSymbolCreateFromJSON(json: String, handle: SymbolHandleRef): Int
  // scalastyle:off parameterNum
  @native def mxExecutorBindX(handle: SymbolHandle,
                              deviceTypeId: Int,
                              deviceID: Int,
                              numCtx: Int,
                              ctxMapKeys: Array[String],
                              ctxMapDevTypes: Array[Int],
                              ctxMapDevIDs: Array[Int],
                              numArgs: Int,
                              argsHandle: Array[NDArrayHandle],
                              argsGradHandle: Array[NDArrayHandle],
                              reqsArray: Array[Int],
                              auxArgsHandle: Array[NDArrayHandle],
                              out: ExecutorHandleRef): Int
  @native def mxExecutorBindEX(handle: SymbolHandle,
                              deviceTypeId: Int,
                              deviceID: Int,
                              numCtx: Int,
                              ctxMapKeys: Array[String],
                              ctxMapDevTypes: Array[Int],
                              ctxMapDevIDs: Array[Int],
                              numArgs: Int,
                              argsHandle: Array[NDArrayHandle],
                              argsGradHandle: Array[NDArrayHandle],
                              reqsArray: Array[Int],
                              auxArgsHandle: Array[NDArrayHandle],
                              sharedExec: ExecutorHandle,
                              out: ExecutorHandleRef): Int
  // scalastyle:on parameterNum
  @native def mxSymbolSaveToFile(handle: SymbolHandle, fname: String): Int
  @native def mxSymbolCreateFromFile(fname: String, handle: SymbolHandleRef): Int
  @native def mxSymbolFree(handle: SymbolHandle): Int

  // Random
  @native def mxRandomSeed(seed: Int): Int

  @native def mxNotifyShutdown(): Int
  
   /**
   *  by liuxianggen 
   *  2016-3-9
   */
  @native def mxScalaOpListArguments(handle: SymbolHandle,arguments: ArrayBuffer[String]):Int
  
  /**
   *  by liuxianggen 
   *  2016-4-9
   */
  @native def mxScalaOpListAuxiliaryStates(handle: SymbolHandle,arguments: ArrayBuffer[String]):Int
  
  
  /**
   * by liuxianggen
   * 2016-3-9
   */
  @native def mxScalaOpInit(handle:OperatorPropertyHandle,
  		paramKeys: Array[String],paramVals: Array[String]):Int
  		
  @native def mxScalaOpPrintParam(handle:OperatorPropertyHandle):Int
  
  @native def mxScalaCreateOperatorProperty(handle:ScalaSymbolHandle,opHandleRef:OperatorPropertyHandleRef):Int
  
    /**
     * @author liuxianggen
     * @date 20160707
     * @brief get the return of NumVisibleOutputs on op
     * @param OperatorPropertyHandle
     * @param MXUintRef
     * @return the NumVisibleOutputs
     * @note
     */
  @native def mxScalaOpNumVisibleOutputs(handle:OperatorPropertyHandle,num: MXUintRef):Int
  
  
//  @native def mxScalaSymbolInferShape(handle: ScalaSymbolHandle,
//                                 numArgs: MXUint,
//                                 keys: Array[String],
//                                 argIndPtr: Array[MXUint],
//                                 argShapeData: Array[MXUint],
//                                 inShapeData: ListBuffer[Array[Int]],
//                                 outShapeData: ListBuffer[Array[Int]],
//                                 auxShapeData: ListBuffer[Array[Int]],
//                                 complete: RefInt): Int
  @native def mxScalaOPCopy(handle:OperatorPropertyHandle,opHandleRef:OperatorPropertyHandleRef):Int
 
  @native def mxScalaToStaticGraph(handleref:StaticGraphHandleRef,arg_node_sg:Array[Int],heads_source:Array[Int],heads_index:Array[Int],nods_opHandles:Array[OperatorPropertyHandle],nods_name_len:Int,nods_name:Array[String],
  	 nods_inputs_len_arr:Array[Int] ,nods_inputs_source_ids:Array[Int],nods_inputs_indexs:Array[Int],nods_backward_source_ids:Array[Int],nods_attr_len_arr:Array[Int],nods_attr_len_arr_len:Int,nods_attrs_keys:Array[String],nods_attrs_values:Array[String]):Int
  	

    /**
     * @author liuxianggen
     * @date 20160724
     * @brief all the global information are listed in there
     * @param handle:the id of StaticGraph
     * @param num_arg_nodes: the number of all the arg_nodes,which are always variable
     * @param numArgs: the number of input args_node
     * @param keys: a array which contains the id of the input arg_nodes
     * @param argIndPtr:  a array which contains the shape size of the input arg_nodes, in the conventional order
     * @param argShapeData:  a array which contains the shape of the input arg_nodes, in the conventional order. 
     * @param inShapeData: the input shape of a symbol, written by jni
     * @param outShapeData: the output shape of a symbol , written by jni
     * @param auxShapeData: the auxiliary shape of a symbol , written by jni
     * @param complete: a flag , written by jni
     * @return
     * @example
     * @note
     */
    @native def mxScalaSGInferShape(handle:StaticGraphHandle, num_arg_nodes:MXUint, numArgs: MXUint,keys: Array[MXUint],argIndPtr: Array[MXUint],argShapeData: Array[MXUint],
  		inShapeData: ListBuffer[Array[Int]],outShapeData: ListBuffer[Array[Int]],auxShapeData: ListBuffer[Array[Int]],complete: RefInt):Int
  		
  		
  @native def mxScalaExecutorBindX(handle: StaticGraphHandle,
                              deviceTypeId: Int,
                              deviceID: Int,
                              numCtx: Int,
                              ctxMapKeys: Array[String],
                              ctxMapDevTypes: Array[Int],
                              ctxMapDevIDs: Array[Int],
                              numArgs: Int,
                              argsHandle: Array[NDArrayHandle],
                              argsGradHandle: Array[NDArrayHandle],
                              reqsArray: Array[Int],
                              auxArgsHandle: Array[NDArrayHandle],
                              out: ExecutorHandleRef): Int

/**
 * NDArray operators
 * by liuxianggen 
 * 2016-4-4
 * 
 */
  @native def mxNDArrayGetData(handle: NDArrayHandle,data_result: MXFloatRef, index: MXUint): Int //has bug,take care
  @native def mxNDArraySetData(handle: NDArrayHandle,data_source: MXFloat, index: MXUint): Int

  /**
   * by liuxianggen
   * 20160729
   */
  @native def mxScalaSymbolSaveToFile(handle: StaticGraphHandle, fname: String): Int
  @native def mxScalaSymbolCreateFromFile(fname: String, handle: StaticGraphHandleRef): Int
  @native def mxStaticGraphFree(handle: StaticGraphHandle): Int
  
  
}
