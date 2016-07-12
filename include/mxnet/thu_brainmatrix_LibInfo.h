/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class thu_brainmatrix_LibInfo */

#ifndef _Included_thu_brainmatrix_LibInfo
#define _Included_thu_brainmatrix_LibInfo
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxGetLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_thu_brainmatrix_LibInfo_mxGetLastError
  (JNIEnv *, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayCreateNone
 * Signature: (Lthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayCreateNone
  (JNIEnv *, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayCreate
 * Signature: ([IIIIILthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayCreate
  (JNIEnv *, jobject, jintArray, jint, jint, jint, jint, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayWaitAll
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayWaitAll
  (JNIEnv *, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayWaitToRead
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayWaitToRead
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxListFunctions
 * Signature: (Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxListFunctions
  (JNIEnv *, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxFuncDescribe
 * Signature: (JLthu/brainmatrix/Base/RefInt;Lthu/brainmatrix/Base/RefInt;Lthu/brainmatrix/Base/RefInt;Lthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxFuncDescribe
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxFuncGetInfo
 * Signature: (JLthu/brainmatrix/Base/RefString;Lthu/brainmatrix/Base/RefString;Lthu/brainmatrix/Base/RefInt;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxFuncGetInfo
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxFuncInvoke
 * Signature: (J[J[F[J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxFuncInvoke
  (JNIEnv *, jobject, jlong, jlongArray, jfloatArray, jlongArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayGetShape
 * Signature: (JLthu/brainmatrix/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayGetShape
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArraySyncCopyToCPU
 * Signature: (J[FI)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArraySyncCopyToCPU
  (JNIEnv *, jobject, jlong, jfloatArray, jint);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArraySlice
 * Signature: (JIILthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArraySlice
  (JNIEnv *, jobject, jlong, jint, jint, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArraySyncCopyFromCPU
 * Signature: (J[FI)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArraySyncCopyFromCPU
  (JNIEnv *, jobject, jlong, jfloatArray, jint);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayLoad
 * Signature: (Ljava/lang/String;Lthu/brainmatrix/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;Lthu/brainmatrix/Base/RefInt;Lscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayLoad
  (JNIEnv *, jobject, jstring, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArraySave
 * Signature: (Ljava/lang/String;[J[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArraySave
  (JNIEnv *, jobject, jstring, jlongArray, jobjectArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayGetContext
 * Signature: (JLthu/brainmatrix/Base/RefInt;Lthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayGetContext
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreCreate
 * Signature: (Ljava/lang/String;Lthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreCreate
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreInit
 * Signature: (JI[I[J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreInit
  (JNIEnv *, jobject, jlong, jint, jintArray, jlongArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStorePush
 * Signature: (JI[I[JI)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStorePush
  (JNIEnv *, jobject, jlong, jint, jintArray, jlongArray, jint);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStorePull
 * Signature: (JI[I[JI)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStorePull
  (JNIEnv *, jobject, jlong, jint, jintArray, jlongArray, jint);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreSetUpdater
 * Signature: (JLthu/brainmatrix/MXKVStoreUpdater;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreSetUpdater
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreIsWorkerNode
 * Signature: (Lthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreIsWorkerNode
  (JNIEnv *, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreGetType
 * Signature: (JLthu/brainmatrix/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreGetType
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreSendCommmandToServers
 * Signature: (JILjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreSendCommmandToServers
  (JNIEnv *, jobject, jlong, jint, jstring);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreBarrier
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreBarrier
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreGetGroupSize
 * Signature: (JLthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreGetGroupSize
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxKVStoreGetRank
 * Signature: (JLthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxKVStoreGetRank
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxListDataIters
 * Signature: (Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxListDataIters
  (JNIEnv *, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterCreateIter
 * Signature: (J[Ljava/lang/String;[Ljava/lang/String;Lthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterCreateIter
  (JNIEnv *, jobject, jlong, jobjectArray, jobjectArray, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterGetIterInfo
 * Signature: (JLthu/brainmatrix/Base/RefString;Lthu/brainmatrix/Base/RefString;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterGetIterInfo
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterBeforeFirst
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterBeforeFirst
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterNext
 * Signature: (JLthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterNext
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterGetLabel
 * Signature: (JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterGetLabel
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterGetData
 * Signature: (JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterGetData
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterGetIndex
 * Signature: (JLscala/collection/mutable/ListBuffer;Lthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterGetIndex
  (JNIEnv *, jobject, jlong, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxDataIterGetPadNum
 * Signature: (JLthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxDataIterGetPadNum
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxExecutorOutputs
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxExecutorOutputs
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxExecutorFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxExecutorFree
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxExecutorForward
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxExecutorForward
  (JNIEnv *, jobject, jlong, jint);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxExecutorBackward
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxExecutorBackward
  (JNIEnv *, jobject, jlong, jlongArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxExecutorPrint
 * Signature: (JLthu/brainmatrix/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxExecutorPrint
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxExecutorSetMonitorCallback
 * Signature: (JLthu/brainmatrix/MXMonitorCallback;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxExecutorSetMonitorCallback
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaSymbolIsAtomic
 * Signature: (JLthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaSymbolIsAtomic
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaSymbolIsVariable
 * Signature: (JLthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaSymbolIsVariable
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolListAtomicSymbolCreators
 * Signature: (Lscala/collection/mutable/ListBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolListAtomicSymbolCreators
  (JNIEnv *, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolGetAtomicSymbolInfo
 * Signature: (JLthu/brainmatrix/Base/RefString;Lthu/brainmatrix/Base/RefString;Lthu/brainmatrix/Base/RefInt;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lthu/brainmatrix/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolGetAtomicSymbolInfo
  (JNIEnv *, jobject, jlong, jobject, jobject, jobject, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolCreateAtomicSymbol
 * Signature: (J[Ljava/lang/String;[Ljava/lang/String;Lthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolCreateAtomicSymbol
  (JNIEnv *, jobject, jlong, jobjectArray, jobjectArray, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolSetAttr
 * Signature: (JLjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolSetAttr
  (JNIEnv *, jobject, jlong, jstring, jstring);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolCompose
 * Signature: (JLjava/lang/String;[Ljava/lang/String;[J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolCompose
  (JNIEnv *, jobject, jlong, jstring, jobjectArray, jlongArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolCreateVariable
 * Signature: (Ljava/lang/String;Lthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolCreateVariable
  (JNIEnv *, jobject, jstring, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolGetAttr
 * Signature: (JLjava/lang/String;Lthu/brainmatrix/Base/RefString;Lthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolGetAttr
  (JNIEnv *, jobject, jlong, jstring, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolListArguments
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolListArguments
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolCopy
 * Signature: (JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolCopy
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolListAuxiliaryStates
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolListAuxiliaryStates
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolListOutputs
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolListOutputs
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolPrintVector
 * Signature: (I[I)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolPrintVector
  (JNIEnv *, jobject, jint, jintArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolCreateGroup
 * Signature: ([JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolCreateGroup
  (JNIEnv *, jobject, jlongArray, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolPrint
 * Signature: (JLthu/brainmatrix/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolPrint
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolGetInternals
 * Signature: (JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolGetInternals
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolInferType
 * Signature: (J[Ljava/lang/String;[ILscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolInferType
  (JNIEnv *, jobject, jlong, jobjectArray, jintArray, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolInferShape
 * Signature: (JI[Ljava/lang/String;[I[ILscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolInferShape
  (JNIEnv *, jobject, jlong, jint, jobjectArray, jintArray, jintArray, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolGetOutput
 * Signature: (JILthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolGetOutput
  (JNIEnv *, jobject, jlong, jint, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxSymbolSaveToJSON
 * Signature: (JLthu/brainmatrix/Base/RefString;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxSymbolSaveToJSON
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxExecutorBindX
 * Signature: (JIII[Ljava/lang/String;[I[II[J[J[I[JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxExecutorBindX
  (JNIEnv *, jobject, jlong, jint, jint, jint, jobjectArray, jintArray, jintArray, jint, jlongArray, jlongArray, jintArray, jlongArray, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxRandomSeed
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxRandomSeed
  (JNIEnv *, jobject, jint);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaOpListArguments
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaOpListArguments
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaOpListAuxiliaryStates
 * Signature: (JLscala/collection/mutable/ArrayBuffer;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaOpListAuxiliaryStates
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaOpInit
 * Signature: (J[Ljava/lang/String;[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaOpInit
  (JNIEnv *, jobject, jlong, jobjectArray, jobjectArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaOpPrintParam
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaOpPrintParam
  (JNIEnv *, jobject, jlong);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaCreateOperatorProperty
 * Signature: (JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaCreateOperatorProperty
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaOpNumVisibleOutputs
 * Signature: (JLthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaOpNumVisibleOutputs
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaOPCopy
 * Signature: (JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaOPCopy
  (JNIEnv *, jobject, jlong, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaToStaticGraph
 * Signature: (Lthu/brainmatrix/Base/RefLong;[I[I[I[JI[Ljava/lang/String;[I[I[I[I[II[Ljava/lang/String;[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaToStaticGraph
  (JNIEnv *, jobject, jobject, jintArray, jintArray, jintArray, jlongArray, jint, jobjectArray, jintArray, jintArray, jintArray, jintArray, jintArray, jint, jobjectArray, jobjectArray);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaSGInferShape
 * Signature: (JII[I[I[ILscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lscala/collection/mutable/ListBuffer;Lthu/brainmatrix/Base/RefInt;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaSGInferShape
  (JNIEnv *, jobject, jlong, jint, jint, jintArray, jintArray, jintArray, jobject, jobject, jobject, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxScalaExecutorBindX
 * Signature: (JIII[Ljava/lang/String;[I[II[J[J[I[JLthu/brainmatrix/Base/RefLong;)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxScalaExecutorBindX
  (JNIEnv *, jobject, jlong, jint, jint, jint, jobjectArray, jintArray, jintArray, jint, jlongArray, jlongArray, jintArray, jlongArray, jobject);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArrayGetData
 * Signature: (JLthu/brainmatrix/Base/RefFloat;I)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArrayGetData
  (JNIEnv *, jobject, jlong, jobject, jint);

/*
 * Class:     thu_brainmatrix_LibInfo
 * Method:    mxNDArraySetData
 * Signature: (JFI)I
 */
JNIEXPORT jint JNICALL Java_thu_brainmatrix_LibInfo_mxNDArraySetData
  (JNIEnv *, jobject, jlong, jfloat, jint);

#ifdef __cplusplus
}
#endif
#endif
