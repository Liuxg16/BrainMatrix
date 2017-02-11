package thu.brainmatrix.gan

import thu.brainmatrix.Symbol
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.Optimizer
import thu.brainmatrix.NDArray
import thu.brainmatrix.Initializer
import thu.brainmatrix.DataBatch
import thu.brainmatrix.Random

/**
 * @author Depeng Liang
 */
class GanConstr(
              symbolGenerator: Symbol,
              symbolMidGenerator: Symbol,
              symbolEncoder: Symbol,
              context: Context,
              dataShape: Shape,
              codeShape: Shape,
              posLabel: Float = 0.9f) {

  // generator
  private val gDataLabelShape = Map("rand" -> codeShape)
  private val (gArgShapes, gOutShapes, gAuxShapes) = symbolGenerator.inferShape(gDataLabelShape)

  private val gArgNames = symbolGenerator.listArguments()
  private val gArgDict = gArgNames.zip(gArgShapes.map(NDArray.empty(_, context))).toMap

  private val gGradDict = gArgNames.zip(gArgShapes).filter { case (name, shape) =>
    !gDataLabelShape.contains(name)
  }.map(x => x._1 -> NDArray.empty(x._2, context) ).toMap

  symbolMidGenerator.listArguments().foreach {println}  
  
  
  private val gData = gArgDict("rand")

  val gAuxNames = symbolGenerator.listAuxiliaryStates()
  val gAuxDict = gAuxNames.zip(gAuxShapes.map(NDArray.empty(_, context))).toMap
  private val gExecutor =
    symbolGenerator.bind(context, gArgDict, gGradDict, "write", gAuxDict, null, null)

  private val gMidExecutor =
    symbolMidGenerator.bind(context, gArgDict, gGradDict, "null", gAuxDict, null, null)
  
    
  
  
  // discriminator
  private val batchSize = dataShape(0)

  private val dDataShape = Map("data" -> dataShape)
  private val dLabelShape = Map("dloss_label" -> Shape(batchSize))
  private val (dArgShapes, outShapes, dAuxShapes) = symbolEncoder.inferShape(dDataShape ++ dLabelShape)

//  println(outShapes)
  
  private val dArgNames = symbolEncoder.listArguments()
  private val dArgDict = dArgNames.zip(dArgShapes.map(NDArray.empty(_, context))).toMap

  private val dGradDict = dArgNames.zip(dArgShapes).filter { case (name, shape) =>
    !dLabelShape.contains(name)
  }.map(x => x._1 -> NDArray.empty(x._2, context) ).toMap

  private val tempGradD = dArgNames.zip(dArgShapes).filter { case (name, shape) =>
    !dLabelShape.contains(name)
  }.map(x => x._1 -> NDArray.empty(x._2, context) ).toMap

  private val dData = dArgDict("data")
  val dLabel = dArgDict("dloss_label")

  val dAuxNames = symbolEncoder.listAuxiliaryStates()
  val dAuxDict = dAuxNames.zip(dAuxShapes.map(NDArray.empty(_, context))).toMap
  private val dExecutor =
    symbolEncoder.bind(context, dArgDict, dGradDict, "write", dAuxDict, null, null)

  val tempOutG = gOutShapes.map(NDArray.empty(_, context)).toArray
  val tempDiffD: NDArray = dGradDict("data")

  var outputsFake: Array[NDArray] = null
  var outputsReal: Array[NDArray] = null
  var midOutput  : NDArray = null

  def initGParams(initializer: Initializer): Unit = {
    gArgDict.filter(x => !gDataLabelShape.contains(x._1))
                   .foreach { case (name, ndArray) => initializer(name, ndArray) }
  }

  def initDParams(initializer: Initializer): Unit = {
    dArgDict.filter(x => !dDataShape.contains(x._1) && !dLabelShape.contains(x._1))
                   .foreach { case (name, ndArray) => initializer(name, ndArray) }
  }

  private var gOpt: Optimizer = null
  private var gParamsGrads: List[(Int, String, NDArray, AnyRef)] = null
  private var dOpt: Optimizer = null
  private var dParamsGrads: List[(Int, String, NDArray, AnyRef)] = null

  def initOptimizer(opt: Optimizer): Unit = {
    gOpt = opt
    gParamsGrads = gGradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
      (idx, name, grad, gOpt.createState(idx, gArgDict(name)))
    }
    dOpt = opt
    dParamsGrads =
      dGradDict.filter(x => !dDataShape.contains(x._1))
      .toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, dOpt.createState(idx, dArgDict(name)))
    }
  }

  private def saveTempGradD(): Unit = {
    val keys = this.dGradDict.keys
    for (k <- keys) {
      this.dGradDict(k).copyTo(this.tempGradD(k))
    }
  }

  // add back saved gradient
  private def addTempGradD(): Unit = {
    val keys = this.dGradDict.keys
    for (k <- keys) {
      this.dGradDict(k).set(this.dGradDict(k) + this.tempGradD(k))
    }
  }

  // update the model for a single batch
  def update(dBatch: DataBatch): Unit = {
    // generate fake image
    this.gData.set(Random.normal(0, 1.0f, this.gData.shape, context))
    this.gExecutor.forward(isTrain = true)
    val outG = this.gExecutor.outputs(0)
    this.dLabel.set(0f)
    this.dData.set(outG)
    this.dExecutor.forward(isTrain = true)
    this.dExecutor.backward()
    this.saveTempGradD()
    
    // update generator
    this.dLabel.set(1f)
    this.dExecutor.forward(isTrain = true)
    this.dExecutor.backward()
    
    this.gExecutor.backward(tempDiffD) //tempdiffd: the difference of the data with the true data in discriminate model
    
    gParamsGrads.foreach { case (idx, name, grad, optimState) =>
      gOpt.update(idx, gArgDict(name), grad, optimState)
    }
    
    this.outputsFake = this.dExecutor.outputs.map(x => x.copy())
    
//    this.gMidExecutor.forward()
    this.midOutput = gArgDict("g4_deconv_weight")
    
    // update discriminator
    this.dLabel.set(posLabel)
    this.dData.set(dBatch.data(0))
    this.dExecutor.forward(isTrain = true)
    this.dExecutor.backward()
    
    // there are two loss: the fake loss from generate model and the true loss from real data
    this.addTempGradD()
    dParamsGrads.foreach { case (idx, name, grad, optimState) =>
      dOpt.update(idx, dArgDict(name), grad, optimState)
    }
    
    this.outputsReal = this.dExecutor.outputs.map(x => x.copy())
    
    this.tempOutG.indices.foreach(i => this.tempOutG(i).set(this.gExecutor.outputs(i))) //store the output
    
  }
}