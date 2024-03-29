package thu.brainmatrix.gan

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import thu.brainmatrix.Context
import thu.brainmatrix.Shape
import thu.brainmatrix.IO
import thu.brainmatrix.NDArray
import thu.brainmatrix.CustomMetric
import thu.brainmatrix.Xavier
import thu.brainmatrix.optimizer.Adam
import thu.brainmatrix.DataBatch
import thu.brainmatrix.Symbol
import thu.brainmatrix.Shape

import thu.brainmatrix.util.CVTool

/**
 * @author Depeng Liang
 */
object GanMnist {

  private val logger = LoggerFactory.getLogger(classOf[GanMnist])

    // a deconv layer that enlarges the feature map
  def deconv2D(data: Symbol, iShape: Shape, oShape: Shape,
    kShape: (Int, Int), name: String, stride: (Int, Int) = (2, 2)): Symbol = {
    val targetShape = (oShape(oShape.length - 2), oShape(oShape.length - 1))
    val net = Symbol.Deconvolution(name)(Map(
                                           "data" -> data,
                                           "kernel" -> s"$kShape",
                                           "stride" -> s"$stride",
                                           "target_shape" -> s"$targetShape",
                                           "num_filter" -> oShape(0),
                                           "no_bias" -> true))
    net
  }

  def deconv2DBnRelu(data: Symbol, prefix: String, iShape: Shape,
      oShape: Shape, kShape: (Int, Int), eps: Float = 1e-5f + 1e-12f): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.BatchNorm(s"${prefix}_bn")(Map("data" -> net, "fix_gamma" -> true, "eps" -> eps))
    net = Symbol.Activation(s"${prefix}_act")(Map("data" -> net, "act_type" -> "relu"))
    net
  }

  def deconv2DAct(data: Symbol, prefix: String, actType: String,
    iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.Activation(s"${prefix}_act")(Map("data" -> net, "act_type" -> actType))
    net
  }

  
  def makeDcganSym(oShape: Shape, ngf: Int = 128, finalAct: String = "sigmoid",
      eps: Float = 1e-5f + 1e-12f): (Symbol, Symbol) = {

    val code = Symbol.Variable("rand")
    var net = Symbol.FullyConnected("g1")(Map("data" -> code,
      "num_hidden" -> 4 * 4 * ngf * 4, "no_bias" -> true))
    net = Symbol.Activation("gact1")(Map("data" -> net, "act_type" -> "relu"))
    // 4 x 4
    net = Symbol.Reshape()(Map("data" -> net, "shape" -> s"(-1, ${ngf * 4}, 4, 4)"))
    // 8 x 8
    net = deconv2DBnRelu(net, prefix = "g2",
      iShape = Shape(ngf * 4, 4, 4), oShape = Shape(ngf * 2, 8, 8), kShape = (3, 3))
    // 14x14
    net = deconv2DBnRelu(net, prefix = "g3",
      iShape = Shape(ngf * 2, 8, 8), oShape = Shape(ngf, 14, 14), kShape = (4, 4))
    // 28x28
    val gout = deconv2DAct(net, prefix = "g4", actType = finalAct, iShape = Shape(ngf, 14, 14),
      oShape = Shape(oShape.toArray.takeRight(3)), kShape = (4, 4))

    
      
    val data = Symbol.Variable("data")
    // 28 x 28
    val conv1 = Symbol.Convolution("conv1")(Map("data" -> data,
      "kernel" -> "(5,5)", "num_filter" -> 20))
    val tanh1 = Symbol.Activation()(Map("data" -> conv1, "act_type" -> "tanh"))
    val pool1 = Symbol.Pooling()(Map("data" -> tanh1,
      "pool_type" -> "max", "kernel" -> "(2,2)", "stride" -> "(2,2)"))
    // second conv
    val conv2 = Symbol.Convolution("conv2")(Map("data" -> pool1,
      "kernel" -> "(5,5)", "num_filter" -> 50))
    val tanh2 = Symbol.Activation()(Map("data" -> conv2, "act_type" -> "tanh"))
    val pool2 = Symbol.Pooling()(Map("data" -> tanh2, "pool_type" -> "max",
      "kernel" -> "(2,2)", "stride" -> "(2,2)"))
    var dflat = Symbol.Flatten()(Map("data" -> pool2))
    var d5 = Symbol.FullyConnected("fc1")(Map("data" -> dflat, "num_hidden" -> 500))
    d5 = Symbol.Activation()(Map("data" -> d5, "act_type" -> "tanh"))
    d5 = Symbol.FullyConnected("fc_dloss")(Map("data" -> d5, "num_hidden" -> 1))
    val dloss = Symbol.LogisticRegressionOutput("dloss")(Map("data" -> d5))

    val group = Symbol.Group(gout,net)
    
    (group, dloss)
  }

  // Evaluation
  def ferr(label: NDArray, pred: NDArray): Float = {
    val predArr = pred.toArray.map(p => if (p > 0.5) 1f else 0f)
    val labelArr = label.toArray
    labelArr.zip(predArr).map { case (l, p) => Math.abs(l - p) }.sum / label.shape(0)
  }

  def main(args: Array[String]): Unit = {
    val anst = new GanMnist
    val parser: CmdLineParser = new CmdLineParser(anst)
      parser.parseArgument(args.toList.asJava)
      assert(anst.mnistDataPath != null)

      val lr = 0.0005f
      val beta1 = 0.5f
      val batchSize = 100
      val randShape = Shape(batchSize, 100)
      val numEpoch = 100
      val dataShape = Shape(batchSize, 1, 28, 28)
      val context = if (anst.gpu == -1) Context.cpu() else Context.gpu(anst.gpu)

      val (symGen, symDec) =
      makeDcganSym(oShape = dataShape, ngf = 32, finalAct = "sigmoid")

      val gMod = new GANModule(
          symGen,
          symDec,
          context = context,
          dataShape = dataShape,
          codeShape = randShape)

      gMod.initGParams(new Xavier(factorType = "in", magnitude = 2.34f))
      gMod.initDParams(new Xavier(factorType = "in", magnitude = 2.34f))

      gMod.initOptimizer(new Adam(learningRate = lr, wd = 0f, beta1 = beta1))

      val params = Map(
        "image" -> s"${anst.mnistDataPath}/train-images-idx3-ubyte",
        "label" -> s"${anst.mnistDataPath}/train-labels-idx1-ubyte",
        "input_shape" -> s"(1, 28, 28)",
        "batch_size" -> s"$batchSize",
        "shuffle" -> "True"
      )

      val mnistIter = IO.MNISTIter(params)

      val metricAcc = new CustomMetric(ferr, "ferr")

      var t = 0
      var dataBatch: DataBatch = null
      for (epoch <- 0 until numEpoch) {
    	  println(s"epch:$epoch")
        mnistIter.reset()
        metricAcc.reset()
        t = 0
        while (mnistIter.hasNext) {
          dataBatch = mnistIter.next()
          gMod.update(dataBatch)
          gMod.dLabel.set(0f)
          metricAcc.update(Array(gMod.dLabel), gMod.outputsFake)
          
          gMod.dLabel.set(1f)
          metricAcc.update(Array(gMod.dLabel), gMod.outputsReal)

          if (t % 50 == 0) {
            val (name, value) = metricAcc.get
            println(s"epoch: $epoch, iter $t, metric=$value")
//            logger.info(s"epoch: $epoch, iter $t, metric=$value")
            CVTool.saveFlattenImage(gMod.tempOutG(0),"output/gout.jpg")
            CVTool.saveFlattenImage(gMod.tempOutG(1),"output/gdeconv14.jpg")
//            Viz.imSave("gout", anst.outputPath, gMod.tempOutG(0), flip = true)
            val diff = gMod.tempDiffD
            val arr = diff.toArray
            val mean = arr.sum / arr.length
            val std = {
              val tmpA = arr.map(a => (a - mean) * (a - mean))
              Math.sqrt(tmpA.sum / tmpA.length).toFloat
            }
            diff.set((diff - mean) / std + 0.5f)
            CVTool.saveFlattenImage(diff,"output/mnist.jpg")
            CVTool.saveFlattenImage(dataBatch.data(0).copy(),"output/data0.jpg")
//            println(diff.shape)
//            Viz.imSave("diff", anst.outputPath, diff, flip = true)
//            Viz.imSave("data", anst.outputPath, dataBatch.data(0), flip = true)
          }

          t += 1
        }
      }
   
  }
}

class GanMnist {
  @Option(name = "--mnist-data-path", usage = "the mnist data path")
  val mnistDataPath: String = "data/"
  @Option(name = "--output-path", usage = "the path to save the generated result")
  val outputPath: String = "output/"
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  val gpu: Int = -1
}