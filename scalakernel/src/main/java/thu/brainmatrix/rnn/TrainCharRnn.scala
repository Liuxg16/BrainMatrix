package thu.brainmatrix.rnn

import thu.brainmatrix._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import thu.brainmatrix.optimizer.Adam

/**
 * Follows the demo, to train the char rnn:
 * https://github.com/dmlc/mxnet/blob/master/example/rnn/char-rnn.ipynb
 * @author Depeng Liang
 */
object TrainCharRnn {

  private val logger = LoggerFactory.getLogger(classOf[TrainCharRnn])

  def main(args: Array[String]): Unit = {
    val incr = new TrainCharRnn
    val parser: CmdLineParser = new CmdLineParser(incr)
      parser.parseArgument(args.toList.asJava)
      assert(incr.dataPath != null && incr.saveModelPath != null)

      // The batch size for training
      val batchSize = 32
      // We can support various length input
      // For this problem, we cut each input sentence to length of 129
      // So we only need fix length bucket
      val buckets = List(129)
      // hidden unit in LSTM cell
      val numHidden = 512
      // embedding dimension, which is, map a char to a 256 dim vector
      val numEmbed = 256
      // number of lstm layer
      val numLstmLayer = 3
      // we will show a quick demo in 2 epoch
      // and we will see result by training 75 epoch
      val numEpoch = 1
      // learning rate
      val learningRate = 0.001f
      // we will use pure sgd without momentum
      val momentum = 0.0f

      val ctx = if (incr.gpu == -1) Context.cpu() else Context.gpu(incr.gpu)
      val vocab = Utils.buildVocab(incr.dataPath)
      println(vocab)
      // generate symbol for a length
      def symGen(seqLen: Int): Symbol = {
        Lstm.lstmUnroll(numLstmLayer, seqLen, vocab.size + 1,
                    numHidden = numHidden, numEmbed = numEmbed,
                    numLabel = vocab.size + 1, dropout = 0.2f)
      }

      // initalize states for LSTM
      val initC = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_c", (batchSize, numHidden))
      val initH = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_h", (batchSize, numHidden))
      val initStates = initC ++ initH
      //regard  '\n' as the separator to train
      val dataTrain = new ButketIo.BucketSentenceIter(incr.dataPath, vocab, buckets,
                                          batchSize, initStates, seperateChar = "\n",
                                          text2Id = Utils.text2Id, readContent = Utils.readContent)
      
      // the network symbol
      val symbol = symGen(buckets(0))

      val datasAndLabels = dataTrain.provideData ++ dataTrain.provideLabel
      val (argShapes, outputShapes, auxShapes) = symbol.inferShape(datasAndLabels)

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

      val argNames = symbol.listArguments()
      val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
      val auxNames = symbol.listAuxiliaryStates()
      val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap

      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabels.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap

      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabels.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val data = argDict("data")
      val label = argDict("softmax_label")

      val executor = symbol.bind(ctx, argDict, gradDict)

      val opt = new Adam(learningRate = learningRate, wd = 0.0001f)

      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new CustomMetric(Utils.perplexity, "perplexity")
      val batchEndCallback = new Callback.Speedometer(batchSize, 50)
      val epochEndCallback = Utils.doCheckpoint(s"${incr.saveModelPath}/obama2")

      for (epoch <- 0 until numEpoch) {
        // Training phase
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false
        // Iterate over training data.
        dataTrain.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && dataTrain.hasNext) {
            val dataBatch = dataTrain.next()

            data.set(dataBatch.data(0))
            label.set(dataBatch.label(0))
            executor.forward(isTrain = true)
            println(executor.outputs(0))
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }

            // evaluate at end, so out_cpu_array can lazy copy
//            evalMetric.update(dataBatch.label, Array(executor.outputs(1)))
            dataBatch.dispose()            
            nBatch += 1
//            batchEndCallback.invoke(epoch, nBatch, evalMetric)
            
          }
          if (doReset) {
            dataTrain.reset()
          }
          // this epoch is done
          epochDone = true
        }
//        val (name, value) = evalMetric.get
//        println(s"Epoch[$epoch] Train-$name=$value")
        val toc = System.currentTimeMillis
        println(s"Epoch[$epoch] Time cost=${toc - tic}")

        epochEndCallback.invoke(epoch, symbol, argDict, auxDict)
      }
      executor.dispose()

  }
}

class TrainCharRnn {
  private val dataPath: String = "./seqData/input1.txt"
  private val saveModelPath: String = "./model/"
  private val gpu: Int = 0
}