package thu.brainmatrix.lstmbyguo

import java.io.File
import java.io.FileNotFoundException
import scala.collection.immutable.Set
import scala.io.Source

import thu.brainmatrix.NDArray
import thu.brainmatrix.Random
import scala.util.control.Breaks
import java.io.PrintWriter
import java.io.FileWriter
import thu.brainmatrix.Shape

/*
 *@author guoshen
 *@date 2016/7/21
 *@ introduction:The model that product the text by char-level use the vanilla rnn.
 * */
class CharRNN {
}

object CharRNN {
  private val inputfilepath: String = "./seqData/inputs.txt" //数据文件所在的绝对路径
  private val outputfilepath: String = "./seqData/outputs.txt"
  private val matrixfilepath: String = "./seqData/matrixs.txt"
  var outputfile = new File(outputfilepath)
  //  outputfile.deleteOnExit() //把旧文件删除了
  outputfile.createNewFile()
  var matrixfile = new File(matrixfilepath)
  matrixfile.createNewFile()
  /**
   * @author guoshen
   * @date 2016/7/21
   * @brief
   * 通过加权的方式进行概率抽样，主要思路如下：
   *   假设，概率分布为pro[0.2,0.3,0.5]
   *   那么计算一个概率和数组sum[0.2,0.5,1.0]
   *   然后随机生成一个[0,1]之间的数rand，将rand与sum里面的数依次比较
   *   选择第一个比rand大的sum，不妨设sum[i]>=rand
   *   返回sum[i]的index -> i
   */
  def plusproform(pro: NDArray): NDArray = {
    var sum: Array[Float] = NDArray.zeros(pro.shape).toArray
    var temp_sum: Float = 0
    for (i <- 0 until pro.size) {
      temp_sum += pro(i)
      sum(i) = temp_sum
    }
    var rand = Math.random().toFloat
    var res = -1
    val loop = new Breaks
    loop.breakable {
      for (i <- 0 until sum.length) {
        if (rand <= sum(i)) { res = i; loop.break() }
      }
    }
    NDArray.array(Array(res), Shape(1, 1))
  }

  def sample(h: NDArray, seed_ix: Int, n: Int, vocab_size: Int, Wxh: NDArray, Whh: NDArray, Why: NDArray, bh: NDArray, by: NDArray): Array[Int] = {
    var x = NDArray.zeros(vocab_size, 1)
    x(seed_ix + 1) = 1 //x是由字符表对应产生的字符向量
    //    println("seed:" + seed_ix + " and x : " + x)
    var ixes: Array[Int] = Array()
    var temph: NDArray = h
    for (t <- 0 until n) {
      temph = NDArray.tanh(NDArray.dot(Wxh, x) + NDArray.dot(Whh, temph) + bh)
      var y = NDArray.dot(Why, temph) + by
      var expy = NDArray.exp(y)
      var p = expy / NDArray.sum(expy).toScalar
      var temp_p = NDArray.array(p.toArray, Shape(1, vocab_size))
      var ix: NDArray = plusproform(temp_p) // NDArray.argmaxChannel(temp_p) //这里应该是利用p的概率分布来生成字符向量，但是似乎没有相应的函数，后面补充 
      x = NDArray.zeros(vocab_size, 1)
      //      println("ix : " + ix(0))
      x(ix(0).toInt) = 1
      ixes = ixes :+ (ix(0).toInt)
    }
    ixes
  }

  def lossfunction(inputs: Array[Int],
                   targets: Array[Int],
                   hprev: NDArray,
                   vocab_size: Int,
                   Wxh: NDArray, Whh: NDArray, Why: NDArray,
                   bh: NDArray, by: NDArray): (Double, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) = {
    val len = inputs.length
    var xs, hs, ys, ps: Array[NDArray] = new Array(len + 1)
    //    println("len:" + len + ",length:" + xs.length)
    hs(0) = hprev
    var loss: Double = 0

    /* forward pass
     *  这里的forward pass的输入是用文本键入的
     *  而sample里面的输入是在输入起始数据之后自己生成的*/
    for (t <- 1 to len) {
      xs(t) = NDArray.zeros(vocab_size, 1)
      xs(t)(inputs(t - 1)) = 1 //根据inputs里面第t个字符对xs(t)进行相应的字符向量初始化
      hs(t) = NDArray.tanh(NDArray.dot(Wxh, xs(t)) + NDArray.dot(Whh, hs(t - 1)) + bh)
      ys(t) = NDArray.dot(Why, hs(t)) + by
      var expys = NDArray.exp(ys(t))
      ps(t) = expys / NDArray.sum(expys).toScalar //预测字符集中每个字符是下个字符的可能性
      //      println(s"啊哈$t    hehe:$hehe")
      loss += -scala.math.log(ps(t).toArray(targets(t - 1))) //这是交叉熵
      //      println("for内loss:" + loss)
      //      println("哦吼" + t)
    }
    println("loss: " + loss)

    /* backward pass*/
    var dWxh = NDArray.zeros(Wxh.shape)
    var dWhh = NDArray.zeros(Whh.shape)
    var dWhy = NDArray.zeros(Why.shape)
    var dbh = NDArray.zeros(bh.shape)
    var dby = NDArray.zeros(by.shape)
    var dhnext = NDArray.zeros(hs(1).shape)

    for (t <- 0 until len) {
      var time = len - t
      var dy = NDArray.copy(ps(time))
      dy(targets(time - 1)) -= 1 //这里将
      dWhy += NDArray.dot(dy, NDArray.transpose(hs(time)))
      dby += dy
      var dh = NDArray.dot(NDArray.transpose(Why), dy) + dhnext
      var dhraw = (NDArray.ones(hs(time).shape) - hs(time) * hs(time)) * dh
      dbh += dhraw
      dWxh += NDArray.dot(dhraw, NDArray.transpose(xs(time)))
      dWhh += NDArray.dot(dhraw, NDArray.transpose(hs(time - 1)))
      dhnext = NDArray.dot(NDArray.transpose(Whh), dhraw)
    }
    var parameterlist: Array[NDArray] = Array(dWxh, dWhh, dWhy, dbh, dby) 
    for (i <- 0 until parameterlist.length) { //这里类似正则项的效果，用于限制参数大小
      parameterlist(i) = NDArray.clip(parameterlist(i), -5, 5)
    }
    (loss, dWxh, dWhh, dWhy, dbh, dby, hs(len))
  }

  def main(args: Array[String]): Unit = {

    var data: String = ""
    var chars: Array[Char] = Array()
    var data_size, vocab_size = 0; //data_size是指输入文本的长度，vocab_size是指字符表的长度
    try {
      val tempdata = Source.fromFile(new File(inputfilepath)).getLines().toList //读出文件所有文本数据，并按行作为list保存
      var set: Set[Char] = Set() //将data里面的字符统计为一个字符集合
      for (i <- tempdata) {
        set = set.++(i.toSet)
        data += i + '\n'
      }
      chars = (set.+('\n')).toArray //小bug，在输入文本里没有换行符的时候这样做是错的
      vocab_size = chars.length; data_size = data.length()
    } catch {
      case e: FileNotFoundException => { println("File Not Found Exception") } // TODO: handle error
    }
    var char_to_ix: Map[Char, Int] = Map() //输入字符，得到对应的字符编号
    var ix_to_char: Map[Int, Char] = Map() //输入字符编号，得到对应的字符
    for (index <- 0 until vocab_size) {
      char_to_ix += (chars(index) -> index)
      ix_to_char += (index -> chars(index))
    }
    //    println(char_to_ix)
    //    println(ix_to_char)

    val hidden_size = 1500 //隐层节点数量
    val seq_length = 25 //每次训练用的样本字符长度
    var learning_rate = 1e-1.toFloat //学习速率

    var Wxh = Random.uniform(0.toFloat, 0.01.toFloat, Shape(hidden_size, vocab_size))
    var Wxh2 = Random.uniform(0.toFloat, 0.01.toFloat, Shape(hidden_size, vocab_size))
    var Whh = Random.uniform(0.toFloat, 0.01.toFloat, Shape(hidden_size, hidden_size))
    var Why = Random.uniform(0.toFloat, 0.01.toFloat, Shape(vocab_size, hidden_size))
    var bh = NDArray.zeros(hidden_size, 1)
    var by = NDArray.zeros(vocab_size, 1)

    var n: Int = 0 //n表示为迭代次数
    var p: Int = 0 //p表示指针，指向输入的起始位置

    var mWxh = NDArray.zeros(Wxh.shape)
    var mWxh2 = NDArray.zeros(Wxh2.shape)
    var mWhh = NDArray.zeros(Whh.shape)
    var mWhy = NDArray.zeros(Why.shape)
    var mbh = NDArray.zeros(bh.shape)
    var mby = NDArray.zeros(by.shape)

    var smooth_loss = -scala.math.log(1.0 / vocab_size) * seq_length
    var hprev = NDArray.zeros(hidden_size, 1)

    while (n <= 1000) {
      if (p + seq_length + 1 >= data_size) {
        p = 0; hprev = NDArray.zeros(hidden_size, 1) //这表示文本全部遍历完成，重置RNN的状态
      }
      var inputs: Array[Int] = Array(); var targets: Array[Int] = Array()
      for (index <- p until p + seq_length) {
        println(index)
        println(data_size)
        inputs = inputs :+ (char_to_ix.apply(data(scala.math.min(index, data_size - 1)))) //apply(key) => value
        targets = targets :+ (char_to_ix.apply(data(scala.math.min(index + 1, data_size))))
      }

      var sample_ix: Array[Int] = Array()
      if (n % 100 == 0) {
        sample_ix = sample(hprev, inputs(0), 200, vocab_size, Wxh, Whh, Why, bh, by) //这个200就是每次生成长度为200的字符串，可自定义修改
        var str = ""
        for (ixs <- sample_ix) str += ix_to_char(ixs)
        val writer = new FileWriter(outputfile, true)
        writer.write("\n\n********************\n\n" + str)
        writer.close()
      }

      var (loss, dWxh, dWhh, dWhy, dbh, dby, temp_hprev) = lossfunction(inputs, targets, hprev, vocab_size, Wxh, Whh, Why, bh, by)
      hprev = temp_hprev
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if (n % 100 == 0) {
        printf("迭代次数:%d,smooth_loss:%f\n", n, smooth_loss)
        val writer = new FileWriter(matrixfilepath, true)
        writer.write("\n\n********************\n\n" + Wxh)
        writer.write("\n\n********************\n\n" + Whh)
        writer.write("\n\n********************\n\n" + Why)
        writer.close()

      }

      var zips = Array(Array(Wxh, dWxh, mWxh), Array(Whh, dWhh, mWhh), Array(Why, dWhy, mWhy), Array(bh, dbh, mbh), Array(by, dby, mby))
      val little = 1e-8.toFloat
      //利用Adagrad来优化学习速率
      for (i <- 0 until zips.length) {
        zips(i)(2) += zips(i)(1) * zips(i)(1)
        zips(i)(0) += -zips(i)(1) * learning_rate / NDArray.sqrt(zips(i)(2) + NDArray.ones(zips(i)(2).shape) * little)
      }
      p += seq_length
      n += 1
      println("第" + n + "轮结束～")
    }
  }
}