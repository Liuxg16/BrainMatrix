package thu.brainmatrix.lstmbyguo

import java.io.File
import java.io.FileNotFoundException
import java.io.FileWriter

import scala.collection.immutable.Set
import scala.io.Source

import thu.brainmatrix.NDArray
import thu.brainmatrix.Random
import thu.brainmatrix.Shape

/*
 *@author guoshen
 *@date 2016/8/11
 *@ introduction:The model that product the text by char-level use the vanilla rnn.
 * */
class StandardLSTM {

}
object StandardLSTM {
  private val inputfilepath: String = "./seqData/inputs.txt" //数据文件所在的绝对路径
  private val outputfilepath: String = "./seqData/outputs.txt"
  private val matrixfilepath: String = "./seqData/matrixs.txt"
  var outputfile = new File(outputfilepath)
  if (outputfile.exists())
    outputfile.delete() //把旧文件删除了
  outputfile.createNewFile()
  var matrixfile = new File(matrixfilepath)
  if (matrixfile.exists())
    matrixfile.delete()
  matrixfile.createNewFile()

  def lossfunction(inputs: Array[Int], targets: Array[Int],
                   hprev: NDArray, cprev: NDArray,
                   vocab_size: Int, cell_size: Int,
                   Wxi: NDArray, Whi: NDArray, bi: NDArray,
                   Wxf: NDArray, Whf: NDArray, bf: NDArray,
                   Wxo: NDArray, Who: NDArray, bo: NDArray,
                   Wxc: NDArray, Whc: NDArray, bc: NDArray,
                   Why: NDArray, by: NDArray): (NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, Double, NDArray, NDArray) = {

    val len: Int = inputs.length
    var x: Array[NDArray] = new Array(len + 1) //input word vector  
    var h: Array[NDArray] = new Array(len + 1) //hidden 
    var i: Array[NDArray] = new Array(len + 1) //input gate 
    var f: Array[NDArray] = new Array(len + 1) //forget gate
    var o: Array[NDArray] = new Array(len + 1) //output gate
    var c_in: Array[NDArray] = new Array(len + 1)
    var c: Array[NDArray] = new Array(len + 1) //cell 
    var y: Array[NDArray] = new Array(len + 1) //output used to turn to p
    var p: Array[NDArray] = new Array(len + 1) //softmax概率层
    var loss: Double = 0.0

    h(0) = hprev
    c(0) = cprev

    //forward pass 
    for (t <- 1 to len) {
      x(t) = NDArray.zeros(vocab_size, 1)
      x(t)(inputs(t - 1)) = 1
      //input gate 
      i(t) = NDArray.sigmod(NDArray.dot(Wxi, x(t)) + NDArray.dot(Whi, h(t - 1)) + bi)
      //forget gate 
      f(t) = NDArray.sigmod(NDArray.dot(Wxf, x(t)) + NDArray.dot(Whf, h(t - 1)) + bf)
      //cell
      c_in(t) = NDArray.sigmod(NDArray.dot(Wxc, x(t)) + NDArray.dot(Whc, h(t - 1)) + bc)
      c(t) = f(t) * c(t - 1) + i(t) * c_in(t) //这个地方不是点乘，目的是为了删除旧记忆，引入新记忆
      //output gate 
      o(t) = NDArray.sigmod(NDArray.dot(Wxo, x(t)) + NDArray.dot(Who, h(t - 1)) + bo)
      //cell output 
      h(t) = o(t) * NDArray.tanh(c(t))
      //softmax
      y(t) = NDArray.dot(Why, h(t))
      var expy = NDArray.exp(y(t))
      p(t) = expy / (NDArray.sum(expy).toScalar)
      println("hehe:" + p(t).toArray(targets(t - 1)))
      loss += -scala.math.log(p(t).toArray(targets(t - 1))) //损失函数,交叉熵
    }
    println("loss :" + loss)

    //backforward pass
    var dWxi = NDArray.zeros(Wxi.shape)
    var dWhi = NDArray.zeros(Whi.shape)
    var dbi = NDArray.zeros(bi.shape)

    var dWxf = NDArray.zeros(Wxf.shape)
    var dWhf = NDArray.zeros(Whf.shape)
    var dbf = NDArray.zeros(bf.shape)

    var dWxo = NDArray.zeros(Wxo.shape)
    var dWho = NDArray.zeros(Who.shape)
    var dbo = NDArray.zeros(bo.shape)

    var dWxc = NDArray.zeros(Wxc.shape)
    var dWhc = NDArray.zeros(Whc.shape)
    var dbc = NDArray.zeros(bc.shape)

    var dWhy = NDArray.zeros(Why.shape)
    var dby = NDArray.zeros(by.shape)

    var hi_next, hf_next, hc_in_next, hc_next, ho_next = NDArray.zeros(cell_size, 1)
    var cc_in_next, cc_next = NDArray.zeros(cell_size, 1)
    var iraw, fraw, c_inraw, craw, oraw = NDArray.zeros(cell_size, 1)
    var dyt, dft, dit, dht, dot, dct, dc_int = NDArray.zeros(cell_size, 1)
    val ones = NDArray.ones(cell_size, 1)
    for (hehe <- 0 until len) {
      var t = len - hehe;

      dyt = NDArray.copy(p(t)) //(vocab_size , 1)
      dyt(targets(t - 1)) -= 1
      dWhy += NDArray.dot(dyt, NDArray.transpose(h(t))) //( vocab_size , cell_size )
      dby += dyt
      dht = NDArray.dot(NDArray.transpose(Why), dyt) + hi_next + hf_next + hc_in_next + ho_next

      var tanhct = NDArray.tanh(c(t))
      dot = dht * tanhct //( cell_size , 1 )
      oraw = (ones - o(t)) * o(t) * dot //( cell_size , 1 )
      dct = dht * o(t) * (ones - tanhct * tanhct) + cc_next

      dWxo += NDArray.dot(oraw, NDArray.transpose(x(t)))
      dWho += NDArray.dot(oraw, NDArray.transpose(h(t - 1)))
      dbo += oraw

      dit = dct * c_in(t) //( cell_size , 1 )
      dft = dct * c(t - 1) //( cell_size , 1 )
      dc_int = dct * i(t) //( cell_size , 1 )

      iraw = (ones - i(t)) * i(t) * dit //( cell_size , 1 )
      fraw = (ones - f(t)) * f(t) * dft //( cell_size , 1 )
      c_inraw = (ones - c_in(t)) * c_in(t) * dc_int //( cell_size , 1 )

      dWxc += NDArray.dot(c_inraw, NDArray.transpose(x(t)))
      dWhc += NDArray.dot(c_inraw, NDArray.transpose(h(t - 1)))
      dbc += c_inraw

      dWxf += NDArray.dot(fraw, NDArray.transpose(x(t)))
      dWhf += NDArray.dot(fraw, NDArray.transpose(h(t - 1)))
      dbf += fraw

      dWxi += NDArray.dot(iraw, NDArray.transpose(x(t)))
      dWhi += NDArray.dot(iraw, NDArray.transpose(h(t - 1)))
      dbi += iraw

      hi_next = NDArray.dot(Whi, iraw)
      hf_next = NDArray.dot(Whf, fraw)
      hc_in_next = NDArray.dot(Whc, c_inraw)
      ho_next = NDArray.dot(Who, oraw)

      cc_next = dct * f(t)
    }
    var parameterlist: Array[NDArray] = Array(
      Wxi, Whi, bi,
      Wxf, Whf, bf,
      Wxo, Who, bo,
      Wxc, Whc, bc,
      Why, by)
    for (i <- 0 until parameterlist.length) {
      parameterlist(i) = NDArray.clip(parameterlist(i), -5, 5)
    }
    (dWxi, dWhi, dbi,
      dWxf, dWhf, dbf,
      dWxo, dWho, dbo,
      dWxc, dWhc, dbc,
      dWhy, dby,
      loss, h(len), c(len))
  }

  def main(args: Array[String]) {
    var data: String = ""
    var chars: Array[Char] = Array()
    var data_size, vocab_size = 0; //data_size是指输入文本的长度，vocab_size是指字符表的长度
    try {
      val tempdata = Source.fromFile(new File(inputfilepath)).getLines().toList //读出文件所有文本数据，并按行作为list保存
      var set: Set[Char] = Set() //将data里面的字符统计为一个字符集合
      for (i <- tempdata) {
        set = set.++(i.toSet); data += i + '\n'
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

    val cell_size: Int = 128 * 4
    val lbd: Float = 0.toFloat
    val rbd: Float = 0.01.toFloat
    //input gate parameters
    var Wxi = Random.uniform(lbd, rbd, Shape(cell_size, vocab_size))
    var Whi = Random.uniform(lbd, rbd, Shape(cell_size, cell_size))
    var Wci = Random.uniform(lbd, rbd, Shape(cell_size, cell_size))
    var bi = NDArray.zeros(cell_size, 1)

    //forget gate parameters
    var Wxf = Random.uniform(lbd, rbd, Shape(cell_size, vocab_size))
    var Whf = Random.uniform(lbd, rbd, Shape(cell_size, cell_size))
    var Wcf = Random.uniform(lbd, rbd, Shape(cell_size, cell_size))
    var bf = NDArray.zeros(cell_size, 1)

    //cell parameters
    var Wxc = Random.uniform(lbd, rbd, Shape(cell_size, vocab_size))
    var Whc = Random.uniform(lbd, rbd, Shape(cell_size, cell_size))
    var bc = NDArray.zeros(cell_size, 1)

    //output gate parameters
    var Wxo = Random.uniform(lbd, rbd, Shape(cell_size, vocab_size))
    var Who = Random.uniform(lbd, rbd, Shape(cell_size, cell_size))
    var Wco = Random.uniform(lbd, rbd, Shape(cell_size, cell_size))
    var bo = NDArray.zeros(cell_size, 1)

    //output parameters
    var Why = Random.uniform(lbd, rbd, Shape(vocab_size, cell_size)) //这一层是softmax层   
    var by = NDArray.zeros(vocab_size, 1)

    val seq_length = 25 //每次训练用的样本字符长度
    val learning_rate = 2e-3.toFloat
    var n: Int = 0 //n表示为迭代次数
    var p: Int = 0 //p表示指针，指向输入的起始位置

    var hprev = NDArray.zeros(cell_size, 1) //上一层的隐层输入，初始化为0

    var cprev = NDArray.zeros(cell_size, 1) //上一层的细胞输入，初始化为0
    while (n < 10000) {
      n += 1
      var inputs: Array[Int] = Array();
      var targets: Array[Int] = Array()
      if (p + seq_length + 1 >= data_size) {
        p = 0
        hprev = NDArray.zeros(cell_size, 1)
        cprev = NDArray.zeros(cell_size, 1)
      }
      for (index <- p until p + seq_length) {
        inputs = inputs :+ (char_to_ix.apply(data(index))) //apply(key) => value
        targets = targets :+ (char_to_ix.apply(data(index + 1)))
      }

      var (dWxi, dWhi, dbi,
        dWxf, dWhf, dbf,
        dWxo, dWho, dbo,
        dWxc, dWhc, dbc,
        dWhy, dby,
        smooth_loss, temp_hprev, temp_cprev) = lossfunction(inputs, targets,
        hprev, cprev,
        vocab_size, cell_size,
        Wxi, Whi, bi,
        Wxf, Whf, bf,
        Wxo, Who, bo,
        Wxc, Whc, bc,
        Why, by)
      hprev = temp_hprev; cprev = temp_cprev

      var mWxi = NDArray.zeros(Wxi.shape)
      var mWhi = NDArray.zeros(Whi.shape)
      var mWci = NDArray.zeros(Wci.shape)
      var mbi = NDArray.zeros(bi.shape)

      var mWxf = NDArray.zeros(Wxf.shape)
      var mWhf = NDArray.zeros(Whf.shape)
      var mWcf = NDArray.zeros(Wcf.shape)
      var mbf = NDArray.zeros(bf.shape)

      var mWxc = NDArray.zeros(Wxc.shape)
      var mWhc = NDArray.zeros(Whc.shape)
      var mbc = NDArray.zeros(bc.shape)

      var mWxo = NDArray.zeros(Wxo.shape)
      var mWho = NDArray.zeros(Who.shape)
      var mWco = NDArray.zeros(Wco.shape)
      var mbo = NDArray.zeros(bo.shape)

      var mWhy = NDArray.zeros(Why.shape)
      var mby = NDArray.zeros(by.shape)
      var zips = Array(
        Array(Wxi, dWxi, mWxi), Array(Whi, dWhi, mWhi), Array(bi, dbi, mbi),
        Array(Wxf, dWxf, mWxf), Array(Whf, dWhf, mWhf), Array(bf, dbf, mbf),
        Array(Wxc, dWxc, mWxc), Array(Whc, dWhc, mWhc), Array(bc, dbc, mbc),
        Array(Wxo, dWxo, mWxo), Array(Who, dWho, mWho), Array(bo, dbo, mbo),
        Array(Why, dWhy, mWhy), Array(by, dby, mby))
      val little = 1e-8.toFloat
      //利用Adagrad来优化学习速率
      for (i <- 0 until zips.length) {
        zips(i)(2) += zips(i)(1) * zips(i)(1)
        zips(i)(0) += -zips(i)(1) * learning_rate / NDArray.sqrt(zips(i)(2) + NDArray.ones(zips(i)(2).shape) * little)
      }

//      if (n % 50 == 0) {
//        printf("迭代次数:%d,smooth_loss:%f\n", n, smooth_loss)
//        val writer = new FileWriter(matrixfilepath, true)
//        //        writer.write("********************Wxi Whi bi********************\n" + Wxi + "\n\n" + Whi + "\n\n" + NDArray.transpose(bi) + "\n")
//        //        writer.write("********************Wxf Whf bf********************\n" + Wxf + "\n\n" + Whf + "\n\n" + NDArray.transpose(bf) + "\n")
//        //        writer.write("********************Wxc Whc bc********************\n" + Wxc + "\n\n" + Whc + "\n\n" + NDArray.transpose(bc) + "\n")
//        //        writer.write("********************Wxo Who bo********************\n" + Wxo + "\n\n" + Who + "\n\n" + NDArray.transpose(bo) + "\n")
//        writer.write("********************Why by********************\n" + Why + "\n")
//        writer.close()
//      }
      p += seq_length

      println("第" + n + "轮结束～")
    }
  }
}