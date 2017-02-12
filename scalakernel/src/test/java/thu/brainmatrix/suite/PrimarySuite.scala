package thu.brainmatrix.suite
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import thu.brainmatrix.rnn.Utils
import thu.brainmatrix.NDArray
import thu.brainmatrix.Shape
import breeze.linalg._
import breeze.plot._
class PrimarySuite extends FunSuite with BeforeAndAfterAll{
	
	
	
	
//	test("plot"){
  def testplot{
		val f = Figure()
		val p = f.subplot(0)
		val x = linspace(0.0,1.0)
		p += plot(x, x :^ 2.0)
		p += plot(x, x :^ 3.0, '.')
		p.xlabel = "x axis"
		p.ylabel = "y axis"
//		f.saveas("lines.png") // save current figure as a .png, eps and pdf also supported
	}
	
//	test("plot1"){
  def testplot1{
		val f = Figure()
		val p = f.subplot(0)
		val x = linspace(0.0,1.0)
		val xx = Array(2d,3d,4d,5d,6d)
		val xxx = DenseVector.create(xx, 0, 1,3)
//		xxx.data.foreach {println}
		p += plot(xxx, xxx :^ 2.0)
//		p += plot(x, x :^ 3.0, '.')
//		p.xlabel = "x axis"
//		p.ylabel = "y axis"
//		f.saveas("lines.png") // save current figure as a .png, eps and pdf also supported
	}
	
	

	
	/**
	 * generate the indexs of the list
	 */
	test("List:indices"){
		val buckets = List(2,3,4)
		val a = buckets.indices
//		println(a)
	}
	
	/**
	 * find is useless!!!
	 */
	test("find"){
		val arr = Array(1,2,3,4,4,5)
//		arr.find(x => x%2==0).foreach(println)
		
	}
  
	
	
	/*
	 * re-generate a list with the same elements but different order
	 */
	test("Random:shuffle"){
		val plan = Array(1,2,3,4)
//		println(scala.util.Random.shuffle(plan.toList))
	}
	
	
	
	test("perplexity"){
		val a = NDArray.diag(Shape(2,3))
//		println(a)
		val b = NDArray.ones(Shape(2,3))*2
		val c = Utils.perplexity(a,b)
//		println(c)
	}
	
	/*
	 * return a iterator contains many groups
	 * @param size
	 *	the number of elements per group
	 */
	test("grouped"){
		val arr = Array(2,3,4,5,3,4,6,7)
		val a = arr.grouped(5)
//		a.next().foreach { print}
	}
	
	
	test("reduce"){
		val arrs = Array(Array(1,2,3,4),Array(6,7,8),Array(6,7,8))
		val ret = arrs.reduce(_++_)
		
//		ret.foreach {println}
	}
	
	
	test("foldLeft"){
		val arrs = Array(Array(1,2,3,4),Array(6,7,8))
		val ret = arrs.foldLeft(Array[Int]())(_++_)
		
//		ret.foreach {println}
	}
	
	test("collection:Set"){
		var rn = Set[Int]()
		rn = rn + 2
		rn = rn + 1
		rn = rn + 2
//		println(rn)
	}
	
	
	test("val a: IndexedSeq[Int]"){
		val a = 2 % 10 +: (0 until 10).map(_ => scala.util.Random.nextInt(90 -1))
//		println(a.toArray.length)
	}
	
	test("sorted"){
		val a = Array(2,7,3,51,7)
//		a.sorted.foreach(println)
	}
	
	
	
	
	
}