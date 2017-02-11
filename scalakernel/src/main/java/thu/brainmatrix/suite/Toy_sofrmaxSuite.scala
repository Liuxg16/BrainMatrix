
package thu.brainmatrix.suite

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import thu.brainmatrix.nce_loss.DataIter_
import thu.brainmatrix.nce_loss.DataIter_nce
import thu.brainmatrix.Shape


class Toy_sofrmaxSuite extends FunSuite with BeforeAndAfterAll{
   test("dataIter_:dispose()"){
	   val dataiter_ = new DataIter_(200,32,24,50)
	     
	   var batch = dataiter_.next()
	   //println(batch.data(0))
	   //println(batch.label(0))  
	   batch.dispose()
	   //println("------------------------------------")
	   dataiter_.next()
	   dataiter_.next()
	   dataiter_.next()
	   dataiter_.next()
	   var batch1 = dataiter_.next()
//	   println(batch1.label(0))
	   
	   //println("------------------------------------")
	   dataiter_.reset()
	   batch1 = dataiter_.next()
//	   println(batch1.data(0))
//	   println(batch1.label(0))
   }
   
   test("testData"){
	   val dataiter_ = new DataIter_(100000,128,100,10000)
//	   println(dataiter_.next().label(0))
   }
   
   test("testData_nce"){
	   val batch_size   = 128
	   val vocab_size   = 100
	   val feature_size = 100
	   val num_label    = 6
	   val data_train = new DataIter_nce(10000,batch_size,feature_size,vocab_size,num_label)
	   val batch = data_train.next()
	   assert(batch.label(0).shape==Shape(128,6))
	   
	     
   }
   
   
   
   
   
   
}
