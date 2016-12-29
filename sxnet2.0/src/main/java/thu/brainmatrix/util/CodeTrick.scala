package thu.brainmatrix.util

object CodeTrick {
    
    def main(args:Array[String]){
        ArrayFillTest
    }
    
    def ArrayFillTest{
      val arr = Array.fill[Float](5)(3f) 
      
     (arr).foreach(println)
    }
}