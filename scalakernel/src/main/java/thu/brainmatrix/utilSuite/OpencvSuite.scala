package thu.brainmatrix.utilSuite
//
//
//import org.opencv.core.Core
//import org.opencv.highgui.Highgui
//import org.opencv.imgproc.Imgproc
//import org.opencv.core.Mat
//import org.opencv.core.CvType
//import org.opencv.core.MatOfInt
//import org.opencv.core.MatOfFloat
//
//import scala.collection.mutable.ArrayBuffer
//
//
class OpencvSuite{
//
//	
//	test("opencv test"){
//        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
//
//        //读取图像，不改变图像的原始信息
//        val m = Highgui.imread("./data/cat.jpg",Highgui.CV_LOAD_IMAGE_COLOR);
//
//        //将图片转换成灰度图片
//        val gray = new Mat(m.size(),CvType.CV_8UC1);
//        Imgproc.cvtColor(m,gray,Imgproc.COLOR_RGB2GRAY);
//
//        //计算灰度直方图
//        val images = new java.util.ArrayList[Mat]()
////        var images = new ArrayBuffer[Mat](); //List<Mat> 
//        images.add(gray);
//
//        val channels= new MatOfInt(0);
//        val histSize = new MatOfInt(256);
//        val ranges= new MatOfFloat(0,256);
//        val hist = new Mat
//        Imgproc.calcHist(images, channels, new Mat(), hist, histSize, ranges);
//
//        //mat求和
//        System.out.println(Core.sumElems(hist));
//
//        //保存转换的图片
//        Highgui.imwrite("output/cat.png",gray);
//
//    }   
//		
//	
//	
//	
//	
}