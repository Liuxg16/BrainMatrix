package thu.brainmatrix.io

import org.apache.http.Header;  
import org.apache.http.HttpResponse;  

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File;  
import java.io.FileOutputStream;  
import java.io.IOException;  

import java.net.URL;
import java.net.URLConnection;
 import java.net.HttpURLConnection;
object FileIO {
    /** 
     * 下载远程文件并保存到本地  
     * @param remoteFilePath 远程文件路径  
     * @param localFilePath 本地文件路径 
     */
    def downloadFile(remoteFilePath:String , localFilePath:String )
    {
        var urlfile:URL  = null;
        var httpUrl:HttpURLConnection = null;
        var bis:BufferedInputStream  = null;
        var bos:BufferedOutputStream = null;
        var f  : File = new File(localFilePath);
        try
        {
            urlfile = new URL(remoteFilePath);
            // force to transmit URLConnection to HttpURLConnection
            httpUrl =(urlfile.openConnection()).asInstanceOf[HttpURLConnection]
            httpUrl.connect();
            bis = new BufferedInputStream(httpUrl.getInputStream());
            bos = new BufferedOutputStream(new FileOutputStream(f));
            var len : Int = 20480000;
            var b:Array[Byte] = Array.fill[Byte](len)('\0') 
            len = bis.read(b)
            while (len!= -1)
            {
//              println(len)
              bos.write(b, 0, len);
              len = bis.read(b)
            }
            bos.flush();
            bis.close();
            httpUrl.disconnect();
        }
        catch
        {
            case ex: Exception => {
                ex.printStackTrace();    
                sys.exit(1)
            }
        }
            
        finally
        {
            try
            {
                bis.close();
                bos.close();
            }
            catch 
            {
              case e:Exception =>
                e.printStackTrace();
            }
        }
    }
    
    def main(args:Array[String]){
        downloadFile("http://data.mxnet.io/data/cifar10/cifar10_val.rec","./data/cifar10_val.rec")
    }
}




   