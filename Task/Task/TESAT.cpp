#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

//--------------------------------【全局变量声明】----------------------------------------------
//		描述：声明全局变量
//-------------------------------------------------------------------------------------------------
//注意，需要把"haarcascade_frontalface_alt.xml"和"haarcascade_eye_tree_eyeglasses.xml"这两个文件复制到工程路径下
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier face_cascade1;
CascadeClassifier eyes_cascade;
string window_name = "Face detection";
RNG rng(12345);



int main( void )
{
  //VideoCapture capture;
  //载入待检测图片
  Mat srcImg=imread("1.jpg");
  //-- 1. 加载级联（cascades）
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !face_cascade1.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //-- 2. 读取图片
  //capture.open(0);
  if(!srcImg.data)
  { printf("读取图片错误，请确定目录下是否有指定图片存在！\n"); return -1;}
  else{
	  // imshow("Original Image",srcImg);
      //-- 3. 对当前帧使用分类器（Apply the classifier to the frame）
	  //if( !srcImg.empty() )
	   detectAndDisplay(srcImg);}
       waitKey(0);
      //else
       //{ printf(" --(!) No captured frame -- Break!"); break; }

      //int c = waitKey(10);
     // if( (char)c == 'c' ) { break; }


  return 0;
}
/*void UnifyPic()
{
   char filename[100];
   char windowname[100];
   int NumOfPic=14;
   for(int i=1;1<=NumOfPic;i++)

   {  sprintf(filename,"F:\TestPic\%d.jpg",i);
	  sprintf(windowname,"window%d.jpg",i);
	  Mat src=imread(filename,1);
	  Mat temp=src;
	  namedWindow(windowname,WINDOW_AUTOSIZE);
	 // imshow(windowname,src);
	  // 重定义大小
      Mat dst=Mat::zeros(180,180,CV_8UC3);
	  resize(temp,dst,dst.size());
	  //灰度化
	  Mat dst_gray;
	  cvtColor( dst, dst_gray, COLOR_BGR2GRAY );
	  imwrite("filename",dst_gray);
   }
}
*/
void detectAndDisplay( Mat frame )
{ //-----------------检测加载图片的人脸并保存起来------------------------------------------
   std::vector<Rect> faces;
   Mat frame_gray;
   //转换为灰度图
   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
  // imshow("Gray Image",frame_gray);
   //直方图均衡
   //equalizeHist( frame_gray, frame_gray );
   //-- 人脸检测
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(60, 60) );
   //标志出人脸并保存
   Mat faceROI;
   for( size_t i = 0; i < faces.size(); i++ )
    {
      Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
      ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
      faceROI = frame_gray( faces[i] );
   }
   //将脸部尺寸定义为180*180的dst_face
   // Mat dst_face=Mat::zeros(180,180,CV_8UC1);
	//resize(faceROI,dst_face,dst_face.size());
	 Mat dst_face;
	 resize(faceROI,dst_face,Size(100,100));
   //-- 显示最终效果图
    //imshow( window_name, dst_face);
	//保存得到的人脸
	char filefacename[100]={0};
	sprintf( filefacename,"F:\\GetFace\\face.jpg");
    imwrite(filefacename,dst_face);
   //——------------连续载入图片库中的图片统一大小并灰度化——----------------------------------------
	char filename[100]={0};
	char filename1[100]={0};
	char windowname[100]={0};
    const int NumOfPic=50;
for(int i=1;i<=NumOfPic;i++)

  {   sprintf(filename,"F:\\TestPic1\\%d.jpg",i);
	  //sprintf(windowname,"window%d.jpg",i);
	  Mat srclib=imread(filename,1);
	  //namedWindow(windowname,WINDOW_AUTOSIZE);
	 // imshow(windowname,src);
	   std::vector<Rect> faceslib;
       Mat srclib_gray;
    //转换为灰度图
      cvtColor( srclib, srclib_gray, COLOR_BGR2GRAY );
     //imshow("Gray Image",frame_gray);
    //直方图均衡
     //equalizeHist( srclib_gray, srclib_gray );
	  //检测库中人脸
	  face_cascade1.detectMultiScale( srclib_gray, faceslib, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(60, 60) );
	  Mat faceROIlib = srclib_gray( faceslib[0] );
	  // 重定义大小
     // Mat dstlib=Mat::zeros(180,180,CV_8UC1);
	  //resize(faceROIlib,dstlib,dstlib.size());
	   Mat dstlib;
	   resize(faceROIlib,dstlib,Size(100,100));
	  //灰度化
	  //Mat dstlib_gray;
	  //cvtColor( dstlib, dstlib_gray, COLOR_BGR2GRAY );
	  sprintf(filename1,"F:\\TestPicNormal\\%d.jpg",i);
	  imwrite(filename1,dstlib);
   }
  
   //对比待检测图片和图片库，找出最为相似的图片
   long long int count[NumOfPic]={0};
   for(int i=1;i<=NumOfPic;i++)
   {
	   sprintf(filename,"F:\\TestPicNormal\\%d.jpg",i);
	   Mat src=imread(filename,0);
	  // Mat error1;
	  // cv::absdiff(src,dst_face,error1);
	   int rowNumber=src.rows;
	   int colNumber=src.cols*src.channels();
	   const int totalNumber=rowNumber*colNumber;
	   int colNumber1=dst_face.cols*dst_face.channels();
	   for(int r=0;r<rowNumber;r++)
	   {   
		   uchar *data_src=src.ptr<uchar>(r);
		  //  cout<<"dst_face="<<dst_face<<";"<<endl<<endl;
		   uchar *data_det=dst_face.ptr<uchar>(r);
		  // cout<<"dst_det="<<dst_face<<";"<<endl<<endl;
		   //printf("%d\n",dst_face.size);
		  // printf("%d",src.size);
		   for(int t=0;t<totalNumber;t++)
		   {  
			  long int error=data_src[t]-data_det[t];
			 // printf("%d",data_src[t]);
			 // printf("%d",data_det[t]);
			  if(error!=0)
			  {
				 count[i-1]+=abs(error);
			  } 
		   }
	   }
   }
   //找出最匹配的一幅图
   long long int min=count[0];
   int index=0;
   for(int s=1;s<=NumOfPic;s++)
   {
     if(count[s-1]<min)
		{ min=count[s-1];
	 index=s;}
   }
   //陈述结果
   imshow("待检测人脸",frame);
   imshow("人脸识别",dst_face);
   sprintf(filename,"F:\\TestPicNormal\\%d.jpg",index);
   Mat MatchPic=imread(filename);
   imshow("匹配人脸",MatchPic);
}
