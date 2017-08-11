#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv/cv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

//--------------------------------��ȫ�ֱ���������----------------------------------------------
//		����������ȫ�ֱ���
//-------------------------------------------------------------------------------------------------
//ע�⣬��Ҫ��"haarcascade_frontalface_alt.xml"��"haarcascade_eye_tree_eyeglasses.xml"�������ļ����Ƶ�����·����
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
  //��������ͼƬ
  Mat srcImg=imread("1.jpg");
  //-- 1. ���ؼ�����cascades��
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !face_cascade1.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //-- 2. ��ȡͼƬ
  //capture.open(0);
  if(!srcImg.data)
  { printf("��ȡͼƬ������ȷ��Ŀ¼���Ƿ���ָ��ͼƬ���ڣ�\n"); return -1;}
  else{
	  // imshow("Original Image",srcImg);
      //-- 3. �Ե�ǰ֡ʹ�÷�������Apply the classifier to the frame��
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
	  // �ض����С
      Mat dst=Mat::zeros(180,180,CV_8UC3);
	  resize(temp,dst,dst.size());
	  //�ҶȻ�
	  Mat dst_gray;
	  cvtColor( dst, dst_gray, COLOR_BGR2GRAY );
	  imwrite("filename",dst_gray);
   }
}
*/
void detectAndDisplay( Mat frame )
{ //-----------------������ͼƬ����������������------------------------------------------
   std::vector<Rect> faces;
   Mat frame_gray;
   //ת��Ϊ�Ҷ�ͼ
   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
  // imshow("Gray Image",frame_gray);
   //ֱ��ͼ����
   //equalizeHist( frame_gray, frame_gray );
   //-- �������
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(60, 60) );
   //��־������������
   Mat faceROI;
   for( size_t i = 0; i < faces.size(); i++ )
    {
      Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
      ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
      faceROI = frame_gray( faces[i] );
   }
   //�������ߴ綨��Ϊ180*180��dst_face
   // Mat dst_face=Mat::zeros(180,180,CV_8UC1);
	//resize(faceROI,dst_face,dst_face.size());
	 Mat dst_face;
	 resize(faceROI,dst_face,Size(100,100));
   //-- ��ʾ����Ч��ͼ
    //imshow( window_name, dst_face);
	//����õ�������
	char filefacename[100]={0};
	sprintf( filefacename,"F:\\GetFace\\face.jpg");
    imwrite(filefacename,dst_face);
   //����------------��������ͼƬ���е�ͼƬͳһ��С���ҶȻ�����----------------------------------------
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
    //ת��Ϊ�Ҷ�ͼ
      cvtColor( srclib, srclib_gray, COLOR_BGR2GRAY );
     //imshow("Gray Image",frame_gray);
    //ֱ��ͼ����
     //equalizeHist( srclib_gray, srclib_gray );
	  //����������
	  face_cascade1.detectMultiScale( srclib_gray, faceslib, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(60, 60) );
	  Mat faceROIlib = srclib_gray( faceslib[0] );
	  // �ض����С
     // Mat dstlib=Mat::zeros(180,180,CV_8UC1);
	  //resize(faceROIlib,dstlib,dstlib.size());
	   Mat dstlib;
	   resize(faceROIlib,dstlib,Size(100,100));
	  //�ҶȻ�
	  //Mat dstlib_gray;
	  //cvtColor( dstlib, dstlib_gray, COLOR_BGR2GRAY );
	  sprintf(filename1,"F:\\TestPicNormal\\%d.jpg",i);
	  imwrite(filename1,dstlib);
   }
  
   //�Աȴ����ͼƬ��ͼƬ�⣬�ҳ���Ϊ���Ƶ�ͼƬ
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
   //�ҳ���ƥ���һ��ͼ
   long long int min=count[0];
   int index=0;
   for(int s=1;s<=NumOfPic;s++)
   {
     if(count[s-1]<min)
		{ min=count[s-1];
	 index=s;}
   }
   //�������
   imshow("���������",frame);
   imshow("����ʶ��",dst_face);
   sprintf(filename,"F:\\TestPicNormal\\%d.jpg",index);
   Mat MatchPic=imread(filename);
   imshow("ƥ������",MatchPic);
}
