//OpenCV libraries
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/contrib/contrib.hpp"

//C++ libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <iostream>

//Error codes
#define VIDEO_ERROR 2
#define VIDEO_END 3

//Misc
#define N_FRAMES 339

//Preprocessor defines
//# define VIDEO
//# define SAMPLE_IMAGE
#define SEQUENCE

//Type of detections involved
#define VJ_HOG_DETECTION 1
#define VJ_DETECTION 2
#define HOG_DETECTION 3

//Namespaces
using namespace cv;
using namespace std;

#include "opencv2/contrib/contrib.hpp"
String pedestrian_cascade_name = "haarcascade_fullbody.xml";
CascadeClassifier pedestrian_cascade;


Mat ROI;
Mat left_frame;	//Frame
Mat right_frame;	//Frame
Mat rgb_frame;
Mat v_disparity;
Mat u_disparity;

int ReturnFrame(VideoCapture capture, Mat &img){
	/*-----------------------------------------------------------
	 *Función que devuelve un objeto de tipo Mat (frame) y un código de error
	 *recibiendo un objeto de tipo VideoCapture
	 *------------------------------------------------------------
	 */
    if (!capture.isOpened())
        return VIDEO_ERROR;
    capture>>img;
    if ((img).empty())
    	return VIDEO_END;
    return 0;
}

void Stereo_SGBM(){
    int SADWindowSize = 0, numberOfDisparities;
    float scale = 1.f ;
    //Stereo disparity using Semi-global block matching algorithm
    StereoSGBM sgbm;
    Mat img1, img2;
    //En caso de procesar la disparidad con imágenes más pequeñas
    if( scale != 1.f )
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(left_frame, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(right_frame, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    else{
    	img1=left_frame;
    	img2=right_frame;
    }

    Size img_size = img1.size();
    Rect roi1, roi2;
    Mat Q;

    numberOfDisparities = 16; //((img_size.width/8) + 15) & -16;	//Hace la primera operación y redondea haciendo el LSNibble cero (divisible entre 16)

    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = 5;

    int cn = img1.channels();

    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 1;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 100;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = false;

    Mat disp;	// Será del tipo CV_16SC1
	Mat disp8;	//CV_8UC1
	Mat dispu;	//CV_16UC1
	Mat negative_disparities;
    sgbm(img1, img2, disp);	//Ojo: disp tiene los valores de disparidades escalados por 16 para hacerlos enteros
    double min, max;
    cv::minMaxLoc(disp, &min, &max);
    cout<<"Minima disparidad: "<<min<<"\nMaxima disparidad: "<<max<<endl;

    disp.convertTo(disp8, CV_8U);	//Trunca valores (solo lo uso para representar)
    imshow("Disparity image", disp8);

    v_disparity=Mat::zeros(disp.rows,numberOfDisparities*16, CV_8U);
	u_disparity=Mat::zeros(numberOfDisparities*16,disp.cols, CV_8U);

	//u-disparity and v-disparity
	//Compute each col histogram
	uchar* d;
	//int suma_px;
    for(int i = 0; i < disp8.rows; i++)
    {
    	//suma_px=0;
        d = disp8.ptr<uchar>(i);	//d[j] is the disparity indicated by the pixel
        for (int j = 0; j < disp8.cols; j++)
        {
        	u_disparity.at<uchar>(d[j],j)++;
        	v_disparity.at<uchar>(i,d[j])++;

        }
    }

    Mat u_treshold;

    imshow("v-disparity", v_disparity);
    imshow("u-disparity", u_disparity);

    //cout<<Mat(u_disparity);
    cout<<"Columnas de u-disparity"<<u_disparity.cols<<endl;
    cout<<"Columnas de disparity 8 bits"<<disp8.cols<<endl;
    cout<<(int)u_disparity.at<char>(300,1180)<<endl;
    cout<<(int)u_disparity.at<char>(10,500)<<endl;

    Mat u_obstaculos;
    Mat u_closed;
    Mat u_eroded;
    
    //Erosión de las líneas horizontales
    
    Mat element = getStructuringElement( MORPH_CROSS, Size(10,10), Point(5,5) );
    morphologyEx( u_disparity, u_closed, MORPH_CLOSE, element );
    
    //Cierre para cerrar las nubes de puntos
    Mat element = getStructuringElement( MORPH_CROSS, Size(10,10), Point(5,5) );
    morphologyEx( u_disparity, u_closed, MORPH_CLOSE, element );

    threshold( u_closed, u_obstaculos, 60, 255,0 );

    imshow("closed_u-disparity", u_closed);
    imshow("obstaculos_u-disparity", u_obstaculos);





}

void DetectPedestrianViolaJones(vector<Rect> &pedestrian_vector){
	/*
	 * Parámetros de la cascada:
	 * Factor de escala 1.1
	 * 3 minNeighbours
	 * Tamaño de peatón mínimo 30x30
	 */
	equalizeHist( left_frame, left_frame );
	pedestrian_cascade.detectMultiScale( left_frame, pedestrian_vector, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(50, 50));
	if (!pedestrian_vector.empty())
    cout << "Peatón detectado VJ= " << Mat(pedestrian_vector) << endl << endl;
	else
		cout<<"Vacio"<<endl;

	//Increase detection window size in order to increase the cases processed by the HOG detection (>64x128)
	for( size_t i = 0; i < pedestrian_vector.size(); i++ ){
		if ((pedestrian_vector[i].x+cvRound(pedestrian_vector[i].width*1.5))<left_frame.cols && (pedestrian_vector[i].x-cvRound(pedestrian_vector[i].width*0.5))>0){
			pedestrian_vector[i].x-=cvRound(pedestrian_vector[i].width*0.5);
			pedestrian_vector[i].width = cvRound(pedestrian_vector[i].width*2);
		}
		if ((pedestrian_vector[i].y+cvRound(pedestrian_vector[i].height*1.5))<left_frame.rows && (pedestrian_vector[i].y-cvRound(pedestrian_vector[i].height*0.5))>0){
			pedestrian_vector[i].y-=cvRound(pedestrian_vector[i].height*0.5);
			pedestrian_vector[i].height = cvRound(pedestrian_vector[i].height*2);
		}
	}
	cout << "Peatón detectado VJp= " << Mat(pedestrian_vector) << endl << endl;
}

void DetectPedestrianHOG(vector<Rect> VJ_pedestrian_vector, vector<Rect> &pedestrian_vector, HOGDescriptor hog){
	//To avoid the rectangles to remain on sucesive frames
	if(VJ_pedestrian_vector.empty())
		pedestrian_vector.clear();

	for( size_t i = 0; i < VJ_pedestrian_vector.size(); i++ ){
		if((VJ_pedestrian_vector[i].width>64)&&(VJ_pedestrian_vector[i].height>128)){
			ROI=left_frame(VJ_pedestrian_vector[i]);	//	Image with the ROI of that suposed pedestrian

			hog.detectMultiScale(ROI, pedestrian_vector, 0, Size(8,8), Size(0,0),1.05, 2);

			if (!pedestrian_vector.empty()){
				//We update the "global" pedestrian detector calculating the coordinates from the big image origin
				pedestrian_vector[i]=Rect(pedestrian_vector[0].x+VJ_pedestrian_vector[i].x, pedestrian_vector[0].y+VJ_pedestrian_vector[i].y,
						pedestrian_vector[0].width*1.1, pedestrian_vector[0].height*1.1);
				cout << "Peatón detectado HOG= " << Mat(pedestrian_vector) << endl << endl;
			}
		}
	}
}

void DetectPedestrianHOGnotROI(vector<Rect> &pedestrian_vector, HOGDescriptor hog){
			hog.detectMultiScale(left_frame, pedestrian_vector, 0, Size(8,8), Size(0,0),1.05, 2);
}

void DrawPedestrians(vector<Rect> pedestrian_vector){
	/*Dibuja el rectángulo alrededor del supuesto peatón en la ventana de vídeo
	 * Le pasamos como parámetro el frame y el vector de rectángulos de peatones
	 */
	size_t a;
	for( a = 0; a < pedestrian_vector.size(); a++ ){
		//Point Bottom_Right( (pedestrian[a].x + pedestrian[a].width), (pedestrian[a].y + pedestrian[a].height) );
		rectangle(left_frame, pedestrian_vector[a], Scalar(255,255,255), 1,8,0);
		//printf("Coordenada x: %d \t Coordenada y: %d \n",pedestrian[a].x,pedestrian[a].y);
	}
}

void SaveImage(vector<Rect> pedestrian_vector, int detector, int num_pedestrian){
	Mat pedestrian;
	char image_number[12];
	string result_path;
	for (size_t i = 0; i<pedestrian_vector.size();i++){
		pedestrian=left_frame(pedestrian_vector[i]);
		switch (detector){
		case VJ_DETECTION:
			result_path="Pedestrians/VJ/";
			sprintf(image_number,"%010d",(int)(num_pedestrian+i));
			imwrite( result_path+(string)image_number+".jpg", pedestrian );
			break;
		case VJ_HOG_DETECTION:
			result_path="Pedestrians/VJ_HOG/";
			sprintf(image_number,"%010d",(int)(num_pedestrian+i));
			imwrite( result_path+(string)image_number+".jpg", pedestrian );
			break;
		case HOG_DETECTION:
			result_path="Pedestrians/HOG/";
			sprintf(image_number,"%010d",(int)(num_pedestrian+i));
			imwrite( result_path+(string)image_number+".jpg", pedestrian );
			break;
		}
	}
}


int main(int argc, char** argv){
	//Window playing the sequence
	namedWindow( "Debug window", WINDOW_AUTOSIZE );
    namedWindow("Disparity image", WINDOW_AUTOSIZE);
    namedWindow("v-disparity", WINDOW_AUTOSIZE);
    namedWindow("u-disparity", WINDOW_AUTOSIZE);

	//Open Cascade Classifier
	if( !pedestrian_cascade.load( pedestrian_cascade_name ) ){ printf("--(!)Error loading pedestrian cascade\n"); return -1; };

	//Create instance of HOGDescriptor
	HOGDescriptor hog;
	//Set the SVM for the HOG descriptors
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	vector<Rect> pedestrianVJ;	//Rectangles around each pedestrian (Viola-Jones Module)
	vector<Rect> pedestrian;	//Rectangles around each pedestrian (HOG module)

#ifdef SAMPLE_IMAGE
	int num_pedestrian_VJ=0, num_pedestrian_VJ_HOG=0, num_pedestrian_HOG;
    if( argc == 1 )
    {
        printf("Usage: PedestrianDetection (<image_filename>)\n");
        return 0;
    }
    left_frame = imread(argv[1], IMREAD_GRAYSCALE);
    right_frame = imread(argv[2], IMREAD_GRAYSCALE);
    if( left_frame.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    if( right_frame.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
	//DetectPedestrianViolaJones(pedestrianVJ);
	//DetectPedestrianHOG(pedestrianVJ, pedestrian, hog);
//    DetectPedestrianHOGnotROI(pedestrian, hog);
//	DrawPedestrians(pedestrian);
	//SaveImage(pedestrianVJ,VJ_DETECTION, num_pedestrian_VJ);
	//SaveImage(pedestrian,VJ_HOG_DETECTION, num_pedestrian_VJ_HOG);
    //SaveImage(pedestrian,HOG_DETECTION, num_pedestrian_HOG);
//	num_pedestrian_VJ+=pedestrianVJ.size();
//	num_pedestrian_VJ_HOG+=pedestrian.size();
    //num_pedestrian_HOG+=pedestrian.size();
	Stereo_SGBM();
    while(1){
#endif


#ifdef VIDEO
//Code to process video format

	//string Video = argv[1];
	Mat frame;

	int errorcode;
	VideoCapture capture("/../PedestrianDetection/Video/OpenCVMat.mp4");
	while(1){
		errorcode = ReturnFrame(capture, frame);
		switch (errorcode){
			case VIDEO_ERROR:
				cout << "Error while trying to open video file " <<endl;

				break;
			case VIDEO_END:
				cout << "I've reached the video end" <<endl;

				break;
			default:
				cout << "Video opened" <<endl;
		}


#endif

#ifdef SEQUENCE
	//Code to process sequence of images




		int i, num_pedestrian_VJ=0, num_pedestrian_VJ_HOG=0, num_pedestrian_HOG=0;
		string path="Video/Peatones/";
		char image_number[12];
		string image_name_left, image_name_right;
		string image_dimensions;



		//Start processing frames
		for(i=0;i<=N_FRAMES;i++){

			sprintf(image_number,"%010d",i);
			cout<<path<<image_number<<".png"<<endl;
			image_name_left=path+"left/"+(string)image_number+".png";
			image_name_right=path+"right/"+(string)image_number+".png";
			left_frame = imread(image_name_left.c_str(), IMREAD_GRAYSCALE);
			right_frame = imread(image_name_right.c_str(), IMREAD_GRAYSCALE);
			printf("Image cols %d \t",left_frame.cols);
			printf("Image rows %d\n",left_frame.rows);


		    if( left_frame.empty() )                      // Check for invalid input
		    {
		        cout <<  "Could not open or find the left image" << endl ;
		        return -1;
		    }

		    if( right_frame.empty() )                      // Check for invalid input
		    {
		        cout <<  "Could not open or find the right image" << endl ;
		        return -1;
		    }


			//DetectPedestrianViolaJones(pedestrianVJ);
			//DetectPedestrianHOG(pedestrianVJ, pedestrian, hog);
		    //DetectPedestrianHOGnotROI(pedestrian, hog);
			//DrawPedestrians(pedestrian);
			//SaveImage(pedestrianVJ,VJ_DETECTION, num_pedestrian_VJ);
			//SaveImage(pedestrian,VJ_HOG_DETECTION, num_pedestrian_VJ_HOG);
		    //SaveImage(pedestrian,HOG_DETECTION, num_pedestrian_HOG);
			//num_pedestrian_VJ+=pedestrianVJ.size();
			//num_pedestrian_VJ_HOG+=pedestrian.size();
		    //num_pedestrian_HOG+=pedestrian.size();
		    Stereo_SGBM();

#endif


			imshow( "Debug window", left_frame );
			int c = waitKey(10);
			if( (char)c == 27 )
				return 0;
		}

}




