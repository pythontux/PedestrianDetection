//OpenCV libraries
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

//C++ libraries
#include <stdio.h>
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

//Namespaces
using namespace cv;
using namespace std;


String pedestrian_cascade_name = "haarcascade_fullbody.xml";
CascadeClassifier pedestrian_cascade;


Mat ROI;
Mat gray_frame;	//Frame

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


void DetectPedestrianViolaJones(vector<Rect> &pedestrian_vector){
	/*
	 * Parámetros de la cascada:
	 * Factor de escala 1.1
	 * 3 minNeighbours
	 * Tamaño de peatón mínimo 30x30
	 */
	equalizeHist( gray_frame, gray_frame );
	pedestrian_cascade.detectMultiScale( gray_frame, pedestrian_vector, 1.1, 1, 0|CASCADE_SCALE_IMAGE, Size(40, 40));
	if (!pedestrian_vector.empty())
    cout << "Peatón detectado VJ= " << Mat(pedestrian_vector) << endl << endl;
	else
		cout<<"Vacio"<<endl;
}


void DetectPedestrianHOG(vector<Rect> VJ_pedestrian_vector, vector<Rect> &pedestrian_vector, HOGDescriptor hog){
	//To avoid the rectangles to remain on the video
	if(VJ_pedestrian_vector.empty())
		pedestrian_vector.clear();

	for( size_t i = 0; i < VJ_pedestrian_vector.size(); i++ ){
		if((VJ_pedestrian_vector[i].width>64)&&(VJ_pedestrian_vector[i].height>128)){
			ROI=gray_frame(VJ_pedestrian_vector[i]);	//	Image with the ROI of that suposed pedestrian

			hog.detectMultiScale(ROI, pedestrian_vector, 0, Size(8,8), Size(0,0),1.05, 2);

			if (!pedestrian_vector.empty()){
				//We update the "global" pedestrian detector calculating the coordinates from the big image origin
				pedestrian_vector[i]=Rect(pedestrian_vector[0].x+VJ_pedestrian_vector[i].x, pedestrian_vector[0].y+VJ_pedestrian_vector[i].y,
						pedestrian_vector[0].width, pedestrian_vector[0].height);
				cout << "Peatón detectado HOG= " << Mat(pedestrian_vector) << endl << endl;
			}
		}
	}
}


void DrawPedestrians(vector<Rect> pedestrian_vector){
	/*Dibuja el rectángulo alrededor del supuesto peatón en la ventana de vídeo
	 * Le pasamos como parámetro el frame y el vector de rectángulos de peatones
	 */
	size_t a;
	for( a = 0; a < pedestrian_vector.size(); a++ ){
		//Point Bottom_Right( (pedestrian[a].x + pedestrian[a].width), (pedestrian[a].y + pedestrian[a].height) );
		rectangle(gray_frame, pedestrian_vector[a], Scalar(255,255,255), 1,8,0);

		//printf("Coordenada x: %d \t Coordenada y: %d \n",pedestrian[a].x,pedestrian[a].y);
	}
}



int main(int argc, char** argv){
	//Window playing the sequence
	namedWindow( "Debug window", WINDOW_AUTOSIZE );

	//Open Cascade Classifier
	if( !pedestrian_cascade.load( pedestrian_cascade_name ) ){ printf("--(!)Error loading pedestrian cascade\n"); return -1; };

	//Create instance of HOGDescriptor
	HOGDescriptor hog;
	//Set the SVM for the HOG descriptors
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

#ifdef VIDEO
//Code to process video format

	//string Video = argv[1];
	Mat frame;
	vector<Rect> pedestrian;
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



#else
	//Code to process sequence of images


		vector<Rect> pedestrianVJ;	//Rectangles around each pedestrian (Viola-Jones Module)
		vector<Rect> pedestrian;	//Rectangles around each pedestrian (HOG module)

		int i;
		string path="Video/Peatones/";
		char* image_number;
		string image_name;



		//Start processing frames
		for(i=0;i<=N_FRAMES;i++){

			sprintf(image_number,"%010d",i);
			cout<<path<<image_number<<".png"<<endl;
			image_name=path+(string)image_number+".png";
			gray_frame = imread(image_name.c_str(), IMREAD_GRAYSCALE);

		    if( gray_frame.empty() )                      // Check for invalid input
		    {
		        cout <<  "Could not open or find the image" << endl ;
		        return -1;
		    }


			DetectPedestrianViolaJones(pedestrianVJ);
			DetectPedestrianHOG(pedestrianVJ, pedestrian, hog);
			DrawPedestrians(pedestrian);
#endif


			imshow( "Debug window", gray_frame );
			int c = waitKey(10);
			if( (char)c == 27 )
				return 0;
		}
}

