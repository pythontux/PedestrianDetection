//OpenCV libraries
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

//C++ libraries
#include <stdio.h>
#include <string.h>
//#include <ctype.h>
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


int ReturnFrame(VideoCapture& capture, Mat &img){
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


void DetectPedestrianViolaJones(Mat &gray_frame, vector<Rect> &pedestrian_vector){
	/*
	 * Parámetros de la cascada:
	 * Factor de escala 1.1
	 * 3 minNeighbours
	 * Tamaño de peatón mínimo 30x30
	 */
	equalizeHist( gray_frame, gray_frame );
	pedestrian_cascade.detectMultiScale( gray_frame, pedestrian_vector, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(40, 40));
	if (!pedestrian_vector.empty())
    cout << "Peatón detectado = " << Mat(pedestrian_vector) << endl << endl;
	else
		cout<<"Vacio"<<endl;
}


void DrawPedestrians(Mat gray_frame, vector<Rect> pedestrian_vector){
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


int main(int argc, char** argv)
{
#ifdef VIDEO
//Code to process video format

	//string Video = argv[1];
	Mat frame;
	vector<Rect> pedestrian;
	int errorcode;
	VideoCapture capture("/home/alvaro/workspace_opencv/PedestrianDetection/Video/OpenCVMat.mp4");
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


		vector<Rect> pedestrian;	//Rectangles around each pedestrian
		Mat img;	//Frame
		int i;
		string path="/home/alvaro/workspace_opencv/PedestrianDetection/Video/Peatones/";
		char* image_number;

		//Window playing the sequence
		namedWindow( "Debug window", WINDOW_AUTOSIZE );

		if( !pedestrian_cascade.load( pedestrian_cascade_name ) ){ printf("--(!)Error loading pedestrian cascade\n"); return -1; };

		//Start processing frames
		for(i=0;i<=N_FRAMES;i++){

			sprintf(image_number,"%010d",i);
			cout<<path<<image_number<<".png"<<endl;
			img = imread((path+(string)image_number+".png").c_str(), IMREAD_GRAYSCALE);

		    if( img.empty() )                      // Check for invalid input
		    {
		        cout <<  "Could not open or find the image" << endl ;
		        return -1;
		    }


			DetectPedestrianViolaJones(img, pedestrian);
			DrawPedestrians(img, pedestrian);
#endif


			imshow( "Debug window", img );
			int c = waitKey(10);
			if( (char)c == 27 )
				return 0;
		}
}
