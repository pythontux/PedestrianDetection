/*************************************************************************************
 * Pedestrian detection from moving vehicle using disparity image + HOG +SVM
 * Author: Álvaro Gregorio Gómez
 * Date: 04/11/2015
 *************************************************************************************/

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

//Parametres
#define BUILDINGS_LINES_LIFE 15
#define NEIGHBOURHOOD_OBSTACLE_X 200
#define NEIGHBOURHOOD_OBSTACLE_Y 20

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

int Stereo_SGBM(vector<Rect> &ROI_disparity){
    int SADWindowSize = 0, numberOfDisparities;
    float scale = 1.f ;
    //Stereo disparity using Semi-global block matching algorithm
    StereoSGBM sgbm;
    Mat img1, img2;

    //En caso de querer escalar las imágenes de entrada
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

/**************************************************
 * Parámetros del SGBM(Semi-Global Block matching)
 *************************************************/

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

    //Matrices de disparidades
    Mat disp;	// Será del tipo CV_16SC1
	Mat disp8;	//CV_8UC1

	//Generamos la imagen de disparidades
    sgbm(img1, img2, disp);	//Ojo: disp tiene los valores de disparidades escalados por 16 para hacerlos enteros

    //Identifico máximo y mínimo (para depurar)
    double min, max;
    cv::minMaxLoc(disp, &min, &max);
    cout<<"Minima disparidad: "<<min<<"\nMaxima disparidad: "<<max<<endl;

    disp.convertTo(disp8, CV_8U);	//No trunca valores ya que el numberOfDisparities es 16
//    imshow("Disparity image", disp8);

    //Histogramas por filas y por columnas
    v_disparity=Mat::zeros(disp.rows,numberOfDisparities*16, CV_8U);
	u_disparity=Mat::zeros(numberOfDisparities*16,disp.cols, CV_8U);

	//Calculamos u-disparity and v-disparity
	uchar* d;
    for(int i = 0; i < disp8.rows; i++)
    {
        d = disp8.ptr<uchar>(i);	//d[j] es el valor de disparidad de ese pixel
        for (int j = 0; j < disp8.cols; j++)
        {
        	u_disparity.at<uchar>(d[j],j)++;
        	v_disparity.at<uchar>(i,d[j])++;
        }
    }


    imshow("v-disparity", v_disparity);
//    imshow("u-disparity", u_disparity);

    /**************************************************************
     * Detección de perfil de carretera por Hough lines
     **************************************************************/
    Mat v_disparity_thresholded, v_disparity_thresholded_opened, v_disparity_thresholded_final;
    threshold( v_disparity, v_disparity_thresholded, 20, 127,0 );	//threshold anterior: 35

    //Erosion con un kernel horizontal para eliminar lineas verticales (obstáculos)
    Mat element_v_erode = getStructuringElement( MORPH_RECT, Size(7,1));
    morphologyEx( v_disparity_thresholded, v_disparity_thresholded_opened, MORPH_ERODE, element_v_erode );

    //Dilatacion con un kernel rectangular para juntar línea carretera
    Mat element_v_close = getStructuringElement( MORPH_RECT, Size(7,7));
    morphologyEx( v_disparity_thresholded_opened, v_disparity_thresholded_final, MORPH_DILATE, element_v_close );


    vector<Vec4i> lines_v_disp;
    static float m_s, b_s;	//y=mx+b ecuacion de la carretera
    bool equation_set=false;
    HoughLinesP( v_disparity_thresholded_final, lines_v_disp, 8, CV_PI/18, 900, 130, 10 );
    for( size_t i = 0; (i < lines_v_disp.size() && !equation_set); i++ )
    {
    	if (lines_v_disp[i][0]!=lines_v_disp[i][2]){	//Evita las líneas verticales
    		line( v_disparity_thresholded, Point(lines_v_disp[i][0], lines_v_disp[i][1]),
    				Point(lines_v_disp[i][2], lines_v_disp[i][3]), Scalar(255,255,255), 1, 8 );
    		//Calculo provisionalmente la pendiente y b con la primera recta que aparezca
    		m_s=(lines_v_disp[i][3]-lines_v_disp[i][1])/float(lines_v_disp[i][2]-lines_v_disp[i][0]);
    		b_s=lines_v_disp[i][1]-m_s*lines_v_disp[i][0];
    		equation_set=true;

    	}
    }
    namedWindow( "v-disparity Hough lines", 1 );
    imshow( "v-disparity Hough lines", v_disparity_thresholded );
    namedWindow( "v-disparity opened Hough lines", 1 );
    imshow( "v-disparity opened Hough lines", v_disparity_thresholded_final );



	/************************************************************
	 * Tratamiento de la u-disparity para identificar obstáculos
	 ************************************************************/

    Mat u_obstaculos, u_obstaculos_no_cierre;
    Mat u_obstaculos_cierre;
    Mat u_dilated;
    Mat u_eroded;
    Mat u_edges;

    //Erosionar con un kernel vertical para eliminar lineas horizontales (carretera)
    Mat element = getStructuringElement( MORPH_RECT, Size(1,5));
    morphologyEx( u_disparity, u_eroded, MORPH_ERODE, element );
    imshow("eroded_u-disparity", u_eroded);

    //Dilatación para cerrar las nubes de puntos
    Mat element2 = getStructuringElement( MORPH_RECT, Size(7,5));
    morphologyEx( u_eroded, u_dilated, MORPH_DILATE, element2 );
    imshow("eroded_dilated_u-disparity", u_dilated);

    //Binarizo
    threshold( u_eroded, u_obstaculos, 5, 255,0 );
    threshold( u_eroded, u_obstaculos_no_cierre, 5, 255,0 );

    //Cierre para cerrar las nubes de puntos
    Mat element4 = getStructuringElement( MORPH_RECT, Size(30,1));
    morphologyEx( u_obstaculos, u_obstaculos_cierre, MORPH_CLOSE, element4 );

    //Visualización
    imshow("obstaculos cierre u-disparity", u_obstaculos);
//    imshow("obstaculos_u-disparity", u_obstaculos);
//    imshow("obstaculos y cierre_u-disparity", u_obstaculos_cierre);


    /**************************************************************
     * Detección de edificios por Hough lines oblicuas en u-disparity
     **************************************************************/
    //Estimo edificios laterales si es que los hay
    vector<Vec4i> lines_buildings;
    static vector<Vec4i> buildings;	//Vector con máximo dos elementos (edificios)
    static int frames_no_l_building = 0, frames_no_r_building = 0;
    buildings.reserve(2);
    HoughLinesP( u_obstaculos, lines_buildings, 5, CV_PI/30, 65, 400, 40 );
    bool left_found = false, right_found = false;
    for( size_t i = 0; (i < lines_buildings.size()) && !left_found && !right_found; i++ )
    {
    	//Dont look for horizontal nor vertical lines
    	if ((abs((lines_buildings[i][1]-lines_buildings[i][3])/double(lines_buildings[i][0]-lines_buildings[i][2]))<1.7) &&
    			(abs((lines_buildings[i][1]-lines_buildings[i][3])/double(lines_buildings[i][0]-lines_buildings[i][2]))>0.05)){
    		if(((lines_buildings[i][0] < (int)(u_disparity.cols/2.0)) || (lines_buildings[i][2] < (int)(u_disparity.cols/2.0))) && left_found == false){
    			for (int a=0; a<4; a++){
    				buildings[0][a] = lines_buildings[i][a];//Borde izquierdo de la carretera
    			}
    			left_found = true;
    		}

    		if(((lines_buildings[i][0] > (int)(u_disparity.cols/2.0)) || (lines_buildings[i][2] > (int)(u_disparity.cols/2.0)))&& right_found == false){
    			for (int a=0; a<4; a++){
    				buildings[1][a] = lines_buildings[i][a];	//Borde derecho de la carretera
    			}
    			right_found = true;
    		}
    	}

    }
	if(!left_found)
		frames_no_l_building++;
	else
		frames_no_l_building=0;
	if(!right_found)
		frames_no_r_building++;
	else
		frames_no_r_building=0;

	if(frames_no_l_building>BUILDINGS_LINES_LIFE)
		buildings[0]=Scalar(0,0,0,0);
	if(frames_no_r_building>BUILDINGS_LINES_LIFE)
		buildings[1]=Scalar(0,0,0,0);

    /**************************************************************
     * Parámetros de rectas definiendo edificios
     * y=mx+b (l->left; r->right)
     **************************************************************/
    float m_l, b_l, m_r, b_r;

    //m=(y2-y1)/(x2-x1)
    //b=y1-m*x1

    if(buildings[0]!=(Vec4i)Scalar(0,0,0,0)){
        m_l=(buildings[0][3]-buildings[0][1])/float(buildings[0][2]-buildings[0][0]);
        if(m_l>0){
        	m_l=0;
        	buildings[0]=Scalar(0,0,0,0);
        }
        b_l=buildings[0][1]-m_l*buildings[0][0];
    }
    else{
    	m_l=0;
    	b_l=0;
    }

    if(buildings[1]!=(Vec4i)Scalar(0,0,0,0)){
    	m_r=(buildings[1][3]-buildings[1][1])/float(buildings[1][2]-buildings[1][0]);
        if(m_r<0){
        	m_r=0;
        	buildings[1]=Scalar(0,0,0,0);
        }
    	b_r=buildings[1][1]-m_l*buildings[1][0];
    }
    else{
    	m_r=0;
    	b_r=0;
    }

    /**************************************************************
     * Muestro las dos lineas de edificios si es que las hay
     **************************************************************/

	if(buildings[0]!=(Vec4i)Scalar(0,0,0,0))
		line( u_disparity, Point(buildings[0][0], buildings[0][1]),
				Point(buildings[0][2], buildings[0][3]), Scalar(255,255,255), 1, 8 );
	if(buildings[1]!=(Vec4i)Scalar(0,0,0,0))
		line( u_disparity, Point(buildings[1][0], buildings[1][1]),
				Point(buildings[1][2], buildings[1][3]), Scalar(255,255,255), 1, 8 );

//    namedWindow( "Buildings Hough lines", 1 );
//    imshow( "Buildings Hough lines", u_disparity );

    /**************************************************************
     * Detección de obstáculos por Hough lines horizontales en u-disparity
     **************************************************************/

    vector<Vec4i> lines_obstacles, obstacles;
    obstacles.reserve(30);
    int indice=0;
    HoughLinesP( u_obstaculos, lines_obstacles, 20, CV_PI/16, 150, 25, 25 );	//u_obstacles threshold 150
    for( size_t i = 0; (i < lines_obstacles.size() && indice<30); i++ ){
    	if((lines_obstacles[i][1] == lines_obstacles[i][3]) && (lines_obstacles[i]!=(Vec4i)Scalar(0,0,0,0))){
    		//Calculo de puntos de lineas de objeto y edificios
    		int y_bl, y_o, x_ol, y_br, x_or;	//b->Building	o->Obstacle
    		y_o=lines_obstacles[i][1];

    		x_ol=std::min(lines_obstacles[i][0], lines_obstacles[i][2]);	//topleft of the object
    		x_or=std::max(lines_obstacles[i][0], lines_obstacles[i][2]);	//topright of the object

    		y_bl=(int)(m_l*x_ol+b_l);
    		y_br=(int)(m_r*x_or+b_r);

    		//Comprobación de que están por debajo de las lineas de la carretera
    		if(((y_o>y_bl)||(!m_l && !b_l)) && ((y_o>y_br)||(!m_r && !b_r))){	//Ojo con las restricciones, darse cuenta de que el eje y de la imagen mira hacia abajo
				//obstacles[indice]=lines_obstacles[i];
    			for (int a=0; a<4; a++){
    				obstacles[indice][a] = lines_obstacles[i][a];//Array de obstáculos
    			}

				line( u_disparity, Point(obstacles[indice][0], obstacles[indice][1]),
					Point(obstacles[indice][2], obstacles[indice][3]), Scalar(255,255,255), 1, 8 );

				//Pongo a cero lineas cercanas
				for(int j=0; j < lines_obstacles.size();j++){
					//Busco todas las líneas horizontales, diferentes de cero y a una distancia menor de 15+15 y las marco como cero
					if((lines_obstacles[j][1] == lines_obstacles[j][3]) && (lines_obstacles[j]!=(Vec4i)Scalar(0,0,0,0)) && (lines_obstacles[j]!=lines_obstacles[i])){
						if((abs(lines_obstacles[j][1]-lines_obstacles[i][1])<NEIGHBOURHOOD_OBSTACLE_Y)&& (abs(lines_obstacles[j][0]-lines_obstacles[i][0])<NEIGHBOURHOOD_OBSTACLE_X) && (abs(lines_obstacles[j][2]-lines_obstacles[i][2])<NEIGHBOURHOOD_OBSTACLE_X)){
							lines_obstacles[j]=Scalar(0,0,0,0);
						}
					}
				}
				indice++;
    		}
    	}
    }
    int n_obstacles = indice;

    namedWindow( "obstacles Hough lines", 1 );
    imshow( "obstacles Hough lines", u_disparity );


    cout<<"n_obstacles: "<<n_obstacles<<endl;

    /********************************************************************************************
     * Sacar límites verticales del objeto en la v-disparity y guardar toda la info en vect<Rect>
     ********************************************************************************************/

    int y_base, y_top, x_left, x_right;
    vector<Rect> ROI_disparity_inside_function;
    ROI_disparity_inside_function.reserve(30);
    for(int i=0; i<n_obstacles; i++){

    	int disparidad_objeto, h;
    	int j;
    	//disparidad y h cogeremos los máximos valores
    	//h es la altura en px
    	disparidad_objeto = obstacles[i][1];	//La disparidad es directamente el eje y de la u-disparity
    	h = (int)u_disparity.at<uchar>(obstacles[i][0], obstacles[i][1]);	//Lo inicializo a un valor cualquiera
    	//Cojo el máximo valor de disparidad (límite inferior mas bajo)
    	for(j=(int)std::min(obstacles[i][0], obstacles[i][2]);j<=(int)std::max(obstacles[i][0], obstacles[i][2]);j++){
    		//Considero la maxima altura en un vecindario vertical para tolerancia a un desplazamiento de la recta
    		for(int p=-15;p<=15;p++){
    			if((obstacles[i+p][1]>0) && (obstacles[i+p][1]<u_disparity.rows)){
    				h = (int)std::max((int)u_disparity.at<uchar>(j, obstacles[i+p][1]), h);
    			}
    		}
    	}
    	//Calculo el valor de la base de la caja mediante la ecuación del perfil de la carretera y el valor de disparidad (x->disparidad)
    	//y=mx+b
    	y_base=(int)(m_s*disparidad_objeto+b_s);
    	if (y_base>disp8.rows){
    		y_base = disp8.rows;	//Trunco para evitar valores mayores que la dimension vertical (ver forma de la recta en la v-disparity)
    	}
    	y_top=y_base-h;
    	if (y_top<0){
    		y_top = 0;	//Trunco para evitar valores negativos debidos a mala estimacion Hough
    	}
    	x_left=std::min(obstacles[i][0], obstacles[i][2]);
		x_right=std::max(obstacles[i][0], obstacles[i][2]);
    	//Creo el rectángulo que define la ROI
    	ROI_disparity[i]=Rect(x_left, y_top, (x_right-x_left), h);
    	cout<<"ROIs"<<ROI_disparity[i]<<endl;
    }
    return n_obstacles;

    /*************************************************************
     * Reduzco el número de líneas asociando líneas cercanas
     ************************************************************/



    /**************************************************************
     * Pruebas para sacar contornos (Detectar y separar obstáculos)
     **************************************************************/

/*
    Mat element3 = getStructuringElement( MORPH_RECT, Size(3,5), Point(2,3) );
    morphologyEx( u_obstaculos_cierre, u_edges, MORPH_DILATE, element3 );

    u_edges = u_edges - u_obstaculos_cierre;

    threshold( u_edges, u_edges, 5, 255,0 );
    imshow("bordes_u-disparity", u_edges);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(u_edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    for (int contour=0;(contour<contours.size()); contour++){
    	Scalar color(rand()&0xFF,rand()&0xFF,rand()&0xFF);
    	drawContours(u_eroded, contours, contour, Scalar(255,255,255), 1, 8, hierarchy);
    }

    imshow("contornos_u-disparity", u_eroded);*/



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

void DrawPedestrians(vector<Rect> &pedestrian_vector, int n_ROI){
	/*Dibuja el rectángulo alrededor del supuesto peatón en la ventana de vídeo
	 * Le pasamos como parámetro el frame y el vector de rectángulos de peatones
	 */
	size_t a;
	for( a = 0; a < n_ROI; a++ ){
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

	vector<Rect> ROI_disparity;
	ROI_disparity.reserve(30);
	int n_ROI;

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
	Stereo_SGBM(ROI_disparity);
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
		    n_ROI = Stereo_SGBM(ROI_disparity);
		    DrawPedestrians(ROI_disparity, n_ROI);

#endif


			imshow( "Debug window", left_frame );
			int c = waitKey(10);
			if( (char)c == 27 )
				return 0;
			else if((char)c == 's'){	//s stops playing the video
				int c = waitKey(0);
			}
		}

}


