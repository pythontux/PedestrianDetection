/*************************************************************************************
* Pedestrian detection from moving vehicle using disparity image + HOG +SVM
 * Author: Ãlvaro Gregorio GÃ³mez
 * Date: 12/12/2015
 * Subject: Laboratorio
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
#define N_FRAMES 240

//Parametros
#define BUILDINGS_LINES_LIFE 15
#define DISPARITY_THRESHOLD 200	//200
#define NEIGHBOURHOOD_OBSTACLE_X 250
#define NEIGHBOURHOOD_OBSTACLE_Y 200
#define MAX_U_DISPARITY_NEIGHBOURHOOD 60//De las 65536 disparidades
#define HEIGHT_OVERSIZE 0
#define ROAD_OBSTACLE_MARGIN 50

//Namespaces
using namespace cv;
using namespace std;

#include "opencv2/contrib/contrib.hpp"
String pedestrian_cascade_name = "haarcascade_fullbody.xml";
CascadeClassifier pedestrian_cascade;


Mat ROI;
Mat left_frame;	//Frame
Mat left_color_frame;
Mat right_frame;	//Frame
Mat disp;	// SerÃ¡ del tipo CV_16SC1
Mat disp_print; //Normalizada
Mat v_disparity;
Mat u_disparity;
Mat v_disparity8;
Mat u_disparity8;
Mat u_disparity_print;
Mat non_free_u_disparity;
Mat u_detect_obstacles;
Mat obstacle_mask, obstacle_mask8;

int Stereo_SGBM(vector<Rect> &ROI_disparity,  vector<Vec4i> &buildings){
/*************************************************************************
 *Esta funciÃ³n rellena un vector de rectÃ¡ngulos con los obstÃ¡culos detectados por tÃ©cnicas
 *de disparidad stereo
 **************************************************************************/
    float scale = 1.f ;
    //Stereo disparity using Semi-global block matching algorithm
    StereoSGBM sgbm;
    Mat img1, img2;

	img1=left_frame;
	img2=right_frame;

    Size img_size = img1.size();

// ParÃ¡metros del SGBM(Semi-Global Block matching)
    int numberOfDisparities = 128; //((img_size.width/8) + 15) & -16;	//Hace la primera operaciÃ³n y redondea haciendo el LSNibble cero (divisible entre 16)
    sgbm.preFilterCap = 61;
    sgbm.SADWindowSize = 5;
    int cn = img1.channels();
    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 1;
    sgbm.numberOfDisparities = numberOfDisparities;	//Con 16 se aprovechan los 8 bits de imÃ¡gen por completo
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 100;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = false;

    //Matrices de disparidades
    Mat disp8;	//CV_8UC1

    //Generamos la imagen de disparidades
    sgbm(img1, img2, disp);	//disp tiene los valores de disparidades escalados por 16
    disp.convertTo(disp, CV_16U);
    disp.convertTo(disp8, CV_8U);
    normalize(disp, disp_print, 0, 255, CV_MINMAX, CV_8U);

    //Identifico mÃ¡ximo y mÃ­nimo (para depurar)
    double min, max;
    cv::minMaxLoc(disp, &min, &max);
    cout<<"Minima disparidad: "<<min<<"\nMaxima disparidad: "<<max<<endl;

    //Histogramas por filas y por columnas
    v_disparity=Mat::zeros(disp.rows,numberOfDisparities*16, CV_16U);
	u_disparity=Mat::zeros(numberOfDisparities*16,disp.cols, CV_16U);

	//Calculamos u-disparity and v-disparity
	ushort* d;
    for(int i = 0; i < disp.rows; i++)
    {
        d = disp.ptr<ushort>(i);	//d[j] es el valor de disparidad de ese pixel
        for (int j = 0; j < disp.cols; j++)
        {
        	v_disparity.at<ushort>(i,(d[j]))++;
        	u_disparity.at<ushort>((d[j]),j)++;
        }
    }


    //Como u and v-disparities tienen muchas filas y columnas respectivamente, Redimensiono

    Mat v_disparity8print;
	Mat u_disparity8print;

	v_disparity.convertTo(v_disparity8, CV_8U);//, 255/65535.0);
	u_disparity.convertTo(u_disparity8, CV_8U);//, 255/65535.0);

	//Las imÃ¡genes que muestro, las redimensiono
	resize(u_disparity8, u_disparity8print, Size(), 1,0.5);
	resize(v_disparity8, v_disparity8print, Size(), 0.5,1);

    imshow("disparity", disp_print);
    //imshow("v-disparity", v_disparity8);
    imshow("u-disparity", u_disparity8print);



    /**************************************************************
     * DetecciÃ³n de perfil de carretera por Hough lines
     **************************************************************/

    //Umbralizado
    Mat v_disparity_thresholded, v_disparity_thresholded_opened, v_disparity_thresholded_final;
    threshold( v_disparity8, v_disparity_thresholded, 3, 127,0 );	//threshold anterior: 35
    //imshow( "v-disparity binarized", v_disparity_thresholded );


    //Erosion con un kernel horizontal para eliminar lineas verticales (obstÃ¡culos)
    Mat element_v_erode = getStructuringElement( MORPH_RECT, Size(2,1));
    morphologyEx( v_disparity_thresholded, v_disparity_thresholded_final, MORPH_ERODE, element_v_erode );


    //ExtracciÃ³n del perfil y dibujo del mismo sobre v-disparity
    vector<Vec4i> lines_v_disp;
    static float m_s, b_s;	//y=mx+b ecuacion de la carretera
    bool equation_set=false;
    HoughLinesP( v_disparity_thresholded_final, lines_v_disp, 8, CV_PI/18, 150, 550, 20 ); //250; 900
    for( size_t i = 0; (i < lines_v_disp.size() && !equation_set); i++ )
    {
    	if (lines_v_disp[i][0]!=lines_v_disp[i][2]){	//Evita las lÃ­neas verticales
    		line( v_disparity_thresholded_final, Point(lines_v_disp[i][0], lines_v_disp[i][1]),
    				Point(lines_v_disp[i][2], lines_v_disp[i][3]), Scalar(255,255,255), 1, 8 );
    		line( v_disparity8, Point(lines_v_disp[i][0], lines_v_disp[i][1]),
    				Point(lines_v_disp[i][2], lines_v_disp[i][3]), Scalar(255,255,255), 1, 8 );
    		//Calculo provisionalmente la pendiente y b con la primera recta que aparezca
    		m_s=(lines_v_disp[i][3]-lines_v_disp[i][1])/float(lines_v_disp[i][2]-lines_v_disp[i][0]);
    		b_s=lines_v_disp[i][1]-m_s*lines_v_disp[i][0];
    		equation_set=true;
    	}
    }
    Mat v_disparity_thresholded_print;
    resize(v_disparity_thresholded_final, v_disparity_thresholded_print, Size(), 0.5,1);
    imshow("v-disparity", v_disparity_thresholded_final);



    /************************************************************************
     * Marco zonas elevadas respecto al perfil de la carretera
     ************************************************************************/

    obstacle_mask=Mat::zeros(disp.rows,disp.cols, CV_16U);
    int disp_road;
    ushort* m;
    for(int i = 0; i < disp.rows; i++)
    {
    	disp_road=(int)((i-b_s)/m_s);	//Calculo la disparidad correspondiente a la carretera d=(y-b)/m
    	d = disp.ptr<ushort>(i);	//d[j] es el valor de disparidad de ese pixel
        m = obstacle_mask.ptr<ushort>(i);    //m[j] es el valor de disparidad del pixel correspondiente de la mÃ¡scara
        for (int j = 0; j < disp.cols; j++)
        {
        	if((int)d[j]>(disp_road+ROAD_OBSTACLE_MARGIN)){
        		m[j]=d[j];
        	}
        }
    }
    normalize(obstacle_mask, obstacle_mask8, 0, 255, CV_MINMAX, CV_8U);
    imshow("v-obstacle_mask", obstacle_mask8);

    uchar* m8;	//8bits mask pointer
    uchar* lf;	//left_frame pointer
    for(int i = 0; i < left_color_frame.rows; i++)
    {
    	lf = left_color_frame.ptr<uchar>(i);	//d[j] es el valor de disparidad de ese pixel
        m8 = obstacle_mask8.ptr<uchar>(i);    //m[j] es el valor de disparidad del pixel correspondiente de la mÃ¡scara
        for (int j = 0; j < left_color_frame.cols; j++)
        {
        	lf[2+3*j]=saturate_cast<uchar>(m8[j]+lf[2+3*j]);	//Le sumo el valor correspondiente al rojo
        }
    }

    /************************************************************************
     * Construyo una non-free u-disparity
     ************************************************************************/
    Mat non_free_u_disparity8, non_free_u_disparity8print;
    non_free_u_disparity=Mat::zeros(u_disparity.rows,u_disparity.cols, CV_16U);
    for(int i = 0; i < disp.rows; i++)
    {
        d = obstacle_mask.ptr<ushort>(i);	//d[j] es el valor de disparidad de ese pixel
        for (int j = 0; j < disp.cols; j++)
        {
        	non_free_u_disparity.at<ushort>((d[j]),j)++;
        }
    }
    non_free_u_disparity.convertTo(non_free_u_disparity8, CV_8U);

	//Las imÃ¡genes que muestro, las redimensiono
	resize(non_free_u_disparity8, non_free_u_disparity8print, Size(), 1,0.5);
	imshow("Free u-disparity", non_free_u_disparity8);


	/***********************************************************************
	 * Tratamiento de la u-disparity para identificar obstÃ¡culos y edificios
	 ***********************************************************************/

    Mat u_obstaculos, u_obstaculos_no_cierre;
    Mat u_obstaculos_cierre;
    Mat u_eroded;
    Mat u_edges;
    Mat u_blur, u_blur_obstacles;

    GaussianBlur(non_free_u_disparity8, u_blur, Size(0,0), 3, 3);


    //Erosionar con un kernel vertical para eliminar lineas horizontales (carretera)
    Mat element = getStructuringElement( MORPH_RECT, Size(1,3));
    morphologyEx( u_blur, u_eroded, MORPH_ERODE, element );
    //imshow("eroded_u-disparity", u_eroded);

    //DilataciÃ³n para cerrar las nubes de puntos
    Mat element2 = getStructuringElement( MORPH_RECT, Size(9,5));
    morphologyEx( u_eroded, u_detect_obstacles, MORPH_DILATE, element2 );
    //imshow("eroded_dilated_u-disparity", u_detect_obstacles);


    //Binarizo
    threshold( u_detect_obstacles, u_obstaculos, 2, 127,0 );
    imshow("binary u-disparity", u_obstaculos);

    Mat element3 = getStructuringElement( MORPH_RECT, Size(13,19));
    morphologyEx( u_obstaculos, u_detect_obstacles, MORPH_DILATE, element3 );
    imshow("binary_dilated_u-disparity", u_detect_obstacles);




    /**************************************************************
     * DetecciÃ³n de edificios por Hough lines oblicuas en u-disparity (deshabilitado)
     **************************************************************/
    //Estimo edificios laterales si es que los hay
    vector<Vec4i> lines_buildings;
    static int frames_no_l_building = 0, frames_no_r_building = 0;
    HoughLinesP( u_blur, lines_buildings, 5, CV_PI/30, 90, 550, 30 );
    bool left_found = false, right_found = false;
    for( size_t i = 0; (i < lines_buildings.size()) && !left_found && !right_found; i++ )
    {
    	//Dont look for horizontal nor vertical lines
    	if ((lines_buildings[i][0]!=lines_buildings[i][2]) && ((lines_buildings[i][1]!=lines_buildings[i][3]) )){
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

    //Remanencia de lÃ­neas en caso de no encontrarlas en el frame actual
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
     * ParÃ¡metros de rectas definiendo edificios
     * y=mx+b (l->left; r->right)
     * m=(y2-y1)/(x2-x1)
     * b=y1-m*x1
     **************************************************************/
    float m_l, b_l, m_r, b_r;
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

		line( u_obstaculos, Point(buildings[0][0], buildings[0][1]),
			Point(buildings[0][2], buildings[0][3]), Scalar(255,255,255), 1, 8 );

	if(buildings[1]!=(Vec4i)Scalar(0,0,0,0))

		line( u_obstaculos, Point(buildings[1][0], buildings[1][1]),
			Point(buildings[1][2], buildings[1][3]), Scalar(255,255,255), 1, 8 );

    Mat u_obstaculos_print;


    //imshow( "Buildings Hough lines", u_obstaculos_print );


    /**************************************************************
     * DetecciÃ³n de obstÃ¡culos por Hough lines horizontales en u-disparity
     **************************************************************/

    vector<Vec4i> lines_obstacles, obstacles;
    obstacles.reserve(40);
    int indice=0;

    GaussianBlur(u_disparity8, u_blur_obstacles, Size(0,0), 2, 2);
    imshow("blurred_u-disparity", u_blur_obstacles);

    HoughLinesP( u_detect_obstacles, lines_obstacles, 5, CV_PI/16, 7, 2, 20 );	//Previous gap:20; previous min 5
    for( size_t i = 0; (i < lines_obstacles.size() && indice<30); i++ ){
    	if((lines_obstacles[i][1] == lines_obstacles[i][3]) && (lines_obstacles[i]!=(Vec4i)Scalar(0,0,0,0))){

    		//Calculo de puntos de lineas de objeto y edificios
    		int y_bl, y_o, x_ol, y_br, x_or;	//b->Building	o->Obstacle
    		y_o=lines_obstacles[i][1];

    		x_ol=std::min(lines_obstacles[i][0], lines_obstacles[i][2]);	//topleft of the object
    		x_or=std::max(lines_obstacles[i][0], lines_obstacles[i][2]);	//topright of the object

    		y_bl=(int)(m_l*x_ol+b_l);
    		y_br=(int)(m_r*x_or+b_r);

    		//ComprobaciÃ³n de que estÃ¡n por debajo de las lineas de la carretera y no detecta por error una linea en y=0 (Aunque deje alguna linea del fondo sin detectar no importa (DESHABILITADO)
    		if(/*((y_o>y_bl)||(!m_l && !b_l)) && ((y_o>y_br)||(!m_r && !b_r)) && */(y_o>DISPARITY_THRESHOLD)){	//Ojo con las restricciones, darse cuenta de que el eje y de la imagen mira hacia abajo
				//obstacles[indice]=lines_obstacles[i];
    			for (int a=0; a<4; a++){
    				obstacles[indice][a] = lines_obstacles[i][a];//Array de obstÃ¡culos
    			}

				//Pongo a cero lineas cercanas en un vecindario
				for(int j=0; j < lines_obstacles.size();j++){
					//Busco todas las lÃ­neas horizontales, diferentes de cero y a una distancia menor de vecindario(defines) y las marco como cero
					if((lines_obstacles[j][1] == lines_obstacles[j][3]) && (lines_obstacles[j]!=(Vec4i)Scalar(0,0,0,0)) && (lines_obstacles[j]!=lines_obstacles[i])){
						if((abs(lines_obstacles[j][1]-lines_obstacles[i][1])<NEIGHBOURHOOD_OBSTACLE_Y)&& (abs(lines_obstacles[j][0]-lines_obstacles[i][0])<NEIGHBOURHOOD_OBSTACLE_X) && (abs(lines_obstacles[j][2]-lines_obstacles[i][2])<NEIGHBOURHOOD_OBSTACLE_X)){
							//Si son mÃ¡s grandes las substituyo por la actual
							if(abs(lines_obstacles[j][0]-lines_obstacles[j][2])>abs(lines_obstacles[i][0]-lines_obstacles[i][2])){
								obstacles[indice]=lines_obstacles[j];
							}
							else{	//Si no las elimino, para asegurarme de no volverlas a coger
								lines_obstacles[j]=Scalar(0,0,0,0);
							}
						}
					}
				}
				line( u_obstaculos, Point(obstacles[indice][0], obstacles[indice][1]),
					Point(obstacles[indice][2], obstacles[indice][3]), Scalar(255,255,255), 1, 8 );
				u_disparity8.copyTo(u_disparity_print);
				line( u_disparity8, Point(obstacles[indice][0], obstacles[indice][1]),
					Point(obstacles[indice][2], obstacles[indice][3]), Scalar(255,255,255), 1, 8 );
				indice++;
    		}
    	}
    }
    int n_obstacles = indice;	//Contador de nÃºmero de obstÃ¡culos

    //VisualizaciÃ³n
    resize(u_obstaculos, u_obstaculos_print, Size(), 1, 0.5);

    cout<<"n_obstacles: "<<n_obstacles<<endl;

    /********************************************************************************************
     * Sacar altura del objeto y guardar todo en ROI
     ********************************************************************************************/

    int y_base, y_top, x_left, x_right;
    for(int i=0; i<n_obstacles; i++){

    	int disparidad_objeto, h, suma_columna;
    	int j;
    	//Cogeremos los mÃ¡ximos valores para disparidad y h
    	//h es la altura en px
    	disparidad_objeto = obstacles[i][1];	//La disparidad es directamente el eje y de la u-disparity
    	h = 0;	//Lo inicializo a un valor cualquiera
    	//Cojo el mÃ¡ximo valor de disparidad (lÃ­mite inferior mas bajo)
    	for(j=std::min(obstacles[i][0], obstacles[i][2]);j<=std::max(obstacles[i][0], obstacles[i][2]);j++){
    		suma_columna=0;
    		//Sumo las maxima altura en un vecindario vertical (Probablemente pertenecientes al objeto
    		for(int row=-MAX_U_DISPARITY_NEIGHBOURHOOD;row<=MAX_U_DISPARITY_NEIGHBOURHOOD;row++){
    			if(((obstacles[i][1]+row)>0) && ((obstacles[i][1]+row)<u_disparity.rows)){	//Compruebo que el punto estÃ© en la imÃ¡gen
    				suma_columna += (int)non_free_u_disparity.at<ushort>((obstacles[i][1]+row), j);
    			}
    		}
    		if(suma_columna>h){
    			h=suma_columna+HEIGHT_OVERSIZE;
    		}
    	}

/*************************************************************************
 * CÃ¡lculo de la caja representando ROI
 *************************************************************************/
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
    	//Creo el rectÃ¡ngulo que define la ROI
    	ROI_disparity[i]=Rect(x_left, y_top, (x_right-x_left), h);
    	cout<<"ROIs"<<ROI_disparity[i]<<endl;
    }
    return n_obstacles;



}

void DrawPedestrians(vector<Rect> &pedestrian_vector, int n_ROI){
	/*Dibuja el rectÃ¡ngulo alrededor del supuesto peatÃ³n en la ventana de vÃ­deo
	 * Le pasamos como parÃ¡metro el frame y el vector de rectÃ¡ngulos de peatones
	 */
	size_t a;
	for( a = 0; a < n_ROI; a++ ){
		//Point Bottom_Right( (pedestrian[a].x + pedestrian[a].width), (pedestrian[a].y + pedestrian[a].height) );
		rectangle(left_color_frame, pedestrian_vector[a], Scalar(0,255,0), 1,8,0);
		//printf("Coordenada x: %d \t Coordenada y: %d \n",pedestrian[a].x,pedestrian[a].y);
		rectangle(disp_print, pedestrian_vector[a], Scalar(255,255,255), 1,8,0);
		//printf("Coordenada x: %d \t Coordenada y: %d \n",pedestrian[a].x,pedestrian[a].y);
	}
}

int main(int argc, char** argv){
	//Window playing the sequence
	vector<Rect> ROI_disparity;
	ROI_disparity.reserve(40);
    vector<Vec4i> buildings;	//Vector con mÃ¡ximo dos elementos (edificios)
    buildings.reserve(2);
	int n_ROI;

	//Codigo para procesar secuencia de imÃ¡genes

		int i, num_pedestrian_VJ=0, num_pedestrian_VJ_HOG=0, num_pedestrian_HOG=0;
		string path="Video/";
		char image_number[12];
		string image_name_left, image_name_right, output_image_name;
		string image_dimensions;

		//Start processing frames
		for(i=0;i<=N_FRAMES;i++){

			sprintf(image_number,"%010d",i);
			cout<<path<<image_number<<".png"<<endl;
			image_name_left=path+"Secuencias/left4/"+(string)image_number+".png";
			image_name_right=path+"Secuencias/right4/"+(string)image_number+".png";
			left_frame = imread(image_name_left.c_str(), IMREAD_GRAYSCALE);
			left_color_frame = imread(image_name_left.c_str(), CV_LOAD_IMAGE_COLOR);
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


		    n_ROI = Stereo_SGBM(ROI_disparity, buildings);
		    DrawPedestrians(ROI_disparity, n_ROI);


		    /******************************************************************************
		     * Guardar imÃ¡genes para debugging
		     ******************************************************************************/
		    output_image_name=path+"StereoROI_16bits/"+(string)image_number+".jpg";
		    imwrite(output_image_name, left_color_frame);

		    output_image_name=path+"StereoROI_16bits/Debug/disparity/"+(string)image_number+".jpg";
		    imwrite(output_image_name, disp_print);

		    output_image_name=path+"StereoROI_16bits/Debug/u_disparity/"+(string)image_number+".jpg";
		    imwrite(output_image_name, u_disparity_print);

		    output_image_name=path+"StereoROI_16bits/Debug/u_detect_obstacles/"+(string)image_number+".jpg";
		    imwrite(output_image_name, u_detect_obstacles);

		    output_image_name=path+"StereoROI_16bits/Debug/v_disparity/"+(string)image_number+".jpg";
		    imwrite(output_image_name, v_disparity8);

		    output_image_name=path+"StereoROI_16bits/Debug/obstacle_mask/"+(string)image_number+".jpg";
		    imwrite(output_image_name, obstacle_mask8);

			imshow( "Debug window", left_color_frame );
			int c = waitKey(10);
			if( (char)c == 27 )
				return 0;
			else if((char)c == 's'){	//s stops playing the video
				int c = waitKey(0);
			}
		}

}






			imshow( "Debug window", left_color_frame );
			int c = waitKey(10);
			if( (char)c == 27 )
				return 0;
			else if((char)c == 's'){	//s stops playing the video
				int c = waitKey(0);
			}
		}

}

