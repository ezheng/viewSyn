#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <opencv/highgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//#include 

class image
{
public:
//	matrix for opencv	
	cv::Mat _R;
	cv::Mat _C;		// C = -R'T
	cv::Mat _T; 	// T = -RC
	cv::Mat _K;
	cv::Mat _proj;	// Proj = KR[I,-C] = k[R,T]
//	matrix for opengl	(There is reduant info in this class)
	glm::mat3x3 _glmR;
	glm::vec3 _glmC;
	glm::vec3 _glmT;
	glm::mat3x3 _glmK;

	glm::vec3 _optCenterPos;
	glm::vec3 _lookAtPos;
	glm::vec3 _upDir;
	glm::vec3 _viewDir;

	glm::mat4x4 _modelViewMatrix;
	glm::mat4x4 _projMatrix;
	
	float _near;
	float _far;

//-------------------
	std::string _imageName;
	cv::Mat _image;
	void setModelViewMatrix();
	void setProjMatrix();
	void setProjMatrix(float Near, float Far);
	
	cv::Mat calculateFundMatrix(const image &im);		
	image(std::string fileName, double * K, double *R, double *T);
	image(){}
	~image(){}
	//image( const image& im);
};



#endif
