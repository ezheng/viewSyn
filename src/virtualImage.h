#ifndef VIRTUAL_IAMGE_H
#define VIRTUAL_IAMGE_H

#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "image.h"

struct plane
{

	glm::vec3 leftBottom;		// these are under world coordinates, not viewpoint coordinate
	glm::vec3 rightBottom;
	glm::vec3 leftTop;
	glm::vec3 rightTop;
		
};

class virtualImage
{
public:
	glm::mat3x3 _glmR;
	glm::vec3 _glmC;
	glm::mat3x3 _glmK;

	glm::vec3 _optCenterPos;
	glm::vec3 _lookAtPos;
	glm::vec3 _upDir;

	glm::mat4x4 _modelViewMatrix;
	glm::mat4x4 _projMatrix;
	
	plane nearPlane;
	plane farPlane;

//-------------------
	std::string _imageName;
	cv::Mat _image;
	void setModelViewMatrix();
	void setProjMatrix();	
	void calcPlaneCoords();
	
	virtualImage( const image& im);
	virtualImage(){}

private:
	glm::vec3 dividew(glm::vec4 input);
};



#endif
