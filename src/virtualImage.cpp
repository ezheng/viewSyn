#include "virtualImage.h"
#include <iostream>
virtualImage::virtualImage(const image& im)
{	
	_glmR = im._glmR;
	_glmC = im._glmC;
	_glmK = im._glmK;
	_optCenterPos = im._optCenterPos;
	_lookAtPos = im._lookAtPos;
	_upDir = im._upDir;

	_modelViewMatrix = im._modelViewMatrix;
	_projMatrix = im._projMatrix;	

	_image = im._image;
}


void virtualImage::setModelViewMatrix()
{
	glm::vec3 viewDir = glm::vec3(0.0f,0.0f,1.0f);
	viewDir = glm::transpose(_glmR) * viewDir ;	
	
	//_optCenterPos = -1* glm::transpose(_glmR) * _glmT;	
	_optCenterPos = _glmC;
	_lookAtPos = _optCenterPos + viewDir;

	_upDir = glm::vec3(0.0f,-1.0f,0.0f);
	_upDir = glm::normalize(glm::transpose(_glmR) * _upDir);
	
	_modelViewMatrix = glm::lookAt(_optCenterPos,_lookAtPos, _upDir);
	
}

void virtualImage::setProjMatrix(float near1,  float far1)
{
	//float near1 = 0.1f;  float far1 = 200.0;

	float bottom = - ((float) _image.rows  - _glmK[2][1])/_glmK[1][1]  * near1 ;	// focal length is in matrix K
	float top    =  _glmK[2][1]/_glmK[1][1]  * near1 ;
	float left   = -_glmK[2][0]/_glmK[0][0]  * near1 ;
	float right	 =  ((float)_image.cols - _glmK[2][0])/_glmK[0][0]  * near1;
	//float bottom = -( ((float) _image.rows  - _glmK[1][2])/_glmK[1][1] ) * near1 ;	// focal length is in matrix K
	//float top    = ( _glmK[1][2]/_glmK[1][1] ) * near1 ;
	//float left   = -( _glmK[0][2]/_glmK[0][0] ) * near1 ;
	//float right	 = ( ((float)_image.cols - _glmK[0][2])/_glmK[0][0] ) * near1;

	_projMatrix = glm::frustum(left,right,bottom,top,near1,far1);
}

void virtualImage::calcPlaneCoords()
{
	glm::mat4x4 inverseModelViewMatrix = glm::inverse(_modelViewMatrix);
	glm::mat4x4 inverseProjectionMatrix = glm::inverse(_projMatrix);
	glm::mat4x4 inverseMVP = inverseModelViewMatrix * inverseProjectionMatrix;
		
	nearPlane.leftBottom = dividew(inverseMVP * glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f)); 
	nearPlane.rightBottom = dividew( inverseMVP * glm::vec4(1.0f, -1.0f, -1.0f, 1.0f));
	nearPlane.leftTop = dividew(inverseMVP * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f));
	nearPlane.rightTop = dividew(inverseMVP * glm::vec4(1.0f, 1.0f, -1.0f, 1.0f));
			
	farPlane.leftBottom = dividew(inverseMVP * glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f));
	farPlane.rightBottom = dividew(inverseMVP * glm::vec4(1.0f, -1.0f, 1.0f, 1.0f));
	farPlane.leftTop = dividew(inverseMVP * glm::vec4(-1.0f, 1.0f, 1.0f, 1.0f));
	farPlane.rightTop = dividew(inverseMVP * glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));

}

glm::vec3 virtualImage::dividew(glm::vec4 input)
{
	return glm::vec3(input.x/input.w, input.y/input.w, input.z/input.w);	
}

