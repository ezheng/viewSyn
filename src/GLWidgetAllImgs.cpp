
#include "GLWidgetAllImgs.h"
#include <gl/GLU.h>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <QKeyEvent>
#ifndef _WIN64
	#include "imdebug.h"
#endif
#include <qtimer.h>

int GLWidgetAllImgs:: printOglError(char *file, int line)
{
    GLenum glErr;
    int    retCode = 0;
    glErr = glGetError();
    if (glErr != GL_NO_ERROR)
    {
        printf("glError in file %s @ line %d: %s\n",
			     file, line, gluErrorString(glErr));
        retCode = 1;
    }
    return retCode;
}

GLWidgetAllImgs::GLWidgetAllImgs(std::vector<image> **allIms, QGLWidget *sharedWidget, const QList<GLWidget*> &imageQGLWidgets, glm::vec3 xyzMin, glm::vec3 xyzMax )
	: QGLWidget((QWidget*)NULL, sharedWidget), _allIms(allIms), _imageQGLWidgets(imageQGLWidgets), _virtualImg(NULL), _xyzMin(xyzMin), _xyzMax(xyzMax)
{
	upDateParam();

	//_timer = new QTimer(this);

	// make a thread for kinect
	_faceTrack.moveToThread(&_KinectThread);	
	QObject::connect( &_KinectThread, SIGNAL(started()), &_faceTrack, SLOT(FaceTrackingThread()));
	QObject::connect( &_faceTrack, SIGNAL(drawKinect_SIGNALS( unsigned char *, int, int, int , int , int , int )), 
		this, SLOT(drawKinect_SLOTS( unsigned char *, int, int, int , int , int , int )), Qt::QueuedConnection);
	//QObject::connect( this, SIGNAL(doFaceTracking_SIGNAL()), &_faceTrack, SLOT(doFaceTracking_SLOT()), Qt::QueuedConnection);
	//QObject::connect(_timer, SIGNAL(timeout()), &_faceTrack, SLOT(doFaceTracking_SLOT()), Qt::QueuedConnection);
	
	std::cout<<"current threadId GLWidgetAllImgs" << GetCurrentThreadId() << std::endl;
}

void GLWidgetAllImgs :: upDateParam()
{
	computeBoundingBox();

	_nearPlane = _allCamRadius * 0.1;
	_farPlane = _allCamRadius * 100;	// this will work for all the cases
	_fieldOfView = 120;	// this can be changed by mouse

	_objCenterPos = _allCamCenter;
	//_optCenterPos = _allCamCenter + glm::vec3(0., 0., -1.5* _allCamRadius);
	_optCenterPos = _allCamCenter - ((_viewingDir * _allCamRadius) * 5.0f);
	std::cout<<" allCamRadius: " << _allCamRadius << std::endl;
	_virtualModelViewMatrix = glm::lookAt( _optCenterPos, 
		_objCenterPos, glm::vec3(0.0, 1.0, 0.0));
	_virtualProjectionMatrix = glm::perspective( _fieldOfView, _aspectRatio, _nearPlane, _farPlane);
}


void GLWidgetAllImgs :: initializeGL(){	

	glewInit();

	
	glColor4f(1., 0., 0., 1.);	

	_KinectThread.start();

	

}


void GLWidgetAllImgs::mousePressEvent(QMouseEvent *event)
{
	_mouseX = event->x();
	_mouseY = event->y();
}

void GLWidgetAllImgs::mouseMoveEvent(QMouseEvent *event)
{
	int deltaX =  event->x() - _mouseX;
	int deltaY =  event->y() - _mouseY;
	//std::cout<< "mouseX: " << _mouseX << " mouseY: " << _mouseY << std::endl;
	//std::cout<< "deltaX: " << deltaX << " deltaY: " << deltaY << std::endl;

	float s = static_cast<float>( std::max(this->width(), this->height()));
	float rangeX = static_cast<float>(deltaX) / (s + 0.00001);
	float rangeY = static_cast<float>(deltaY) / (s + 0.00001);
	//std::cout<< "rangeX: " << rangeX << " rangeY: " << rangeY << std::endl;
	//std::cout<< "deltaX: " << deltaX << " deltaY: " << deltaY << std::endl; 
	if (event->buttons() & Qt::LeftButton) {
		glm::mat4x4 inverseVirtualModelViewMatrix = glm::inverse(_virtualModelViewMatrix);
		glm::vec4 dir =   inverseVirtualModelViewMatrix *  glm::vec4(deltaY, deltaX, 0.0f, 0.0f);
		glm::normalize(dir);
		float mag = sqrt(pow(rangeX,2) + pow(rangeY,2)) * 180;
		//glm::vec3 aaa = glm::vec3(0., 0., -1.5* _allCamRadius);
		_virtualModelViewMatrix =  _virtualModelViewMatrix * glm::translate(_objCenterPos) * glm::rotate(mag, dir.x, dir.y, dir.z) * glm::translate(-_objCenterPos);	
	}
	else if (event->buttons() & Qt::RightButton){
		// do the translation		
		glm::mat4x4 inverseProjectionMatrix = glm::inverse(_virtualProjectionMatrix);
		// TPM = P((P^-1)TP)M
		_virtualModelViewMatrix = inverseProjectionMatrix * glm::translate(rangeX, -rangeY, 0.0f) * _virtualProjectionMatrix *  _virtualModelViewMatrix;
		//_virtualProjectionMatrix = glm::translate(rangeX, -rangeY, 0.0f) * _virtualProjectionMatrix;
	}
	
	updateGL();
	_mouseX = event->x();
	_mouseY = event->y();
}

void GLWidgetAllImgs::wheelEvent(QWheelEvent * event)
{
	bool ctrlPressed = event->modifiers() == Qt::ControlModifier;
	int numDegrees = event->delta();
		
	if(numDegrees > 0 && ctrlPressed )
		_camMinDistance *= 1.01f;
	else if(numDegrees < 0 && ctrlPressed && _camMinDistance > 0.02 )
		_camMinDistance /= 1.01f;
	else if(numDegrees > 0 && !ctrlPressed && _fieldOfView > 0.02)
	{
		_fieldOfView /= 1.01f; // zoom in
		_virtualProjectionMatrix = glm::perspective( _fieldOfView, _aspectRatio, _nearPlane, _farPlane);	
	}
	else
	{
		_fieldOfView *= 1.01f; // zoom out
		_virtualProjectionMatrix = glm::perspective( _fieldOfView, _aspectRatio, _nearPlane, _farPlane);
	}
	//makeCurrent();
	//display();
	//swapBuffers();
	updateGL();
}

void GLWidgetAllImgs::keyPressEvent(QKeyEvent* event) {

	switch(event->key()) {
	case Qt::Key_Escape:
		close();
		break;
	
	default:
		event->ignore();
		break;
	}
}

void GLWidgetAllImgs::updateVirtualView_slot(virtualImage virImg)
{
	if(_virtualImg == NULL)
		_virtualImg = new virtualImage();

	*_virtualImg = virImg;
	updateGL();
}


void GLWidgetAllImgs :: resizeGL(int w, int h){
	_windowWidth = w;
	_windowHeight = h;	

	
	_aspectRatio = static_cast<float>(w/2)/ static_cast<float>(h);
	

	_virtualProjectionMatrix = glm::perspective( _fieldOfView, _aspectRatio, _nearPlane, _farPlane);
}

void GLWidgetAllImgs::display()
{
	glClearColor(0.5, 0.5, 0.5, 1.);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	glLoadMatrixf(&(_virtualProjectionMatrix[0][0]));

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glLoadMatrixf(&(_virtualModelViewMatrix[0][0]));
	glViewport(0, 0, _windowWidth/2, _windowHeight);
	for(size_t i = 0; i< (*_allIms)->size(); i++)
	{		
		drawOneCam( (**_allIms)[i], _imageQGLWidgets[i]); 
	}
	drawCoordinate();
//	drawObjectScope();

	if(_virtualImg != NULL)
		drawOneCam( *_virtualImg);

	printOglError(__FILE__, __LINE__);
}

// make a slot to receive virtual image info, and then reDisplay

void GLWidgetAllImgs :: drawObjectScope()
{
	float xmin = _xyzMin.x; 	float ymin =  _xyzMin.y; 	float zmin = _xyzMin.z;
	float xmax = _xyzMax.y;		float ymax = _xyzMax.y;		float zmax = _xyzMax.z;


	//float xmin = -0.054568f; 	float ymin =  0.001728f; 	float zmin = -0.042945f;
	//float xmax = 0.047855f;		float ymax = 0.161892f;		float zmax = 0.032236f;

	glBegin(GL_LINE_STRIP);
		glVertex3f( xmin, ymin, zmin);
		glVertex3f( xmax, ymin, zmin);
		glVertex3f( xmax, ymax, zmin);
		glVertex3f( xmin, ymax, zmin);
		glVertex3f( xmin, ymin, zmin);
	//--------------	
		glVertex3f( xmin, ymin, zmax);
		glVertex3f( xmax, ymin, zmax);
		glVertex3f( xmax, ymax, zmax);
		glVertex3f( xmin, ymax, zmax);
		glVertex3f( xmin, ymin, zmax);
	glEnd();
	glBegin(GL_LINES);
		glVertex3f( xmax, ymin, zmin);
		glVertex3f( xmax, ymin, zmax);
	//--------------
		glVertex3f( xmax, ymax, zmin);
		glVertex3f( xmax, ymax, zmax);
	//--------------	
		glVertex3f( xmin, ymax, zmin);
		glVertex3f( xmin, ymax, zmax);
	glEnd();
}

void GLWidgetAllImgs :: paintGL(){

	display();
	displayImage(_kinectColorImage._textureID);
}


void GLWidgetAllImgs :: drawKinect_SLOTS( unsigned char *data, int width, int height, int left, int right, int bottom, int top)
{
	/*POINT leftTop = {rectFace.left, rectFace.top};
	POINT rightTop = {rectFace.right - 1, rectFace.top};
	POINT leftBottom = {rectFace.left, rectFace.bottom - 1};
	POINT rightBottom = {rectFace.right - 1, rectFace.bottom - 1};
	std::cout<< "leftTop: " << leftTop.x << " " << leftTop.y << std::endl;
	std::cout<< "rightTop: " << rightTop.x << " " << rightTop.y << std::endl;
	std::cout<< "leftBottom: " << leftBottom.x << " " << leftBottom.y << std::endl;
	std::cout<< "rightBottom: " << rightBottom.x << " " << rightBottom.y << std::endl;*/
	//std::cout << left << " " << right << " " << bottom << " " << top << std::endl;

	// display the image
	//imdebug("rgba w=%d h=%d %p", width, height, data);
	


	static bool isFirstTime = true;
	static bool isLastFrameDetected = false;
	float centerX = static_cast<float>( left + right)/2.0f;
	float centerY = static_cast<float>( top + bottom)/2.0f;

	makeCurrent();
	if(isFirstTime)
	{
		_kinectImageAspectRatio = float(height)/float(width);
		_kinectColorImage.setTextureSize(width, height);
		_kinectColorImage.createRGBA(data);
		printOglError(__FILE__, __LINE__);
		isFirstTime = false;
		_kinectPos_X = centerX;
		_kinectPos_Y = centerY;
		//emit newPosKinect_SIGNAL(centerX, centerY, true);
	}
	else
	{
		_kinectColorImage.upLoadRGBA(data, width, height);

		if(!isLastFrameDetected)
		{
			if(left != -1 || right != -1 || top != -1 || bottom != -1)
			{
				emit newPosKinect_SIGNAL(centerX, centerY, true);
				isLastFrameDetected = true;
			}
		}
		else
		{
			if(left != -1 || right != -1 || top != -1 || bottom != -1)
			{
				int deltaX = centerX - _kinectPos_X;
				int deltaY = centerY - _kinectPos_Y;
				//if( deltaX * deltaX + deltaY * deltaY > 4.0f)
				if( abs(deltaX)  > 2.0f)
				{
					// emit signal
					if(isLastFrameDetected)
						emit newPosKinect_SIGNAL(deltaX, deltaY, false);
					_kinectPos_X = centerX;
					_kinectPos_Y = centerY;
				}
				isLastFrameDetected = true;
			}
			else
				isLastFrameDetected = false;
		}
		
	}
	//// set the viewport and display image
	//makeCurrent();
	
	//displayImage(_kinectColorImage._textureID);
	//this->swapBuffers();

	
	updateGL();
	// emit signal for updating view synthesis first
	
	// emit signal to get new image
	QTimer::singleShot(0, &_faceTrack, SLOT(doFaceTracking_SLOT()));

}

void GLWidgetAllImgs:: displayImage(GLuint texture)
{
	//GLint prevProgram;
	if(texture == 0)
		return;
	GLint prevProgram;
	glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
	glUseProgram(0);
	//glClearColor(0.0f,0.0f,0.0f,1.0f);
	//glClear(GL_COLOR_BUFFER_BIT);  
	printOglError(__FILE__, __LINE__);
	glActiveTexture(GL_TEXTURE0);
	printOglError(__FILE__, __LINE__);
	glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);    
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	//printOpenGLError();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
    glLoadIdentity();

	//glViewport(0, 0, _windowWidth, _windowHeight);
	glViewport(_windowWidth/2, 0, _windowWidth/2, _windowWidth/2 * _kinectImageAspectRatio);

	//printOpenGLError();	
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); 	glVertex3f(-1.0, 1.0, 0.5);
    glTexCoord2f(1.0, 0.0); 	glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(1.0, 1.0); 	glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(0.0, 1.0); 	glVertex3f(-1.0, -1.0, 0.5);
    glEnd();
	//printOpenGLError();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
//    glDisable(GL_TEXTURE_2D);	
	glUseProgram(GLuint(prevProgram));
	//printOpenGLError();	
}


void GLWidgetAllImgs::drawCoordinate()
{
	glBegin(GL_LINES);
	// x axis
		glColor3f(1., 0., 0.);
		glVertex3f(0.,0.,0.); glVertex3f(_camMinDistance,0.,0.);
	// y axis	
		glColor3f(0., 1., 0.);
		glVertex3f(0.,0.,0.); glVertex3f(0., _camMinDistance,0.);
	// z axis	
		glColor3f(0., 0., 1.);
		glVertex3f(0.,0.,0.); glVertex3f(0.,0.,_camMinDistance);
	glEnd();
}

void GLWidgetAllImgs::drawOneCam(const virtualImage &img)
{
	glm::vec3 optCenterPos = img._optCenterPos;
	glm::vec3 lookAtPos = img._lookAtPos;
	glm::vec3 upDir = img._upDir;

	glm::vec3 camLookAtDir = glm::normalize(lookAtPos - optCenterPos);
	glm::vec3 camUpDir = glm::normalize(upDir);
	glm::vec3 camLeftDir = glm::cross(camUpDir, camLookAtDir);

	float scaleFocalLength = img._glmK[0][0] / static_cast<float>(img._image.cols);
	float scaleImageHeight = img._image.rows/ static_cast<float>(img._image.cols);

	float cameraSizeScale = _camMinDistance;
	//	p1 ---- p2
	//	p4 ---- p3
	glm::vec3 p1 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (camUpDir * scaleImageHeight + camLeftDir) * cameraSizeScale/2.0f;
	glm::vec3 p2 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (camUpDir * scaleImageHeight - camLeftDir) * cameraSizeScale/2.0f;
	glm::vec3 p3 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (-camUpDir * scaleImageHeight - camLeftDir) * cameraSizeScale/2.0f;
	glm::vec3 p4 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (-camUpDir * scaleImageHeight + camLeftDir) * cameraSizeScale/2.0f;


	float currentColor[4];
	glGetFloatv(GL_CURRENT_COLOR,currentColor);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glBegin(GL_LINE_LOOP);	
		glVertex3f(p1.x, p1.y,p1.z); 
		glVertex3f(p2.x, p2.y,p2.z);
		glVertex3f(p3.x, p3.y,p3.z);
		glVertex3f(p4.x, p4.y,p4.z);		
	glEnd();
	glBegin(GL_LINES);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p1.x, p1.y, p1.z);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p2.x, p2.y, p2.z);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p3.x, p3.y, p3.z);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p4.x, p4.y, p4.z);
	glEnd();

	// draw near planes and far planes
	glBegin(GL_LINE_LOOP);
		glVertex3fv( &(img.nearPlane.leftBottom[0]));
		glVertex3fv( &(img.nearPlane.leftTop[0]));
		glVertex3fv( &(img.nearPlane.rightTop[0]));
		glVertex3fv( &(img.nearPlane.rightBottom[0]));
	glEnd();
	glBegin(GL_LINE_LOOP);
		glVertex3fv( &(img.farPlane.leftBottom[0]));
		glVertex3fv( &(img.farPlane.leftTop[0]));
		glVertex3fv( &(img.farPlane.rightTop[0]));
		glVertex3fv( &(img.farPlane.rightBottom[0]));
	glEnd();
	glBegin(GL_LINES);
		glVertex3fv( &(img.nearPlane.leftBottom[0])); glVertex3fv( &(img.farPlane.leftBottom[0]));
		glVertex3fv( &(img.nearPlane.leftTop[0])); glVertex3fv( &(img.farPlane.leftTop[0]));
		glVertex3fv( &(img.nearPlane.rightTop[0])); glVertex3fv( &(img.farPlane.rightTop[0]));
		glVertex3fv( &(img.nearPlane.rightBottom[0])); glVertex3fv( &(img.farPlane.rightBottom[0]));
	glEnd();

	glColor4f(currentColor[0], currentColor[1], currentColor[2], currentColor[3]);


}

void GLWidgetAllImgs::drawOneCam(const image &img, GLWidget* _oneImageQGLWidgets)
{

	glm::vec3 optCenterPos = img._optCenterPos;
	glm::vec3 lookAtPos = img._lookAtPos;
	glm::vec3 upDir = img._upDir;

	/*std::cout<< "optCenterPos" << std::endl;
	for(int i = 0; i<3; i++)
		std::cout<< optCenterPos[i]<< " ";
	std::cout<<std::endl;
	std::cout<< "lookAtPos: " << std::endl;
	for(int i = 0; i<3; i++)
		std::cout<< lookAtPos[i]<< " ";
	std::cout<<std::endl;*/


	glm::vec3 camLookAtDir = glm::normalize(lookAtPos - optCenterPos);
	glm::vec3 camUpDir = glm::normalize(upDir);
	glm::vec3 camLeftDir = glm::cross(camUpDir, camLookAtDir);

	float scaleFocalLength = img._glmK[0][0] / static_cast<float>(img._image.cols);
	float scaleImageHeight = img._image.rows/ static_cast<float>(img._image.cols);

	float cameraSizeScale = _camMinDistance;
	//	p1 ---- p2
	//	p4 ---- p3
	glm::vec3 p1 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (camUpDir * scaleImageHeight + camLeftDir) * cameraSizeScale/2.0f;
	glm::vec3 p2 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (camUpDir * scaleImageHeight - camLeftDir) * cameraSizeScale/2.0f;
	glm::vec3 p3 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (-camUpDir * scaleImageHeight - camLeftDir) * cameraSizeScale/2.0f;
	glm::vec3 p4 = optCenterPos + camLookAtDir * (cameraSizeScale ) + (-camUpDir * scaleImageHeight + camLeftDir) * cameraSizeScale/2.0f;

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, _oneImageQGLWidgets->_tex._textureID);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glBegin(GL_QUADS);	
		glTexCoord2f(0.0f,1.0f);
		glVertex3f(p1.x, p1.y,p1.z); 
		glTexCoord2f(1.0f,1.0f);
		glVertex3f(p2.x, p2.y,p2.z);
		glTexCoord2f(1.0f,0.0f);
		glVertex3f(p3.x, p3.y,p3.z);
		glTexCoord2f(0.0f,0.0f);
		glVertex3f(p4.x, p4.y,p4.z);		
	glEnd();
	glDisable(GL_TEXTURE_2D);

	glBegin(GL_LINES);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p1.x, p1.y, p1.z);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p2.x, p2.y, p2.z);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p3.x, p3.y, p3.z);
		glVertex3f(optCenterPos.x, optCenterPos.y,optCenterPos.z);
		glVertex3f(p4.x, p4.y, p4.z);
	glEnd();

	//draw upDir:
	/*glColor4f(1.0f, 1.0f, 1.0f, 0.0f);
	glm::vec3 p5 = p4 + camUpDir * cameraSizeScale;
	glBegin(GL_LINES);
		glVertex3f(p4.x, p4.y, p4.z);
		glVertex3f(p5.x, p5.y, p5.z);
	glEnd();
	glColor4f(0.0f, 0.0f, 1.0f, 0.0f);*/
}


void GLWidgetAllImgs:: computeBoundingBox()
{
	// find the bounding box of all cameras
	int numOfCams = (*_allIms)->size(); 
	if(numOfCams == 0) {std::cout<< "WARNING: there is no cameras" <<std::endl; exit(0);}
	
	_allCamCenter = glm::vec3(0.,0.,0.); 
	for( int i = 0; i < numOfCams; i++)
	{
		_allCamCenter += (**_allIms)[i]._glmC;
	}
	_allCamCenter = _allCamCenter / static_cast<float>(numOfCams);
	_allCamRadius = -1.0f;
	//_allCamMaxDistance = MAX_FLOAT;
	for( int i = 0; i< numOfCams; i++)
	{
		float radius = glm::distance( (**_allIms)[i]._glmC, _allCamCenter);
		if( radius > _allCamRadius)
			_allCamRadius = radius;
	}
	if(numOfCams == 1)
		_allCamRadius = 1.0f;
	//------------------------------------------------------------------------
	if(numOfCams > 1)
	{
		_camMinDistance = glm::distance((**_allIms)[0]._glmC, (**_allIms)[1]._glmC );
		for( int i = 0; i< numOfCams; i++)
			for( int j = i + 1; j<numOfCams;j++)
			{
				if( _camMinDistance > glm::distance((**_allIms)[0]._glmC, (**_allIms)[1]._glmC ))
					_camMinDistance = glm::distance((**_allIms)[0]._glmC, (**_allIms)[1]._glmC );
			}
	}
	else
	{
		_camMinDistance = 1.0f;	
	}

	_viewingDir = (** _allIms)[0]._viewDir;
	_viewingDir = glm::normalize(_viewingDir);

}