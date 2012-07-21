#include <QtGui/QMouseEvent>
#include "GLWidget.h"
#include <gl/GLU.h>
#include <iostream>
#include <cmath>

#define printOpenGLError() printOglError(__FILE__, __LINE__)

int GLWidget:: printOglError(char *file, int line)
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

GLWidget::GLWidget(image *im, int id, const QGLWidget * shareWidget) : QGLWidget( (QWidget *)NULL ,shareWidget), _im(im), _tex(im->_image.cols, im->_image.rows) {
	setMouseTracking(true);
	_x1 = _y1 = _x2 = _y2 = 0.0;
	_id = id;		
}

void GLWidget::initializeGL() {
	// upload texture
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);	
	glEnable(GL_POLYGON_SMOOTH);	
	glClearColor(1.0, 0, 1.0, 0);
	_tex.create(_im->_image.data);	

	printOpenGLError();
}

void GLWidget::resizeGL(int w, int h) {	
	glViewport(0, 0, w, h);	
}


int GLWidget::heightForWidth ( int w ) const
{
	return 1;
}

void GLWidget::paintGL() {
	display();
}

void GLWidget::display()
{

	glClear(GL_COLOR_BUFFER_BIT); 
	glBindTexture(GL_TEXTURE_2D, _tex._textureID);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
 
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    //glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
	glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);

    glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
    glLoadIdentity();
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(0, 0, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, 0, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(0, 1.0, 0.5);
    glEnd();

	glDisable(GL_TEXTURE_2D);
	//_x1 = 0; _y1 = 1.0; _x2 = 1; _y2 = 0.0;
	glColor4f(1.0f,1.0f,1.0f, 1.0f);
	glBegin(GL_LINES);
		 glVertex2f(_x1, _y1); 
		 glVertex2f(_x2, _y2);
	glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();	
}

void GLWidget::mousePressEvent(QMouseEvent *event) {
	//printf("%d, %d\n", event->x(), event->y());
	std::cout<< "paintGL is called, id:"<< _id << std::endl;
	int realX = std::floor( static_cast<double>(event->x()) * static_cast<double>(_im->_image.cols) / static_cast<double>(this->width()) + 0.5);
	int realY = std::floor(static_cast<double>(event->y()) * static_cast<double>(_im->_image.rows) / static_cast<double>(this->height()) + 0.5);
	_x1 = _y1 = _x2 = _y2 = -1.0;
	makeCurrent();
	display();
	displayClickedPoint(event->x(), event->y());
	swapBuffers();

	emit posChanged(realX, realY, *_im);
}

void GLWidget::displayClickedPoint(int x, int y)
{
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
	glOrtho(0, 1.0f, 0, 1.0f, -1.0f, 1.0f);
    glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
    glLoadIdentity();

	//_x1 = 0; _y1 = 1.0; _x2 = 1; _y2 = 0.0;
	double pointX, pointY;
	pointX = static_cast<double>(x) / static_cast<double>(this->width());
	pointY = static_cast<double>(y) / static_cast<double>(this->height());
	glColor4f(1.0f,1.0f,1.0f, 1.0f);
	glPointSize(5.0);
	glBegin(GL_POINTS);
		 glVertex2f(pointX, 1.0 - pointY); 		 
		//glVertex2f(0.5, 0.5); 		 
	glEnd();

	glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();	
}



void GLWidget::setPosValue(int posX, int posY, image refImg)
{
	// find and draw the line
	cv::Mat fundMatrix = _im->calculateFundMatrix(refImg);
	cv::Mat pointsPos = (cv::Mat_ <double>(3, 1) << static_cast<double>(posX), static_cast<double>(posY), 1.0);
	cv::Mat line = fundMatrix * pointsPos;
	drawLine(line);
	makeCurrent();
	display();
	swapBuffers();
}

void GLWidget::drawLine( cv::Mat &line)
{
	double a = line.at<double>(0);
	double b = line.at<double>(1);
	double c = line.at<double>(2);

	double x1,y1,x2,y2;
	if(a ==0 && b==0) {std::cout<< "WARNING: epipolar line is not correctly calculated"; return;}
	if(a == 0)
	{
		x1 = -1.0; x2 = static_cast<double>(_im->_image.cols); y1 = y2 = -c/b;
	}
	else if(b==0)
	{
		x1 = x2 = -c/a; y1 = -1.0; y2 = static_cast<double>(_im->_image.rows);
	}
	else
	{	// choose arbitray two points
		if(abs(b) > abs(a)){
			x1 = -1.0; y1 = (-a * x1 - c)/b;
			x2 = static_cast<double>(_im->_image.cols); y2 = (-a * x2 - c)/b;
		}
		else{
			y1 = -1.0; x1 = (-b * y1 -c)/a;
			y2 = static_cast<double>(_im->_image.rows); x2 = (-b * y2 -c)/a;
		}
	}
	// normalize
	_x1 = x1/_im->_image.cols; _x2 = x2 /_im->_image.cols;
	_y1 = (_im->_image.rows - y1) /_im->_image.rows; 
	_y2 = (_im->_image.rows - y2) /_im->_image.rows;

}


void GLWidget::mouseMoveEvent(QMouseEvent *event) {	

}

void GLWidget::keyPressEvent(QKeyEvent* event) {
	switch(event->key()) {
	case Qt::Key_Escape:
		close();
		break;
	default:
		event->ignore();
		break;
	}
}