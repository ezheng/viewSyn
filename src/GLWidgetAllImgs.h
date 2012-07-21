#ifndef _GLWIDGETALLIMGS_H
#define _GLWIDGETALLIMGS_H

#include <QtOpenGL/QGLWidget>
#include "image.h"
#include <vector>
#include <glm/glm.hpp>
#include <QWheelEvent>
#include "GLWidget.h"
#include "virtualImage.h"

class GLWidgetAllImgs : public QGLWidget {
	Q_OBJECT
public:
	//GLWidgetAllImgs(std::vector<image> *allIms, int id, QWidget *parent = NULL);
	GLWidgetAllImgs(std::vector<image> **allIms, QGLWidget *sharedWidget, const QList<GLWidget*>& imageQGLWidgets, glm::vec3 xyzMin, glm::vec3 xyzMax);
	~GLWidgetAllImgs() {
		if( _virtualImg != NULL)
		{delete _virtualImg;}
	}

	std::vector<image>**_allIms;

// pose of virtual view is stored in _virtualImg
	virtualImage *_virtualImg;

// this is the pos of virtual view? No. 
	glm::vec3 _allCamCenter;
	glm::vec3 _optCenterPos;
	glm::vec3 _objCenterPos;
	float _allCamRadius;
	float _aspectRatio;
	float _nearPlane;
	float _farPlane;
	float _fieldOfView;
	glm::vec3 _viewingDir;

	glm::vec3 _virtualPointCenterPos;
	glm::mat4x4 _virtualProjectionMatrix;
	glm::mat4x4 _virtualModelViewMatrix;

	void upDateParam();
	
protected:
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();	
	void keyPressEvent(QKeyEvent* event);
	void mousePressEvent(QMouseEvent *event);
	void wheelEvent(QWheelEvent *event);
	void mouseMoveEvent(QMouseEvent *event);

private:
	int printOglError(char *file, int line);	
	//void drawOneCam(const image &img);
	void drawOneCam(const image &img, GLWidget* _oneImageQGLWidgets);
	void drawOneCam(const virtualImage &img); 

	void drawObjectScope();
	void computeBoundingBox();
	void display();
	void drawCoordinate();	

	float _camMinDistance;
	float _windowWidth;
	float _windowHeight;

	int _mouseX;
	int _mouseY;
	bool _keyControlPressed;
	QList<GLWidget*> _imageQGLWidgets;

	glm::vec3 _xyzMin;
	glm::vec3 _xyzMax;

public slots:
	void updateVirtualView_slot(virtualImage virImg);


};

#endif