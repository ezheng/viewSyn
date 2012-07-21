#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <QtOpenGL/QGLWidget>
#include "image.h"
#include "texture2D.h"

class GLWidget : public QGLWidget {

	Q_OBJECT // must include this if you use Qt signals/slots

public:
	GLWidget(image *im, int id, const QGLWidget * shareWidget);
	image *_im;
	texture2D _tex;
	virtual int heightForWidth ( int w ) const;
	
signals:
	void posChanged(int posX, int posY, image refImg);

public slots:	
    void setPosValue(int posX, int posY, image refImg);

protected:
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();	
	void mouseMoveEvent(QMouseEvent *event);
	void keyPressEvent(QKeyEvent *event);
	void mousePressEvent(QMouseEvent *event);

	
private:
	int printOglError(char *file, int line);
	void display();
	void drawLine( cv::Mat &line);
	double _x1,_y1,_x2,_y2; // points on epipolar line
	int _id;
	void displayClickedPoint(int x, int y);
};

#endif  /* _GLWIDGET_H */