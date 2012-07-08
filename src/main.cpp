#include <GL\glew.h>
#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include "GLWidget.h"

#include "image.h"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "opencv/highgui.h"
#include "opencv/cxcore.h"
#include "mainWindowForm.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>

int main(int argc, char *argv[]) {
		
	QApplication app(argc, argv);
	std::vector<image> imageSet;
	mainWindowForm *mainWindow = new mainWindowForm();
	mainWindow->show();
	return app.exec();
}