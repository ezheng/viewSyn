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

	/*glm::vec4 a(1.0f);
	glm::mat3 b(1,2,3,4,5,6,7,8,9);
	glm::mat4 c(b);*/
	//glm::mat4 d(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
	//glm::mat3 e(d);

	QApplication app(argc, argv);
	std::vector<image> imageSet;
	mainWindowForm *mainWindow = new mainWindowForm();
	mainWindow->show();
	return app.exec();
}