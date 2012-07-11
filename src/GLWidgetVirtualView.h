#ifndef GLWIDGET_VIRTUAL_VIEW_H
#define GLWIDGET_VIRTUAL_VIEW_H


#include "vsShaderLib.h"
#include <QtOpenGL/QGLWidget>
#include "virtualImage.h"
#include <glm\glm.hpp>
#include "GLWidgetAllImgs.h"
#include "GLWidget.h"
#include "framebufferObject.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "texture2D.h"

struct planeSweepParameters
{
	int _virtualWidth;
	int _virtualHeight;
	unsigned int _numOfPlanes;
	int _numOfCameras; // the number does not include the virtual view cam.
	int _halfsizeOfMask;
	int _nearPlane;
	int _farPlane;
};


class GLWidgetVirtualView : public QGLWidget 
{
Q_OBJECT
public:
	GLWidgetVirtualView(std::vector<image> **allIms, QGLWidget *sharedWidget, const QList<GLWidget*>& imageQGLWidgets);
	~GLWidgetVirtualView();
	void setObjCenterPos(glm::vec3 objCenterPos){ _objCenterPos = objCenterPos;}

private:
	std::vector<image> **_allIms;
	glm::vec3 _objCenterPos;
	QList<GLWidget*> _imageQGLWidgets;
	virtualImage _virtualImg;	// the position will be changed

protected:
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);

	int _mouseX;
	int _mouseY;

signals:
	void updateVirtualView_signal(virtualImage virImg);

private:
	VSShaderLib _shaderHandle;
	VSShaderLib _shaderHandleDisplayLayerTexture;

	planeSweepParameters _psParam;
	GLuint _cost3DTexID;
	GLuint _color3DTexID;

	GLuint _psVertexBufferHandle;
	GLuint _psVertexArrayObjectHandle;
	GLuint _displayLayerTextureVBOHandle;
	GLuint _displayLayerTextureVAOHandle;
	
	FramebufferObject *_fbo;
	void initTexture3D(GLuint &RTT3D, int imageWidth, int imageHeight, int numOfLayers, bool isColorTexture);
	void initializeVBO_VAO(float *vertices, int numOfPrimitive, GLuint &vboObject, GLuint &vaoObject);
	void initializeVBO_VAO_DisplayLayerTexture(float *vertices, GLuint &vboObject, GLuint &vaoObject);
	void displayLayedTexture(GLuint &texture);
	int printOglError(char *file, int line);

	struct cudaGraphicsResource *_cost3D_CUDAResource;
	struct cudaGraphicsResource *_color3D_CUDAResource;

	cudaArray *_cost3D_CUDAArray;
	cudaArray *_color3D_CUDAArray;


	void _CUDA_SAFE_CALL( cudaError_t error, std::string fileName, int lineNum);
	void doCudaProcessing(cudaArray *cost3D_CUDAArray, cudaArray *color3D_CUDAArray, cudaArray *syncView_CUDAArray);
	unsigned char* _outArray;	// memory in GPU

	texture2D _syncView;
	cudaArray *_syncView_CUDAArray;
	struct cudaGraphicsResource *_syncView_CUDAResource;

	void displayImage(GLuint texture, int imageWidth, int imageHeight);
	float *_mask;	// mask for gaussian filter
	float *_maskCUDA;
};




#endif