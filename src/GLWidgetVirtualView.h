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
#include <QTime>

class planeSweepParameters
{
public:
	int _virtualWidth;
	int _virtualHeight;
	unsigned int _numOfPlanes;
	int _numOfCameras; // the number does not include the virtual view cam.
	float _near;
	float _far;
	float _gaussianSigma;
		
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
	float _step;
protected:
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseDoubleClickEvent(QMouseEvent *event);

	int _mouseX;
	int _mouseY;

signals:
	void updateVirtualView_signal(virtualImage virImg);

public slots:
	void computeImageError();
	//void setPlaneParam_slot(planeSweepParameters param);
	void psFarPlaneChanged(double farPlanePos);
	void psNearPlaneChanged(double nearPlanePos);
	void psGSSigmaChanged(double sigma);
	void psNumPlaneChanged(double numOfPlanes);


private:
	VSShaderLib _shaderHandle;
	VSShaderLib _shaderHandleDisplayLayerTexture;
	VSShaderLib _shaderHandleRenderScene;

	planeSweepParameters _psParam;
	GLuint _cost3DTexID;
//	GLuint _color3DTexID;

	GLuint _psVertexBufferHandle;
	GLuint _psVertexArrayObjectHandle;

	GLuint _renderVertexBufferHandle;
	GLuint _renderVertexArrayObjectHandle;

	GLuint _displayLayerTextureVBOHandle;
	GLuint _displayLayerTextureVAOHandle;
	
	FramebufferObject *_fbo;
	FramebufferObject *_fboRenderImage;
	GLuint _depthTextureForRenderImage;
	void initDepthTextureForRenderImage(GLuint &depthTexture);


	void initTexture3D(GLuint &RTT3D, int imageWidth, int imageHeight, int numOfLayers, bool isColorTexture);
	void initializeVBO_VAO(float *vertices, int numOfPrimitive, GLuint &vboObject, GLuint &vaoObject);
	void initializeRenderVBO_VAO(GLuint &vboObject, GLuint &vaoObject);

	void initializeVBO_VAO_DisplayLayerTexture(float *vertices, GLuint &vboObject, GLuint &vaoObject);
	//void displayLayedTexture(GLuint &texture);
	void displayLayedTexture(GLuint &texture1, GLuint &texture2);
	int printOglError(char *file, int line);

	struct cudaGraphicsResource *_cost3D_CUDAResource;
//	struct cudaGraphicsResource *_color3D_CUDAResource;
	struct cudaGraphicsResource *_syncView_CUDAResource;
	//struct cudaGraphicsResource *_depthmap_CUDAResource;
	struct cudaGraphicsResource *_depthmap1_CUDAResource;
	struct cudaGraphicsResource *_depthmap2_CUDAResource;


	cudaArray *_cost3D_CUDAArray;
//	cudaArray *_color3D_CUDAArray;
	cudaArray *_syncView_CUDAArray;
	cudaArray *_depthmap1_CUDAArray;
	cudaArray *_depthmap2_CUDAArray;


	void _CUDA_SAFE_CALL( cudaError_t error, std::string fileName, int lineNum);
	void doCudaProcessing(cudaArray *cost3D_CUDAArray, cudaArray *color3D_CUDAArray, cudaArray *syncView_CUDAArray, cudaArray *depthmapView_CUDAArray);
	unsigned char* _outArray;	// memory in GPU

	texture2D _syncView;
	//texture2D _depthmapView;
	texture2D _depthmap1;
	texture2D _depthmap2;

	texture2D _renderedImage1;
	texture2D _renderedImage2;


	bool _display_Color_Depth;

	void displayImage(GLuint texture, int imageWidth, int imageHeight);

	std::string _warpingGeoFileName;
	std::string _warpingFragFileName;
	void writeGeometryShaderFile( std::string fileName);
	void writeFragmentShaderFile(std::string fileName);

	float computeErrorForOneImage(int texture1, int texture2);

	void findNearestCam(int * nearCamIndex, glm::vec3 fixedPos, int notIncluded = -1);
	void createDistTable();
	int _distTable[16];
	void doCudaGetDepth(cudaArray* cost3D_CUDAArray, cudaArray* depthmap_CUDAArray, cudaArray* syncView_CUDAArray);
	//void renderUsingDepth(int refIndex);
	void renderUsingDepth(int refIndex, int refIndex1);

	int _numOfVertices;
	QTime _t;
	float _totalTime;
	int _numOfFrame;

	


signals:
	void updateGL_SIGNAL();
};




#endif