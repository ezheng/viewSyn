#include "GLWidgetVirtualView.h"
#include <opencv\cxcore.h>
#include <QMouseEvent>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include "utility.h"
#include "glm/gtc/matrix_transform.hpp"
#include <imdebuggl.h>
#include <fstream>
#include "GaussianBlurCUDA.h"
#include <glm/gtc/quaternion.hpp>

extern void launchCudaProcess(cudaArray *cost3D_CUDAArray, cudaArray *color3D_CUDAArray, unsigned char *out_array, int imgWidth, int imgHeight, int numOfImages, unsigned int numOfCandidatePlanes);
extern void launchCudaGetDepthMap(cudaArray *cost3D_CUDAArray, cudaArray *depthmap_CUDAArray, cudaArray *depthmapView_CUDAArray,
	int width, int height, unsigned int numOfCandidatePlanes, float near, float far, float step);
extern void launchCudaWriteDepthIndexToImage(cudaArray *depthmap_CUDAArray, cudaArray *syncView_CUDAArray, int width, int height, float near, float far, float step);

#define printOpenGLError() printOglError(__FILE__, __LINE__)
#define CUDA_SAFE_CALL(err) _CUDA_SAFE_CALL(err, __FILE__, __LINE__)

int GLWidgetVirtualView:: printOglError(char *file, int line)
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


GLWidgetVirtualView :: GLWidgetVirtualView(std::vector<image> **allIms, QGLWidget *sharedWidget,
	const QList<GLWidget*>& imageQGLWidgets): 
	_allIms(allIms), QGLWidget((QWidget*)NULL, sharedWidget), _virtualImg((**allIms)[4], 4),
		_mouseX(0), _mouseY(0), _imageQGLWidgets(imageQGLWidgets), _cost3DTexID(0), _fbo(NULL), _fboRenderImage(0),_depthTextureForRenderImage(0),
		_psVertexBufferHandle(0), 
		_psVertexArrayObjectHandle(0), _syncView((**allIms)[0]._image.cols, (**allIms)[0]._image.rows), 
		_renderedImage1((**allIms)[0]._image.cols, (**allIms)[0]._image.rows), 
		_renderedImage2((**allIms)[0]._image.cols, (**allIms)[0]._image.rows), 
		_renderVertexArrayObjectHandle(0), _renderVertexBufferHandle(0),
		_display_Color_Depth(true),
		_depthmap1((**allIms)[0]._image.cols, (**allIms)[0]._image.rows), 
		_depthmap2((**allIms)[0]._image.cols, (**allIms)[0]._image.rows),

		_weightOfView(1.0f)
{ 
	
	int width, height;
	if( (*allIms)->size() <1){	
		width = 200, height = 100;
	} // set a predefined size if there is no image
	else{
		width = (**allIms)[0]._image.cols; 
		height = (**allIms)[0]._image.rows;
	}
	this->setGeometry(0,0, width, height);
	_psParam._virtualHeight = (**allIms)[0]._image.rows; 
	_psParam._virtualWidth = (**allIms)[0]._image.cols; 
	_psParam._numOfPlanes = 120;
	_psParam._numOfCameras  = 5;	
	_psParam._gaussianSigma = 1.0f;
	_psParam._near = 3.5f;
	_psParam._far = 10.f;
	//_psParam._near = .45f;
	//_psParam._far = .6f;

	_virtualImg.setProjMatrix(_psParam._near, _psParam._far);

	// prepare the shader file given the number of cameras available
	std::string filePath = std::string(std::getenv("SHADER_FILE_PATH"));
	_warpingGeoFileName = filePath + "\\warping.geom";
	_warpingFragFileName = filePath + "\\warping.frag";
	//writeGeometryShaderFile(_warpingGeoFileName);
	//writeFragmentShaderFile(_warpingFragFileName);

	_nearestCamIndex = _virtualImg._camIndex + 1;
	_nearestCamIndex = _nearestCamIndex >= _psParam._numOfCameras? (_nearestCamIndex-1):_nearestCamIndex;
}

void GLWidgetVirtualView::psFarPlaneChanged(double farPlanePos)
{
	std::cout<< "psFarPlaneChanged"<< std::endl;
	_psParam._far = static_cast<float>(farPlanePos);
	_virtualImg.setProjMatrix(_psParam._near, _psParam._far);
	updateGL();
	_virtualImg.calcPlaneCoords();
	emit updateVirtualView_signal(_virtualImg);
}

void GLWidgetVirtualView::psNearPlaneChanged(double nearPlanePos)
{
	std::cout<< "psNearPlaneChanged"<< std::endl;
	_psParam._near = static_cast<float>(nearPlanePos);
	_virtualImg.setProjMatrix(_psParam._near, _psParam._far);
	updateGL();
	_virtualImg.calcPlaneCoords();
	emit updateVirtualView_signal(_virtualImg);
}
void GLWidgetVirtualView::psGSSigmaChanged(double sigma)
{
	std::cout<< "psGSSigmaChanged"<< std::endl;
	_psParam._gaussianSigma = static_cast<float>(sigma);
	updateGL();

}
void GLWidgetVirtualView::psNumPlaneChanged(double numOfPlanes)
{
	//std::cout<< "psNumPlaneChanged"<< std::endl;
	//_psParam._numOfPlanes = static_cast<int>(50);
	//// re-initialize the textures
	//updateGL();
}

void GLWidgetVirtualView::initTexture3D(GLuint & RTT3D, int imageWidth, int imageHeight, int numOfLayers, bool isColorTexture)
{
		glGenTextures(1, &RTT3D);
    glBindTexture(GL_TEXTURE_3D, RTT3D);
    // set basic parameters
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    
	if(isColorTexture)
	{
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, imageWidth, imageHeight, numOfLayers, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
	else
	{	
		glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F , imageWidth, imageHeight, numOfLayers, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}

	printOpenGLError();
}

void GLWidgetVirtualView::initializeVBO_VAO(float *vertices, int numOfPrimitive, GLuint &vboObject, GLuint &vaoObject)
{
	if(vboObject > 0 || vaoObject >0)	{
		std::cout<< "WARNING: VAO or VBO is not empty. None of them are recreated" << std::endl;
		return;
	}
	glGenBuffers(1, & vboObject);
	glBindBuffer(GL_ARRAY_BUFFER, vboObject);
	glBufferData(GL_ARRAY_BUFFER, 3 *sizeof(float) * numOfPrimitive, vertices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vaoObject);
	glBindVertexArray(vaoObject);

	glEnableVertexAttribArray(VSShaderLib::VERTEX_COORD_ATTRIB); // Vertex position
	glVertexAttribPointer( VSShaderLib::VERTEX_COORD_ATTRIB, 3,		// Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, or 4. The initial value is 4.
		GL_FLOAT, GL_FALSE, 0,(GLubyte *)NULL );	// 0 is the stride
	
	printOpenGLError();
}

void GLWidgetVirtualView::initializeRenderVBO_VAO(GLuint &vboObject, GLuint &vaoObject)
{
	if(vboObject > 0 || vaoObject >0)	{
		std::cout<< "WARNING: VAO or VBO is not empty. None of them are recreated" << std::endl;
		return;
	}
	int imageWidth = _psParam._virtualWidth;
	int imageHeight = _psParam._virtualHeight;

	int numOfPrimitive = (imageWidth - 1) * (imageHeight - 1) * 2;
	_numOfVertices = numOfPrimitive * 3; // each triangle has 3 vertices
	int numOfDigits = _numOfVertices * 2; // each vertices has two coordinates: x and y

	float *vertices = new float[numOfDigits];

//	float offsetX = 1.0f/imageWidth/2.0f;
//	float offsetY = 1.0f/imageHeight/2.0f;
	for(int i = 0; i<imageHeight - 1; i++)
	{
		for(int j = 0; j<imageWidth - 1; j++)
		{
			// 1st triangle
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 ] = (j+0.5)/imageWidth;
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 1] = (i+0.5)/imageHeight;

			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 2] = (j + 1.5)/imageWidth;
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 3] = (i + 1.5)/imageHeight;

			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 4] = (j + 1.5)/imageWidth;
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 5] = (i + 0.5)/imageHeight;
			// 2nd triangle
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 6] = (j + 0.5)/imageWidth;
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 7] = (i + 0.5)/imageHeight;

			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 8] = (j + 1.5)/imageWidth;
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 9] = (i + 1.5)/imageHeight;

			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 10] = (j + 0.5)/imageWidth;
			vertices[(i * (imageWidth - 1) +j) * 3 * 2 * 2 + 11] = (i + 1.5)/imageHeight;	
		}
	}
	glGenBuffers(1, & vboObject);
	glBindBuffer(GL_ARRAY_BUFFER, vboObject);
	glBufferData(GL_ARRAY_BUFFER, numOfDigits*sizeof(float), vertices, GL_STATIC_DRAW);

	//float xy[6] = {-0.5, -0.5, -0.5, 0.5, 0.5, 0.7};
	//glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), xy, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vaoObject);
	glBindVertexArray(vaoObject);

	glEnableVertexAttribArray(VSShaderLib::VERTEX_COORD_ATTRIB); // Vertex position
	glVertexAttribPointer( VSShaderLib::VERTEX_COORD_ATTRIB, 2,		// Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, or 4. The initial value is 4.
		GL_FLOAT, GL_FALSE, 0,(GLubyte *)NULL );	// 0 is the stride
	printOpenGLError();
}

void GLWidgetVirtualView::initializeGL()
{
	glewInit();	// Initialize glew
	CUDA_SAFE_CALL(cudaGLSetGLDevice(0));
	// create an empty 2d texture for view synthesis
	_syncView.create(NULL);	// just allocate memory, no image data is uploaded
	_renderedImage1.create(NULL);
	_renderedImage2.create(NULL);
	initDepthTextureForRenderImage(_depthTextureForRenderImage);

	//_depthmapView.create(NULL);
	printOpenGLError();
	//_depthmap1.createGL_R32UI();
	_depthmap1.createGL_R32I();
	printOpenGLError();
	//_depthmap2.createGL_R32UI();
	_depthmap2.createGL_R32I();
	printOpenGLError();
	//--------------------------------------------------------
	// set up shader
	std::string filePath = std::string(std::getenv("SHADER_FILE_PATH"));

	_shaderHandle.init();
	_shaderHandle.loadShader(VSShaderLib::VERTEX_SHADER, (filePath + "\\warping.vert").c_str());
	std::cout<<"vertex shader: " << _shaderHandle.getShaderInfoLog(VSShaderLib::VERTEX_SHADER)<<std::endl;
	_shaderHandle.loadShader(VSShaderLib::GEOMETRY_SHADER, _warpingGeoFileName.c_str());
	std::cout<<"geometry shader: " << _shaderHandle.getShaderInfoLog(VSShaderLib::GEOMETRY_SHADER)<<std::endl;
	_shaderHandle.loadShader(VSShaderLib::FRAGMENT_SHADER, (filePath + "\\warping.frag").c_str());
	std::cout<<"fragment shader: " << _shaderHandle.getShaderInfoLog(VSShaderLib::FRAGMENT_SHADER)<< std::endl;
	_shaderHandle.prepareProgram();
	// set up 3d texture that I can render to (number of layers should be set )	
	initTexture3D( _cost3DTexID, _psParam._virtualWidth, _psParam._virtualHeight, _psParam._numOfPlanes, false);
//	initTexture3D( _color3DTexID, _psParam._virtualWidth, _psParam._virtualHeight, _psParam._numOfPlanes, true);
	printOpenGLError();

	_shaderHandleRenderScene.init();
	_shaderHandleRenderScene.loadShader(VSShaderLib::VERTEX_SHADER, (filePath + "\\renderScene.vert").c_str());
	std::cout<<"renderScene vertex shader: " << _shaderHandleRenderScene.getShaderInfoLog(VSShaderLib::VERTEX_SHADER)<<std::endl;
	_shaderHandleRenderScene.loadShader(VSShaderLib::GEOMETRY_SHADER, (filePath + "\\renderScene.geom").c_str());
	std::cout<<"renderScene geometry shader: " << _shaderHandleRenderScene.getShaderInfoLog(VSShaderLib::GEOMETRY_SHADER)<<std::endl;
	_shaderHandleRenderScene.loadShader(VSShaderLib::FRAGMENT_SHADER, (filePath + "\\renderScene.frag").c_str());
	std::cout<<"renderScene fragment shader: " << _shaderHandleRenderScene.getShaderInfoLog(VSShaderLib::FRAGMENT_SHADER)<< std::endl;
	_shaderHandleRenderScene.prepareProgram();
	printOpenGLError();

	//--------------------------------------------------------
	// set up vbo
	float vertices[3] = {0.0f, 0.0f, 0.0f};
	initializeVBO_VAO(vertices, 1, _psVertexBufferHandle, _psVertexArrayObjectHandle);	// here 1 is the number of primitives

	// set up fbo?
	_fbo = new FramebufferObject();
	_fbo->Bind();
	_fbo->AttachTexture(GL_TEXTURE_3D, _cost3DTexID, GL_COLOR_ATTACHMENT0, 0, -1); // -1 means no specific layer is specified, 0 is the mipmap level
	//_fbo->AttachTexture(GL_TEXTURE_3D, _color3DTexID, GL_COLOR_ATTACHMENT1, 0, -1); // -1 means no specific layer is specified, 0 is the mipmap level
	//GLenum drawBufs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	//glDrawBuffers(2, drawBufs);
	_fbo->IsValid(std::cout);
	_fbo->Disable();

	_fboRenderImage = new FramebufferObject();
	_fboRenderImage->Bind();
	_fboRenderImage->AttachTexture(GL_TEXTURE_2D, _depthTextureForRenderImage, GL_DEPTH_ATTACHMENT);
	_fboRenderImage->IsValid(std::cout);
	_fboRenderImage->Disable();

	//------------------------
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glDisable(GL_DEPTH_TEST);	// do not need depth buffer
	glEnable(GL_TEXTURE_2D);
	printOpenGLError();

	// set up shader for displaying layered texture
	_shaderHandleDisplayLayerTexture.init();
	_shaderHandleDisplayLayerTexture.loadShader(VSShaderLib::VERTEX_SHADER, (filePath + "\\displayLayeredTex.vert").c_str());
	std::cout<<"vertex shader: " << _shaderHandleDisplayLayerTexture.getShaderInfoLog(VSShaderLib::VERTEX_SHADER)<<std::endl;
	_shaderHandleDisplayLayerTexture.loadShader(VSShaderLib::FRAGMENT_SHADER, (filePath + "\\displayLayeredTex.frag").c_str());
	std::cout<<"fragment shader: " << _shaderHandleDisplayLayerTexture.getShaderInfoLog(VSShaderLib::FRAGMENT_SHADER)<< std::endl;
	_shaderHandleDisplayLayerTexture.prepareProgram();
	printOpenGLError();
	// set up vbo and vao for displaying layered textures
	float verticesQuadWithTexCoord[] = {-1.0, -1.0, 0.5, 1.0, -1.0, 0.5,  1.0, 1.0, 0.5,  -1.0, 1.0, 0.5 ,
		// tex coord
	0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
	initializeVBO_VAO_DisplayLayerTexture(verticesQuadWithTexCoord, _displayLayerTextureVBOHandle, _displayLayerTextureVAOHandle);
	initializeRenderVBO_VAO(_renderVertexBufferHandle, _renderVertexArrayObjectHandle);

	printOpenGLError();

	// register the 3d texture so that CUDA can use it
	size_t free, total; float mb = 1<<20;
	cudaMemGetInfo (&free, &total); std::cout<< "free memory is: " << free/mb << "MB total memory is: " << total/mb << " MB" << std::endl;

	CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&_cost3D_CUDAResource, _cost3DTexID, 
				  GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore ));// register the 3d texture
	cudaMemGetInfo (&free, &total); std::cout<< "free memory is: " << free/mb << "MB total memory is: " << total/mb << " MB" << std::endl;

	//CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&_color3D_CUDAResource, _color3DTexID, 
	//			  GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone ));// register the 3d texture
		
	cudaMemGetInfo (&free, &total); std::cout<< "free memory is: " << free/mb << "MB total memory is: " << total/mb << " MB" << std::endl;

	CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&_syncView_CUDAResource, _syncView._textureID, 
				  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ));// register the 2d texture
	
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&_depthmap1_CUDAResource, _depthmap1._textureID, 
				  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ));// register the 2d surface texture
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&_depthmap2_CUDAResource, _depthmap2._textureID, 
				  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ));// register the 2d surface texture

	
	createDistTable();
	printOpenGLError();

	_t.start();
	_totalTime = 0;
	_numOfFrame = 0;
	QObject::connect(this, SIGNAL(updateGL_SIGNAL()), this, SLOT(updateGL()), Qt::QueuedConnection);
}

void GLWidgetVirtualView::initDepthTextureForRenderImage(GLuint &depthTexture)
{
	glGenTextures(1, &depthTexture);	
	glBindTexture(GL_TEXTURE_2D, depthTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	printOpenGLError();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F,	_psParam._virtualWidth, _psParam._virtualHeight, 0, GL_DEPTH_COMPONENT,GL_UNSIGNED_BYTE, NULL);
	printOpenGLError();
	
}

void GLWidgetVirtualView:: displayImage(GLuint texture, int imageWidth, int imageHeight)
{
	GLint prevProgram;
	glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

	glUseProgram(0);
	glClear(GL_COLOR_BUFFER_BIT);  
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);    
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	printOpenGLError();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
    glLoadIdentity();

    glViewport(0, 0, imageWidth, imageHeight);
	printOpenGLError();	
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0); 	glVertex3f(-1.0, 1.0, 0.5);
    glTexCoord2f(1.0, 1.0); 	glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(1.0, 0.0); 	glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(0.0, 0.0); 	glVertex3f(-1.0, -1.0, 0.5);
    glEnd();
	printOpenGLError();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
//    glDisable(GL_TEXTURE_2D);	
	glUseProgram(GLuint(prevProgram));
	printOpenGLError();	

}

void GLWidgetVirtualView::doCudaProcessing(cudaArray *cost3D_CUDAArray, cudaArray *color3D_CUDAArray, 
	cudaArray *syncView_CUDAArray, cudaArray *depthmapView_CUDAArray)
{
	int width = this->_psParam._virtualWidth;
	int height = this->_psParam._virtualHeight;
	int numOfImages = this->_psParam._numOfCameras;
	unsigned int numOfCandidatePlanes = this->_psParam._numOfPlanes;

	// do gaussian filter:
	GaussianBlurCUDA gaussianF(width, height, _psParam._gaussianSigma);
	gaussianF.Filter(cost3D_CUDAArray, _psParam._numOfPlanes);
		
	size_t free, total; float mb = 1<<20;	
	cudaMemGetInfo (&free, &total); std::cout<< "free memory is: " << free/mb << "MB total memory is: " << total/mb << " MB" << std::endl;
	CUDA_SAFE_CALL(cudaMalloc((void**)&_outArray, width * height * 4 * sizeof(GLubyte)));
//	{
	//imdebugTexImage(GL_TEXTURE_3D, _color3DTexID,  GL_RGBA);
	
		//cudaTimer timer;
		//timer.start();
		
	//if(_display_Color_Depth)
	//	launchCudaProcess(cost3D_CUDAArray,color3D_CUDAArray, _outArray, width, height, numOfImages, numOfCandidatePlanes);
	//else	
		//launchCudaGetDepthMap(cost3D_CUDAArray, depthmapView_CUDAArray, width, height, numOfCandidatePlanes, _psParam._near, _psParam._far, _step);
		//gaussianF.Filter(color3D_CUDAArray, _psParam._numOfPlanes);
		
		//timer.stop();
		
		//CUDA_SAFE_CALL(cudaDeviceSynchronize());
		//imdebugTexImage(GL_TEXTURE_3D, _color3DTexID,  GL_RGBA, 15);
//	}
	CUDA_SAFE_CALL(cudaMemcpyToArray(syncView_CUDAArray, 0, 0, _outArray, height * width * 4 * sizeof(GLubyte),
 		cudaMemcpyDeviceToDevice));		// Copy the data back to the texture

	CUDA_SAFE_CALL(cudaFree((void*)_outArray));
}

void GLWidgetVirtualView::_CUDA_SAFE_CALL( cudaError_t err, std::string file, int line)
{
   if (err != cudaSuccess) {
       // printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
       //         file, line );
	   std::cout<< cudaGetErrorString( err ) << " in file: " << file << " at line: " << line << std::endl;
        exit( EXIT_FAILURE );
    }
}

void GLWidgetVirtualView::initializeVBO_VAO_DisplayLayerTexture(float *vertices, GLuint &vboObject, GLuint &vaoObject)
{
	glGenBuffers(1, &vboObject);
	glBindBuffer(GL_ARRAY_BUFFER, vboObject);
	glBufferData(GL_ARRAY_BUFFER, 20 * sizeof(float), vertices, GL_STATIC_DRAW);	// generate vbo, and upload the data

	glGenVertexArrays(1, &vaoObject);
	glBindVertexArray(vaoObject);
	glEnableVertexAttribArray(VSShaderLib::VERTEX_COORD_ATTRIB); // Vertex position
	glVertexAttribPointer( VSShaderLib::VERTEX_COORD_ATTRIB, 3,		// Specifies the number of components per generic vertex attribute. Must be 1, 2, 3, or 4. The initial value is 4.
		GL_FLOAT, GL_FALSE, 0,(GLubyte *)NULL );	// 0 is the stride

	glEnableVertexAttribArray(VSShaderLib::TEXTURE_COORD_ATTRIB);
	glVertexAttribPointer(VSShaderLib::TEXTURE_COORD_ATTRIB, 2,
		GL_FLOAT, GL_FALSE, 0, (void *)48);	// 12 * sizeof(float)
}

void GLWidgetVirtualView::displayLayedTexture(GLuint &texture1, GLuint &texture2)
{
	//glClearColor(1 ,0.5,0,0);
	glClear(GL_COLOR_BUFFER_BIT); 
	glUseProgram(_shaderHandleDisplayLayerTexture.getProgramIndex());
	glDisable(GL_DEPTH_TEST);

	int textureUint = 0;
	glActiveTexture(GL_TEXTURE0 + textureUint);
	glBindTexture(GL_TEXTURE_2D, texture1);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);	
	//_shaderHandleDisplayLayerTexture.setUniform("numOfLayers", static_cast<int>(_psParam._numOfPlanes));
	_shaderHandleDisplayLayerTexture.setUniform("texs0",&textureUint);
	printOpenGLError();	
	textureUint ++;
	glActiveTexture(GL_TEXTURE0 + textureUint);
	glBindTexture(GL_TEXTURE_2D, texture2);
	_shaderHandleDisplayLayerTexture.setUniform("texs1",&textureUint);

	_shaderHandleDisplayLayerTexture.setUniform("weight", _weightOfView);
	_shaderHandleDisplayLayerTexture.setUniform("x_texSize", 1.0f/ static_cast<float>(_psParam._virtualWidth));
	_shaderHandleDisplayLayerTexture.setUniform("y_texSize", 1.0f/ static_cast<float>(_psParam._virtualHeight));
	//std::cout<<_weightOfView << std::endl;
	
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
    glLoadIdentity();

	GLfloat modelView[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	_shaderHandleDisplayLayerTexture.setUniform("modelViewMatrix1", modelView);
	GLfloat projection[16];
	glGetFloatv(GL_PROJECTION_MATRIX, projection);
	_shaderHandleDisplayLayerTexture.setUniform("projectionMatrix1", projection);
	
	glViewport(0, 0, _psParam._virtualWidth, _psParam._virtualHeight);
	printOpenGLError();
	
	// bind and draw the quads
	glBindVertexArray(_displayLayerTextureVAOHandle);
	glDrawArrays(GL_QUADS, 0 , 4);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_DEPTH_TEST);
}

GLWidgetVirtualView::~GLWidgetVirtualView()
{
	if(_fbo != NULL)
		delete _fbo;
	if(_fboRenderImage != NULL)
		delete _fboRenderImage;
}

void GLWidgetVirtualView::resizeGL(int w, int h)
{
	//glViewport(0, 0, w, h);
	bool firstEntry= true;
	if(firstEntry)
	{	glViewport(0, 0, _psParam._virtualWidth, _psParam._virtualHeight); firstEntry = false;}
	else
		glViewport(0, 0, w, h);	
}

void GLWidgetVirtualView::findNearestCam(int nearCamIndex[2], glm::vec3 fixedPos, int notIncluded)
{
	// calculate based on the distance 
	
	float minDist = 1000000000000.0f;
	float minDist2nd = minDist + 1.0f;

	for(int i = 0; i< _psParam._numOfCameras; i++)
	{		
		if(notIncluded == i)
			continue;
		float dist = glm::distance(fixedPos, (**_allIms)[i]._glmC);
		if(dist < minDist)
		{
			minDist2nd = minDist;
			nearCamIndex[1] = nearCamIndex[0]; 

			//----------
			minDist = dist;
			nearCamIndex[0] = i;
		}
		else if(dist < minDist2nd)
		{
			minDist2nd = dist;
			nearCamIndex[1] = i;
		}
	}
}

void GLWidgetVirtualView::createDistTable()
{
	// init
	for(int i = 0; i<16; i++)
		_distTable[i] = 0;
	// 
	for(int ref = 0; ref < _psParam._numOfCameras; ref++)
	{
		findNearestCam( &(_distTable[ref*2]), (**_allIms)[ref]._glmC, ref);
	}

}

void GLWidgetVirtualView::paintGL()
{
	_totalTime += _t.restart();
	 ++_numOfFrame;
	 if(_numOfFrame % 50 == 0)
	 std::cout << "frame rate is: " <<  static_cast<float>(_numOfFrame) / _totalTime * 1000 << " Hz"<< std::endl;

	// ***** maybe no depth buffer is needed in first pass
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 	
	
	//find the nearest two cameras.
	int nearCamIndex[2] = {0};
	//findNearestCam(nearCamIndex, _virtualImg._glmC);
	nearCamIndex[0] = _virtualImg._camIndex;
	nearCamIndex[1] = _nearestCamIndex;

	// set up the uniforms: images, transformation matrix, etc...
	glUseProgram(_shaderHandle.getProgramIndex());
	_step = 2.0f / static_cast<float>(_psParam._numOfPlanes + 1);
	_shaderHandle.setUniform("step", &_step);
		// matrix:
	glm::mat4 projScaleTrans = glm::translate(glm::vec3(0.5f)) * glm::scale(glm::vec3(0.5f));
	
	// *****
	//int numOfImages = _psParam._numOfCameras;
	//glm::mat4 *modelViewProj = new glm::mat4[numOfImages];

	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_depthmap1_CUDAResource, 0));
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&_depthmap1_CUDAArray, _depthmap1_CUDAResource, 0, 0));	// 0th layer, 0 mipmap level
	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_depthmap2_CUDAResource, 0));
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&_depthmap2_CUDAArray, _depthmap2_CUDAResource, 0, 0));	// 0th layer, 0 mipmap level
	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_syncView_CUDAResource, 0));
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&_syncView_CUDAArray, _syncView_CUDAResource, 0, 0));	// 0th layer, 0 mipmap level

	for(int ref = 0; ref<2; ref++)
	{
		//int ref = refr;
		int refIndex = nearCamIndex[ref];
		//std::cout<<ref<<std::endl;

		glActiveTexture(GL_TEXTURE0 + 3);
		glBindTexture(GL_TEXTURE_2D, _imageQGLWidgets[refIndex]->_tex._textureID);
		int x = 3;
		_shaderHandle.setUniform("tex2", &x);	// this is the reference camera


		(**_allIms)[refIndex].setProjMatrix(_psParam._near, _psParam._far);
		//glm::mat4 virtInverseModelViewProj = glm::inverse((**_allIms)[refIndex]._modelViewMatrix) 
		//	* glm::inverse((**_allIms)[refIndex]._projMatrix);
		glm::mat4 virtInverseModelViewProj = glm::inverse((**_allIms)[refIndex]._projMatrix * (**_allIms)[refIndex]._modelViewMatrix) ;
			
		float allTransformMatrix[16 * 2] = {0};
		for(int i = 0; i<2; i++)
		{
			int nearIndex = _distTable[2*refIndex + i];
			glm::mat4x4 modelViewProj = (**_allIms)[nearIndex]._projMatrix * (**_allIms)[nearIndex]._modelViewMatrix;
			glm::mat4x4 transformMatrix = projScaleTrans * modelViewProj * virtInverseModelViewProj;
			std::copy ( &(transformMatrix[0][0]), &(transformMatrix[0][0]) + 16, allTransformMatrix + 16 * i );
			std::stringstream ss; ss<<i;

			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_2D, _imageQGLWidgets[nearIndex]->_tex._textureID);
			_shaderHandle.setUniform(("tex" + ss.str()).c_str(), &i);
			printOpenGLError();
		}
		_shaderHandle.setUniform( "transformMatrix[0]", allTransformMatrix);

		_fbo->Bind();	// bind the 3d cost textures. 
		_fbo->IsValid(std::cout);

		// draw() using vao
		glBindVertexArray(_psVertexArrayObjectHandle);
		printOpenGLError();

		//if(ref == 0)
		glDrawArraysInstanced(GL_POINTS, 0, 1, _psParam._numOfPlanes); 
		//glFinish();	

		// unbind fbo, vao
		glBindVertexArray(0);
		_fbo->Disable();

	//	for(int layer = 20; layer < 80; layer++)
	//		imdebugTexImage(GL_TEXTURE_3D, _color3DTexID,  GL_RGBA, layer);
		
		if(ref == 0)
		{
			// register color image
			

			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_cost3D_CUDAResource, 0));
			CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&_cost3D_CUDAArray, _cost3D_CUDAResource, 0, 0));	// 0th layer, 0 mipmap level
			doCudaGetDepth(_cost3D_CUDAArray, _depthmap1_CUDAArray, _syncView_CUDAArray, nearCamIndex[ref]);
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_cost3D_CUDAResource, 0));
		

		}
		else if (ref == 1)
		{
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_cost3D_CUDAResource, 0));
			CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&_cost3D_CUDAArray, _cost3D_CUDAResource, 0, 0));	// 0th layer, 0 mipmap level
			doCudaGetDepth(_cost3D_CUDAArray, _depthmap2_CUDAArray, _syncView_CUDAArray, nearCamIndex[ref]);
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_cost3D_CUDAResource, 0));

		}
	}
	// unmap
	
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_depthmap2_CUDAResource, 0));
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_depthmap1_CUDAResource, 0));
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_syncView_CUDAResource, 0));
	//imdebugTexImage(GL_TEXTURE_2D, _syncView._textureID, GL_RGBA);
	//displayImage(_syncView._textureID, _psParam._virtualWidth, _psParam._virtualHeight);
	
	// --------------------------
	if(_display_Color_Depth)
	{
		//glEnable(GL_DEPTH_TEST);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 	
		renderUsingDepth(nearCamIndex[0], nearCamIndex[1]);
		//glDisable(GL_DEPTH_TEST);
	}
	else
		displayImage(_syncView._textureID, _psParam._virtualWidth, _psParam._virtualHeight);

	//
	
	//emit updateGL_SIGNAL(); 
	// that's it!!!	
}

void GLWidgetVirtualView::renderUsingDepth(int refIndex, int refIndex1)
{
	glUseProgram(_shaderHandleRenderScene.getProgramIndex());
	_shaderHandleRenderScene.setUniform("step", &_step);

	for(int i = 0; i< 2; i++)
	{
		glActiveTexture(GL_TEXTURE0); 
		if(i == 0)
			glBindTexture(GL_TEXTURE_2D, _depthmap1._textureID);
		else 
			glBindTexture(GL_TEXTURE_2D, _depthmap2._textureID);
		int id = 0;
		_shaderHandleRenderScene.setUniform("depthTex0", &id);
	
	//_virtualImg.setProjMatrix(0.1, 1000);
		glm::mat4x4 transform;
		if( i == 0)
			transform = _virtualImg._projMatrix * _virtualImg._modelViewMatrix * glm::inverse( (**_allIms)[refIndex]._projMatrix * (**_allIms)[refIndex]._modelViewMatrix);
		else
			transform = _virtualImg._projMatrix * _virtualImg._modelViewMatrix * glm::inverse( (**_allIms)[refIndex1]._projMatrix * (**_allIms)[refIndex1]._modelViewMatrix);

		_shaderHandleRenderScene.setUniform("transform0", &transform[0][0]);
	
		glActiveTexture(GL_TEXTURE1);
		if( i == 0)
			glBindTexture(GL_TEXTURE_2D, _imageQGLWidgets[refIndex]->_tex._textureID);
		else
			glBindTexture(GL_TEXTURE_2D, _imageQGLWidgets[refIndex1]->_tex._textureID);
		id = 1;
		_shaderHandleRenderScene.setUniform("textures0", &id);

	/*glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, _imageQGLWidgets[refIndex1]->_tex._textureID);
	id = 3;
	_shaderHandleRenderScene.setUniform("textures", &id);*/

		_fboRenderImage->Bind();
		if( i == 0)
			_fboRenderImage->AttachTexture(GL_TEXTURE_2D, _renderedImage1._textureID, GL_COLOR_ATTACHMENT0);
		else
			_fboRenderImage->AttachTexture(GL_TEXTURE_2D, _renderedImage2._textureID, GL_COLOR_ATTACHMENT0);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
		glBindVertexArray(_renderVertexArrayObjectHandle);
		printOpenGLError();
		glDrawArrays(GL_TRIANGLES, 0, _numOfVertices ); 
		_fboRenderImage->Disable();
	}
	
	displayLayedTexture(_renderedImage1._textureID, _renderedImage2._textureID);
	//displayImage(_renderedImage1._textureID, _psParam._virtualWidth, _psParam._virtualHeight);
	

}

void GLWidgetVirtualView::doCudaGetDepth(cudaArray* cost3D_CUDAArray, cudaArray* depthmap_CUDAArray, cudaArray* syncView_CUDAArray, int refIndex)
{
	int width = this->_psParam._virtualWidth;
	int height = this->_psParam._virtualHeight;
	int numOfImages = this->_psParam._numOfCameras;
	unsigned int numOfCandidatePlanes = this->_psParam._numOfPlanes;

	// do gaussian filter:
	if(_psParam._gaussianSigma != 0)
	{
		GaussianBlurCUDA gaussianF(width, height, _psParam._gaussianSigma);
		gaussianF.Filter(cost3D_CUDAArray, _psParam._numOfPlanes);
	}
	launchCudaGetDepthMap(cost3D_CUDAArray, depthmap_CUDAArray,syncView_CUDAArray, width, height, _psParam._numOfPlanes, _psParam._near, _psParam._far, _step);

	// run high pass filter to get reliable depth. Unreliable pixel is set to -1.

	GaussianBlurCUDA gaussianF(width, height, 3.0f);
	gaussianF.RemoveUnreliableDepth(depthmap_CUDAArray);
	// filling holes:
	
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&_colorImage_CUDAResource, _imageQGLWidgets[refIndex]->_tex._textureID, 
				  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly ));// register the 3d texture
	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_colorImage_CUDAResource, 0));
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&_colorImage_CUDAArray, _colorImage_CUDAResource, 0, 0));	// 0th layer, 0 mipmap level
	gaussianF.fillHolesDepth(depthmap_CUDAArray, _colorImage_CUDAArray);
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_colorImage_CUDAResource, 0));
	CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(_colorImage_CUDAResource));
	
	// write the depthmap for viewing purpose:
	launchCudaWriteDepthIndexToImage( depthmap_CUDAArray,syncView_CUDAArray, width, height, _psParam._near, _psParam._far, _step);

}

void GLWidgetVirtualView::computeImageError()
{
	// the image array
	glm::mat4x4 oldProjMatrix = _virtualImg._projMatrix;
	glm::mat4x4 oldModelviewMatrix = _virtualImg._modelViewMatrix;

	float cost = 0;
	// _virtual view: _syncView._textureID
	for(int i = 0; i<_psParam._numOfCameras; i++)
	{
		_virtualImg._projMatrix = (**_allIms)[i]._projMatrix;
		_virtualImg._modelViewMatrix = (**_allIms)[i]._modelViewMatrix;
		makeCurrent();		
		paintGL();
		cost += computeErrorForOneImage(_syncView._textureID, _imageQGLWidgets[i]->_tex._textureID );
	}
	_virtualImg._projMatrix = oldProjMatrix;
	_virtualImg._modelViewMatrix = oldModelviewMatrix;

	std::cout << " the average cost is: " << cost << std::endl;
	updateGL();

}

float GLWidgetVirtualView::computeErrorForOneImage(int texture1, int texture2)
{
	// read back the value
//	imdebugTexImage(GL_TEXTURE_2D, texture1, GL_RGBA);
//	imdebugTexImage(GL_TEXTURE_2D, texture2, GL_RGBA);

	int w, h;
  int prevTexBind;
  glGetIntegerv(GL_TEXTURE_2D, &prevTexBind);
  glBindTexture(GL_TEXTURE_2D, texture1);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
 
  //
   GLubyte *data1 = new GLubyte[w * h * 4 ];
   GLubyte *data2 = new GLubyte[w * h * 4 ];
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data1 );

   glBindTexture(GL_TEXTURE_2D, texture2);
   glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data2 );

   float cost = 0;
   for(int i = 0; i < w * h; i += 4)
	{
		cost += abs(data1[i] - data2[i]);
		cost += abs(data1[i+1] - data2[i+1]);
		cost += abs(data1[i+2] - data2[i+2]);
	}   
  cost = cost / (w * h * 3);
  glBindTexture(GL_TEXTURE_2D, prevTexBind);
  delete data1;
  delete data2;
  return cost;
}


void GLWidgetVirtualView::mousePressEvent(QMouseEvent *event)
{
	_mouseX = event->x();
	_mouseY = event->y();
	std::cout<< "X: " << _mouseX << "Y: " << _mouseY << std::endl;
}

void GLWidgetVirtualView::mouseDoubleClickEvent(QMouseEvent *event)
{
	_display_Color_Depth = !_display_Color_Depth;
	updateGL();
}

void GLWidgetVirtualView::mouseMoveEvent(QMouseEvent *event)
{
	int deltaX =  event->x() - _mouseX;
	int deltaY =  event->y() - _mouseY;

	float s = static_cast<float>( std::max(this->width(), this->height()));
	float rangeX = static_cast<float>(deltaX) / (s + 0.00001);
	float rangeY = static_cast<float>(deltaY) / (s + 0.00001);

	if (event->buttons() & Qt::LeftButton){
		// rotation
		glm::mat4x4 inverseVirtualModelViewMatrix = glm::inverse(_virtualImg._modelViewMatrix);
		glm::vec4 dir =   inverseVirtualModelViewMatrix *  glm::vec4(deltaY, deltaX, 0.0f, 0.0f);
		glm::normalize(dir);

		float mag = sqrt(pow(rangeX,2) + pow(rangeY,2)) * 180;
		//glm::mat4x4 transformMatrix = glm::translate(_objCenterPos) * glm::rotate(mag, dir.x, dir.y, dir.z) * 
		//	glm::translate(-_objCenterPos);

		glm::vec4 a = glm::vec4(_objCenterPos, 1.0f) + (glm::rotate(mag, dir.x, dir.y, dir.z) * glm::vec4(_virtualImg._glmC - _objCenterPos, 0.0f));
		_virtualImg._glmC = glm::vec3(a.x/a.w, a.y/a.w, a.z/a.w);

		glm::mat4x4 b = glm::transpose(glm::rotate(mag, dir.x, dir.y, dir.z) * glm::mat4x4(glm::transpose(_virtualImg._glmR)));
		_virtualImg._glmR = glm::mat3(b);
		_virtualImg.setModelViewMatrix();
		_virtualImg.setProjMatrix();
		_virtualImg.calcPlaneCoords();
		emit updateVirtualView_signal(_virtualImg);

	}
	else if(event->buttons() & Qt::RightButton)
	{
		// translation
		glm::mat4x4 inverseVirtualModelViewMatrix = glm::inverse(_virtualImg._modelViewMatrix);
		glm::mat4x4 inverseVirtualProjectionMatrix = glm::inverse(_virtualImg._projMatrix);
		
		//glm::vec4 dir = inverseVirtualModelViewMatrix * inverseVirtualProjectionMatrix * glm::vec4(rangeX, -rangeY, 0.0f, 0.0f);
		glm::vec4 dir = inverseVirtualModelViewMatrix * glm::vec4(rangeX, -rangeY, 0.0f, 0.0f) * glm::distance(_virtualImg._glmC, _objCenterPos);
		//glm::vec4 dir = inverseVirtualModelViewMatrix * glm::vec4(rangeX, -rangeY, 1.0f, 0.0f);
		//dir = dir * 100.0f;
		glm::vec3 normalizedDir = glm::vec3(dir.x, dir.y, dir.z);

		//_virtualImg._glmC = _virtualImg._glmC + glm::vec3(dir.x, dir.y, dir.z);
		if(_weightOfView <= 0.0f)
		{
			_virtualImg._camIndex = _nearestCamIndex;
			_weightOfView = 1.0f;
		}
		if(_weightOfView >= 1.0f)
		{
			_weightOfView = 1.0f;
			// then based on the angle of dir and camera center vector to determine which camera to use
			
			if(_virtualImg._camIndex == 0 )
				_nearestCamIndex = 1;
			else if(_virtualImg._camIndex == _psParam._numOfCameras -1)
				_nearestCamIndex = _virtualImg._camIndex - 1;
			else 
			{
				int nearestCamIndex1 = _virtualImg._camIndex + 1;
				int nearestCamIndex2 = _virtualImg._camIndex - 1;
				if( glm::dot( (**_allIms)[nearestCamIndex1]._glmC - (**_allIms)[_virtualImg._camIndex]._glmC, normalizedDir) > 0 )
					_nearestCamIndex = nearestCamIndex1;
				else
					_nearestCamIndex = nearestCamIndex2;
			}
			// calculate the nearest left, and nearest right camera
		}
		if( glm::dot( (**_allIms)[_nearestCamIndex]._glmC - (**_allIms)[_virtualImg._camIndex]._glmC, normalizedDir) > 0 )
		{
			_weightOfView -= 0.1f;
			_weightOfView = (_weightOfView < 0)? 0.0f : _weightOfView;
		}
		else
		{
			_weightOfView += 0.1f;
			_weightOfView = (_weightOfView > 1)? 1.0f : _weightOfView;
		}
		// _weightOfView and the cameras used for interpolation is ready, then do the interpolation. Use dir to update _weightOfView
		_virtualImg._glmC = glm::mix((**_allIms)[_nearestCamIndex]._glmC, (**_allIms)[_virtualImg._camIndex]._glmC, _weightOfView );

		//	quat_cast (detail::tmat3x3< T > const &x)
		glm::quat qt = 	glm::mix(glm::quat_cast((**_allIms)[_nearestCamIndex]._glmR), glm::quat_cast((**_allIms)[_virtualImg._camIndex]._glmR) , _weightOfView);
		_virtualImg._glmR = glm::mat3_cast(qt);
 	//Returns a SLERP interpolated quaternion of x and y according a. 

		_virtualImg.setModelViewMatrix();
		_virtualImg.setProjMatrix();
		_virtualImg.calcPlaneCoords();
		emit updateVirtualView_signal(_virtualImg);

	}
	_mouseX = event->x();
	_mouseY = event->y();

	updateGL();
}

void GLWidgetVirtualView::writeGeometryShaderFile( std::string fileName)
{
	std::ofstream inF(fileName);
	std::stringstream ss; ss<<_psParam._numOfCameras;

	std::string s;
	s += "#version 420\n";
	s += "layout(points) in;\n";
	s += "layout(triangle_strip, max_vertices = 4) out;\n";
	s += "uniform mat4x4 transformMatrix[" + ss.str() + "];\n";
	s += "uniform float step;\n";
	s += "in int instanceID[];\n";
	s += "out vec4 ProjTexCoord[" + ss.str() + "];\n";
	s += "void main() {";
	s += "float depth = -1.0f + step * float( instanceID[0] + 1);	float length  = 1.0f;\n";
	//
	s += "gl_Position = gl_in[0].gl_Position + vec4( -length, -length, depth , 0.0f);\n";
	s += "for(int i = 0; i< transformMatrix.length; i++) {ProjTexCoord[i] = transformMatrix[i] * gl_Position;}\n";
	s += "gl_Layer = instanceID[0];\n EmitVertex();\n";
	//
	s += "gl_Position = gl_in[0].gl_Position + vec4( -length, length, depth , 0.0f);\n";
	s += "for(int i = 0; i< transformMatrix.length; i++) {ProjTexCoord[i] = transformMatrix[i] * gl_Position;}\n";
	s += "gl_Layer = instanceID[0];\n EmitVertex();\n";
	//
	s += "gl_Position = gl_in[0].gl_Position + vec4( length, -length, depth , 0.0f);\n";
	s += "for(int i = 0; i< transformMatrix.length; i++) {ProjTexCoord[i] = transformMatrix[i] * gl_Position;}\n";
	s += "gl_Layer = instanceID[0];\n EmitVertex();\n";
	//
	s += "gl_Position = gl_in[0].gl_Position + vec4( length, length, depth , 0.0f);\n";
	s += "for(int i = 0; i< transformMatrix.length; i++) {ProjTexCoord[i] = transformMatrix[i] * gl_Position;}\n";
	s += "gl_Layer = instanceID[0];\n EmitVertex();\n";
	//
	s += "EndPrimitive();}";
	inF << s << std::endl;
	inF.close();
}

void GLWidgetVirtualView::writeFragmentShaderFile(std::string fileName)
{
	std::ofstream inF(fileName);
	std::stringstream ss; //ss<<_psParam._numOfCameras;

	std::string s;
//-----
	ss<< _psParam._numOfCameras;
	s += "#version 420\nin vec4 ProjTexCoord[" + ss.str() +"];\n";
	for( int i = 0; i<_psParam._numOfCameras; i++)
	{
		ss.str(std::string());
		ss<<i;
		s += "uniform sampler2D tex" + ss.str() + ";\n";
	}
	s += "layout(location = 0) out float meanCost;\n layout(location = 1) out vec4 meanColor;\n";
	s += "void main()\n{\n";
	ss.str(std::string());
	ss << _psParam._numOfCameras;
	s += "vec4 projTexColor["+ss.str() +"];\nbool t[" + ss.str() + "];\n";
	s += "for(int i = 0; i<" + ss.str() +"; i++)\n{t[i] = false;}\n";
	s += "vec4 baseColor = vec4(0,0,0,0);\nfloat numOfViews = 0;\n";
	for( int i = 0; i<_psParam._numOfCameras; i++)
	{
		ss.str(std::string());
		ss<<i;
		s += "if(ProjTexCoord["+ ss.str() + "].x/ProjTexCoord["+ss.str()+"].w > 0 && ProjTexCoord["+ss.str()+"].x/ProjTexCoord["+ss.str()+"].w <1.0 && ProjTexCoord["+ss.str()+"].y/ProjTexCoord["+ss.str()+"].w > 0 && ProjTexCoord["+ss.str()+"].y/ProjTexCoord["+ss.str()+"].w < 1.0 && ProjTexCoord["+ss.str()+"].z > 0.0f)\n";
		s += "{\n projTexColor["+ ss.str() + "] = textureProj(tex"+ ss.str() + ", ProjTexCoord["+ ss.str() + "]);\n";
		s += "baseColor = baseColor + projTexColor["+ ss.str() + "];\nnumOfViews = numOfViews + 1.0;\nt["+ ss.str() + "] = true;\n}\n";
	}
	s += "meanColor = vec4(0,0,0,1.0f);\nif(numOfViews <=1){\nmeanCost = 1000000.0f;\n}\nelse\n{\n";
	s += "baseColor = baseColor/numOfViews;\nmeanCost = 0.0f;\n";

	s += "for(int i = 0; i<projTexColor.length(); i++)\n{\nif(t[i] == true){\n";
	s += "meanCost = meanCost +  float(pow(projTexColor[i].x - baseColor.x, 2)) + float(pow(projTexColor[i].y - baseColor.y, 2)) + float(pow(projTexColor[i].z - baseColor.z, 2));\n";
	s += "meanColor.xyz = meanColor.xyz + projTexColor[i].xyz;\n}\n}\nmeanColor.xyz = meanColor.xyz/numOfViews;\n}\n}\n";
//-----
	inF << s << std::endl;
	inF.close();
}