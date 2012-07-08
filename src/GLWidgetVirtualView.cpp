#include "GLWidgetVirtualView.h"
#include <opencv\cxcore.h>
#include <QMouseEvent>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>

#define printOpenGLError() printOglError(__FILE__, __LINE__)

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
	_allIms(allIms), QGLWidget((QWidget*)NULL, sharedWidget), _virtualImg((**allIms)[0]),
		_mouseX(0), _mouseY(0), _imageQGLWidgets(imageQGLWidgets), _cost3DTexID(0), _fbo(NULL), _psVertexBufferHandle(0),
		_psVertexArrayObjectHandle(0)
{
	int width, height;
	if( (*allIms)->size() <1){	
		width = 200, height = 100;} // set a predefined size if there is no image
	else{
		width = (**allIms)[0]._image.cols; 
		height = (**allIms)[0]._image.rows;
	}
	this->setGeometry(0,0, width, height);
	_psParam._virtualHeight = (**allIms)[0]._image.rows; 
	_psParam._virtualWidth = (**allIms)[0]._image.cols; 
	_psParam._numOfPlanes = 25;
	_psParam._numOfCameras  = 3;

}

void GLWidgetVirtualView::initTexture3D(GLuint & RTT3D, int imageWidth, int imageHeight, int numOfLayers)
{
		glGenTextures(1, &RTT3D);
    glBindTexture(GL_TEXTURE_3D, RTT3D);
    // set basic parameters
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, imageWidth, imageHeight, numOfLayers, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

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

void GLWidgetVirtualView::initializeGL()
{
	glewInit();	// Initialize glew
	//--------------------------------------------------------
	// set up shader
	std::string filePath = std::string(std::getenv("SHADER_FILE_PATH"));

	_shaderHandle.init();
	_shaderHandle.loadShader(VSShaderLib::VERTEX_SHADER, (filePath + "\\warping.vert").c_str());
	std::cout<<"vertex shader: " << _shaderHandle.getShaderInfoLog(VSShaderLib::VERTEX_SHADER)<<std::endl;
	_shaderHandle.loadShader(VSShaderLib::GEOMETRY_SHADER, (filePath + "\\warping.geom").c_str());
	std::cout<<"geometry shader: " << _shaderHandle.getShaderInfoLog(VSShaderLib::GEOMETRY_SHADER)<<std::endl;
	_shaderHandle.loadShader(VSShaderLib::FRAGMENT_SHADER, (filePath + "\\warping.frag").c_str());
	std::cout<<"fragment shader: " << _shaderHandle.getShaderInfoLog(VSShaderLib::FRAGMENT_SHADER)<< std::endl;
	_shaderHandle.prepareProgram();
	// set up 3d texture that I can render to (number of layers should be set )
	printOpenGLError();
	initTexture3D( _cost3DTexID, _psParam._virtualWidth, _psParam._virtualHeight, _psParam._numOfPlanes);
	//--------------------------------------------------------
	// set up vbo
	float vertices[3] = {0.0f, 0.0f, 0.0f};
	initializeVBO_VAO(vertices, 1, _psVertexBufferHandle, _psVertexArrayObjectHandle);	// here 1 is the number of primitives
	// set up fbo?
	_fbo = new FramebufferObject();
	_fbo->Bind();
	_fbo->AttachTexture(GL_TEXTURE_3D, _cost3DTexID, GL_COLOR_ATTACHMENT0, 0, -1); // -1 means no specific layer is specified, 0 is the mipmap level
	_fbo->Disable();
	//------------------------
	glClearColor(1.0, 0.0, 1.0, 0);
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

	printOpenGLError();

	// register the 3d texture so that CUDA can use it

	//CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&cuda_tex_array_resource, projected3DTex, 
	//			  GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone));
	CUDA_SAFE_CALL(cudaGLSetGLDevice(0));

	CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&cost3D_CUDAResource, _cost3DTexID, 
				  GL_TEXTURE_3D, cudaGraphicsRegisterFlagsNone));// register the 3d texture
	 
}

void GLWidgetVirtualView::CUDA_SAFE_CALL( cudaError_t err, std::string file, int line)
{
   if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
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

void GLWidgetVirtualView::displayLayedTexture(GLuint &texture)
{
	glClearColor(0.75 ,0.0,0,1);
	glClear(GL_COLOR_BUFFER_BIT); 
	glUseProgram(_shaderHandleDisplayLayerTexture.getProgramIndex());
	
	int textureUint = 0;
	glActiveTexture(GL_TEXTURE0 + textureUint);
	glBindTexture(GL_TEXTURE_3D, texture);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);	
	//glEnable(GL_TEXTURE_3D);
	_shaderHandleDisplayLayerTexture.setUniform("numOfLayers", _psParam._numOfPlanes);
	
	_shaderHandleDisplayLayerTexture.setUniform("texs",&textureUint);
	printOpenGLError();	
	
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
}


GLWidgetVirtualView::~GLWidgetVirtualView()
{
	if(_fbo != NULL)
		delete _fbo;
}

void GLWidgetVirtualView::resizeGL(int w, int h)
{
	glViewport(0, 0, w, h);
}

void GLWidgetVirtualView::paintGL()
{
	// ***** maybe no depth buffer is needed in first pass

	//glClear(GL_COLOR_BUFFER_BIT); 
	
	printOpenGLError();
	//---------------------
	// set up the uniforms: images, transformation matrix, etc...
	glUseProgram(_shaderHandle.getProgramIndex());
	float step = 2.0f / static_cast<float>(_psParam._numOfPlanes + 1);
	_shaderHandle.setUniform("step", &step);
	printOpenGLError();
	// matrix:
	glm::mat4 projScaleTrans = glm::translate(glm::vec3(0.5f)) * glm::scale(glm::vec3(0.5f));
	glm::mat4 virtInverseModelViewProj = glm::inverse(_virtualImg._modelViewMatrix) 
		* glm::inverse(_virtualImg._projMatrix);
	
	// *****
	int numOfImages = 3;
	glm::mat4 *modelViewProj = new glm::mat4[numOfImages];
	for(int i = 0; i < numOfImages; i++)
	{
		modelViewProj[i] = (**_allIms)[i]._projMatrix * (**_allIms)[i]._modelViewMatrix;
		glm::mat4x4 transformMatrix = projScaleTrans * modelViewProj[i] * virtInverseModelViewProj;
		std::stringstream ss; ss<<i;
		std::string uniformVarName = "transformMatrix" + ss.str();
		_shaderHandle.setUniform( uniformVarName.c_str(), &transformMatrix[0][0]);
		printOpenGLError();
		// images:
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, _imageQGLWidgets[i]->_tex._textureID);
		_shaderHandle.setUniform(("tex" + ss.str()).c_str(), &i);
		printOpenGLError();
	}
	printOpenGLError();
	
	// bind to fbo
	_fbo->Bind();	
	_fbo->IsValid(std::cout);

	glClear(GL_COLOR_BUFFER_BIT); 
	// draw() using vao
	glBindVertexArray(_psVertexArrayObjectHandle);
	printOpenGLError();
	glDrawArraysInstanced(GL_POINTS, 0, 1, _psParam._numOfPlanes); 

	// unbind fbo, vao
	glBindVertexArray(0);
	_fbo->Disable();
	//*****
	
	displayLayedTexture(_cost3DTexID);

	// --------------------------------- By doing this I get the layered texture 
	//  map the texture to CUDA, and then find the colors, and then render by writing a cuda kernel
	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &cost3D_CUDAResource, 0));	// one resource and stream 0
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&cost3D_CUDAArray, cost3D_CUDAResource, 0, 0));	// 0th layer, 0 mipmap level




	// that's it!!!
	printOpenGLError();
}

void GLWidgetVirtualView::mousePressEvent(QMouseEvent *event)
{
	_mouseX = event->x();
	_mouseY = event->y();
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
		_virtualImg._glmC = _virtualImg._glmC + glm::vec3(dir.x, dir.y, dir.z);
		_virtualImg.setModelViewMatrix();
		_virtualImg.setProjMatrix();
		_virtualImg.calcPlaneCoords();
		emit updateVirtualView_signal(_virtualImg);

	}
	_mouseX = event->x();
	_mouseY = event->y();

	updateGL();
}