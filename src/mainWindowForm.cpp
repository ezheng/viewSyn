#include "mainWindowForm.h"
#include <iostream>
#include <string>
#include "image.h"
#include <fstream>
#include <qfiledialog.h>
#include <QMdiSubWindow>

#include <QtOpenGL/QGLWidget>
#include "GLWidgetAllImgs.h"

#include <qthread.h>

mainWindowForm::mainWindowForm(void): _busHandler(NULL), _imagesForm(NULL), _allImagesForm(NULL), _wasCapturing(false),
	_virtualViewForm(NULL), _widgetForContext(NULL), _xyzMin(0,0,0), _xyzMax(0,0,0)
{
	ui.setupUi(this);	

	QObject::connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openFile_slot()));	
	QObject::connect(ui.actionStart_Capture, SIGNAL(triggered()), this, SLOT(startCapture_slot()));
	QObject::connect(ui.actionView_Synthesis, SIGNAL(triggered()), this, SLOT(startViewSynthesis_slot()));
	QObject::connect(ui.actionLoad_Cam_Param, SIGNAL(triggered()), this, SLOT(loadCamParam_slot()), Qt::QueuedConnection);

	_allImages = new std::vector<image>;
	_allImagesBackBuffer = new std::vector<image>;


	QGLFormat format;
	format.setVersion(4,0);
	_widgetForContext = new QGLWidget(format); 
	
}

void mainWindowForm::startViewSynthesis_slot()
{
	if( _virtualViewForm == NULL)
	{
		_virtualViewForm = new GLWidgetVirtualView( &_allImages, _widgetForContext, _imagesForm->_glWidgets);
		QObject::connect(ui.actionPrint_Error, SIGNAL(triggered()), _virtualViewForm, SLOT(computeImageError()), Qt::UniqueConnection);
	}
	else 
		return;

	// this is the center of rotation. Maybe removed in the future version *****
	//_virtualViewForm->setObjCenterPos(_allImagesForm->_objCenterPos);
	//_virtualViewForm->setObjCenterPos(glm::vec3(0,0,0));
	_virtualViewForm->setObjCenterPos((_xyzMax + _xyzMin)/2.0f);
	//--------------------------------------------------------------------
	QMdiSubWindow *subWindow2 = new QMdiSubWindow();	
	QObject::connect( _virtualViewForm, SIGNAL(updateVirtualView_signal(virtualImage)), _allImagesForm, SLOT(updateVirtualView_slot(virtualImage)));
	//
	QObject::connect(this, SIGNAL(redrawImages()), _virtualViewForm, SLOT(updateGL()), Qt::UniqueConnection);
	

	std::cout<< "width: " << _virtualViewForm->geometry().width() << std::endl;
	std::cout<< "height: " << _virtualViewForm->geometry().height() << std::endl;

	subWindow2->setGeometry(_virtualViewForm->geometry());
    subWindow2->setWidget(_virtualViewForm);
	subWindow2->setAttribute(Qt::WA_DeleteOnClose);
    ui.mdiArea->addSubWindow(subWindow2);
	subWindow2->show();
	//subWindow2->adjustSize();


	QObject::connect(_imagesForm->ui.doubleSpinBox_farPlane, SIGNAL(valueChanged(double)), _virtualViewForm, SLOT(psFarPlaneChanged(double)), Qt::UniqueConnection); 
	QObject::connect(_imagesForm->ui.doubleSpinBox_nearPlane, SIGNAL(valueChanged(double)), _virtualViewForm, SLOT(psNearPlaneChanged(double)), Qt::UniqueConnection); 
	QObject::connect(_imagesForm->ui.doubleSpinBox_planeNum, SIGNAL(valueChanged(double)), _virtualViewForm, SLOT(psNumPlaneChanged(double)), Qt::UniqueConnection); 
	QObject::connect(_imagesForm->ui.doubleSpinBox_sigma, SIGNAL(valueChanged(double)), _virtualViewForm, SLOT(psGSSigmaChanged(double)), Qt::UniqueConnection);

	QObject::connect( _allImagesForm, SIGNAL(newPosKinect_SIGNAL(float, float, bool)), _virtualViewForm, SLOT(newPosKinect_SLOT(float, float, bool)), Qt::UniqueConnection);

	_virtualViewForm->psFarPlaneChanged(_imagesForm->ui.doubleSpinBox_farPlane->value());
	_virtualViewForm->psNearPlaneChanged(_imagesForm->ui.doubleSpinBox_nearPlane->value());
	_virtualViewForm->psNumPlaneChanged(_imagesForm->ui.doubleSpinBox_planeNum->value());
	_virtualViewForm->psGSSigmaChanged(_imagesForm->ui.doubleSpinBox_sigma->value());


}

mainWindowForm::~mainWindowForm(void)
{
	if(_busHandler != NULL)
		delete _busHandler;
	if(_widgetForContext != NULL)
		delete _widgetForContext;

	if(_allImages != NULL)
		delete _allImages;
	if(_allImagesBackBuffer != NULL)
		delete _allImagesBackBuffer;
	//if(_virtualViewForm != NULL)
	//	delete _virtualViewForm;
}

void mainWindowForm::startCapture_slot()
{		
	// call the bus to capture and copy the images	
	if(_busHandler != NULL)
	{
		delete _busHandler;
		_busHandler = NULL;
		//_timer.stop();
	}
	if(_busHandler == NULL)
	{
		// read the calibration data
		

		_busHandler = new allImageCaptureManager(_allImages); 
		_allImagesBackBuffer->resize( _busHandler->returnNumberOfCams());
				
		_busHandler->moveToThread(&_CamCaptureThread);
		QObject::connect( this, SIGNAL(retrieveImgsAllParallel_SIGNAL()), _busHandler, SLOT(retrieveImgsAllParallel_SLOTS()));
		QObject::connect( _busHandler, SIGNAL(imageReady_SIGNAL()), this, SLOT(retrieveImages()));
		_CamCaptureThread.start();

		// triggering the capture process
		emit retrieveImgsAllParallel_SIGNAL();
		
	} 

	if(_busHandler->returnNumberOfCams() == 0)
	{
		delete _busHandler;
		_busHandler = NULL;
		return;
	}
	
}


// this function is executed when the cameraReady is signalled.
void mainWindowForm::retrieveImages()
{
	if(_busHandler == NULL || _busHandler->returnNumberOfCams() ==0)
		 return;
	
	_busHandler->swapBuffer(&_allImages);
	emit retrieveImgsAllParallel_SIGNAL();	// after obtaining the new image(by swapping the buffer), tell the camera to capture image

	//---------------------------------------------------------------------
	if(_imagesForm == NULL)
	{
		_imagesForm = new viewSynForm(&_allImages, _widgetForContext/*, _imageMutex*/);
		showImageWindow();
		//_wasCapturing = true;
	}
	else
	{
		static bool init = true;
		if(init)
		{
			//QObject::connect(this, SIGNAL(redrawCameraPoses()), _allImagesForm, SLOT(updateGL()), Qt::UniqueConnection);
			
			for(int i = 0; i< _imagesForm->_glWidgets.size(); i++)
			{
				QObject::connect(this, SIGNAL(redrawImages()), _imagesForm->_glWidgets[i] , SLOT(updateGL()), Qt::UniqueConnection);
			}
			init = false;
		}
		
		_imagesForm->setUpImages(_allImages, _widgetForContext);		
		
		//if(!_wasCapturing)
		//{
		//	_allImagesForm->upDateParam();
		//	_wasCapturing = true;
		//emit redrawCameraPoses();	
		//}	

		_allImagesForm->updateGL();
		
		emit redrawImages();
	}
	
}

void mainWindowForm::loadCamParam_slot()
{
	QString qFileName = QFileDialog::getOpenFileName(this,
     //tr("Open Image"), "C:\\Enliang\\data\\middleBury\\temple", tr("Image List Files (*.txt)"));
	  tr("Open Image"), "C:\\Enliang\\data\\fountain_dense\\fromLiang\\images\\quarter_size", tr("Image List Files (*.txt)"));
	std::string fileName = qFileName.toLocal8Bit().constData();

	if(_allImages == _allImagesBackBuffer)
	{
		std::cout<< "buffer error" << std::endl;
	}
	//std::cout<< "_allImages: " << _allImages << std::endl;
	//std::cout<< "_allImagesBackBuffer: " << _allImagesBackBuffer << std::endl;
		
	readCalibrationData(fileName, _allImages, _allImagesBackBuffer);

	_allImagesForm->upDateParam();


}

void mainWindowForm::openFile_slot()
{	
	QString qFileName = QFileDialog::getOpenFileName(this,
     //tr("Open Image"), "C:\\Enliang\\data\\middleBury\\temple", tr("Image List Files (*.txt)"));
	  tr("Open Image"), "C:\\Enliang\\data\\fountain_dense\\fromLiang\\images\\quarter_size", tr("Image List Files (*.txt)"));
	
	std::string fileName = qFileName.toLocal8Bit().constData();
	std::cout << fileName << std::endl;
	if(fileName.empty())
		return;
	//------------------
	readImages(fileName);

	//_wasCapturing = false;
	// show the new window
	if(_imagesForm == NULL)
	{	
		_imagesForm = new viewSynForm(&_allImages, _widgetForContext/*, _imageMutex*/);	
		showImageWindow();
		// connects the widgets for setting up plane-sweeping param


	}
	else
	{
		_imagesForm->setUpImages(_allImages, _widgetForContext);	
		// set up signal and redraw
		QObject::connect(this, SIGNAL(redrawCameraPoses()), _allImagesForm, SLOT(updateGL()), Qt::UniqueConnection);
		for(int i = 0; i< _imagesForm->_glWidgets.size(); i++)
		{
			QObject::connect(this, SIGNAL(redrawImages()), _imagesForm->_glWidgets[i] , SLOT(updateGL()), Qt::UniqueConnection);
		}
		_allImagesForm->upDateParam();
		emit redrawCameraPoses();
		emit redrawImages();
	}	
}

void mainWindowForm::showImageWindow()
{
	QMdiSubWindow *subWindow = new QMdiSubWindow();	 
	subWindow->setGeometry(_imagesForm->geometry());
    subWindow->setWidget(_imagesForm);
	subWindow->setAttribute(Qt::WA_DeleteOnClose);
    ui.mdiArea->addSubWindow(subWindow);
	subWindow->show();

	_allImagesForm = new GLWidgetAllImgs(&_allImages, _widgetForContext, _imagesForm->_glWidgets, _xyzMin, _xyzMax);	
	QMdiSubWindow *subWindow1 = new QMdiSubWindow();	 
    subWindow1->setWidget(_allImagesForm);
	subWindow1->setGeometry(100,100, 400,400);
	subWindow1->setAttribute(Qt::WA_DeleteOnClose);
    ui.mdiArea->addSubWindow(subWindow1);
	subWindow1->show();

//	subWindow1->moveToThread(&_allImageThread);
//	_allImageThread.start();


}

void mainWindowForm::readCalibrationData(std::string fileName, std::vector<image> * allImages, std::vector<image>* allImagesBackBuffer)
{
	 std::ifstream in( fileName);
	 if(!in.is_open())
	 {
		return;
	 }
	 unsigned int numOfImages; 
	 in>>numOfImages;
	 //if(numOfImages != _allImages
	 if(numOfImages > _allImages->size())
	 {std::cout<< "number of images does not match. There might be some errors" << std::endl; return;}

	 //_allImages->clear();
	 for(unsigned int i = 0; i < numOfImages; i++)
	 {
		std::string imageName;
		in>>imageName;
		double K[9], R[9], T[9];
		for(int j = 0; j < 9; j++)
			in>>K[j];		
		for(int j = 0; j < 9; j++)
			in>>R[j];
		for(int j = 0; j < 3; j++)
			in>>T[j];

		// update the K, R, T of im


		(*allImages)[i].updateCamParam(K, R, T);
		(*allImagesBackBuffer)[i].updateCamParam(K,R,T);

	
		//image im(imageName, K, R, T);				
		//if(im._image.empty())
		//{ std::cout<< "images " << im._imageName <<" cannot be found" << std::endl; }
		//else
		//_allImages->push_back(im);
	 }
	 //
	 int numOfImages_2; in >> numOfImages_2;
	  
	 
	 if(numOfImages_2 == numOfImages)
	 {
		for(unsigned int i =0; i < numOfImages; i++)
		{
			double kc[5];
			for(int j = 0; j<5; j++)
				in>>kc[j];
			(*allImages)[i].setupDistortionParam(kc);
			(*allImagesBackBuffer)[i].setupDistortionParam(kc);
		}
	 }
	 // load the color calibration table 
	 int numOfImages_3; in >> numOfImages_3;
	 if(numOfImages_3 == numOfImages)
	 {
		for(unsigned int i = 0; i < numOfImages; i++)
		{
			for(unsigned int j = 0; j<256; j++)
			{
				in >> (*allImages)[i]._LUT[j];	
			}
		}
	 }
	 in.close();
}

void mainWindowForm::readImages(std::string fileName)
{
	 std::ifstream in( fileName);
	 if(!in.is_open())
	 {
		 std::cout<< "Files cannot open" << std::endl; return;
	 }
	 int numOfImages; 
	 in>>numOfImages;

	 if(numOfImages>8)
	 {
		 std::cout<< "Warning: too many images, only 8 images are loaded"<< std::endl;
		numOfImages = 8;
	 }
	 _allImages->clear();
	 for(int i = 0; i < numOfImages; i++)
	 {
		std::string imageName;
		in>>imageName;
		double K[9], R[9], T[9];
		for(int i = 0; i < 9; i++)
			in>>K[i];		
		for(int i = 0; i < 9; i++)
			in>>R[i];
		for(int i = 0; i < 3; i++)
			in>>T[i];
		//
		image im(imageName, K, R, T);				
		if(im._image.empty())
		{ std::cout<< "images " << im._imageName <<" cannot be found" << std::endl; }
		else
		_allImages->push_back(im);
	 }
	 in.close();

	 // read the bounding box for the images
	 //std::string filePath = 
	 size_t pos= fileName.find_last_of('/');
	 if( pos != std::string::npos)
	 {
		std::string boundingFileName = fileName.substr(0, pos+1);
		boundingFileName += "bound.txt";	
		
		in.open(boundingFileName);

		 if(!in.is_open())
		 {
			std::cout<< "object bounding data is not available" << std::endl; return;
		 }
		// float x; 
		// in >> x;
		 in >> _xyzMin[0] >> _xyzMin[1] >> _xyzMin[2] >> _xyzMax[0] >> _xyzMax[1] >> _xyzMax[2];
		in.close();

	 }

}

