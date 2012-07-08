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
	_virtualViewForm(NULL), _widgetForContext(NULL)
{
	ui.setupUi(this);	

	QObject::connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openFile_slot()));	
	QObject::connect(ui.actionStart_Capture, SIGNAL(triggered()), this, SLOT(startCapture_slot()));
	QObject::connect(ui.actionView_Synthesis, SIGNAL(triggered()), this, SLOT(startViewSynthesis_slot()));
	
	_timer.setInterval(30);
	QObject::connect(&_timer, SIGNAL(timeout()), this, SLOT(retrieveImages()), Qt::QueuedConnection);  

	_allImages = new std::vector<image>;
	_allImagesBackBuffer = new std::vector<image>;

	QGLFormat format;
	format.setVersion(4,2);
	//format.setProfile(QGLFormat::CoreProfile);
	_widgetForContext = new QGLWidget(format);
	
}

void mainWindowForm::startViewSynthesis_slot()
{
	if( _virtualViewForm == NULL)
		_virtualViewForm = new GLWidgetVirtualView( &_allImages, _widgetForContext, _imagesForm->_glWidgets);
	else 
		return;

	// this is the center of rotation. Maybe removed in the future version *****
	//_virtualViewForm->setObjCenterPos(_allImagesForm->_objCenterPos);
	_virtualViewForm->setObjCenterPos(glm::vec3(0,0,0));
	//--------------------------------------------------------------------
	QMdiSubWindow *subWindow2 = new QMdiSubWindow();	
	QObject::connect( _virtualViewForm, SIGNAL(updateVirtualView_signal(virtualImage)), _allImagesForm, SLOT(updateVirtualView_slot(virtualImage)));


	std::cout<< "width: " << _virtualViewForm->geometry().width() << std::endl;
	std::cout<< "height: " << _virtualViewForm->geometry().height() << std::endl;

	subWindow2->setGeometry(_virtualViewForm->geometry());
    subWindow2->setWidget(_virtualViewForm);
	subWindow2->setAttribute(Qt::WA_DeleteOnClose);
    ui.mdiArea->addSubWindow(subWindow2);
	subWindow2->show();
	//subWindow2->adjustSize();


}

mainWindowForm::~mainWindowForm(void)
{
	if(_busHandler != NULL)
		delete _busHandler;
	if(_widgetForContext != NULL)
		delete _widgetForContext;

	delete _allImages;
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
		_timer.stop();
	}
	if(_busHandler == NULL)
	{			
		_busHandler = new allImageCaptureManager(_allImagesBackBuffer); 
	}
	if(_busHandler->returnNumberOfCams() == 0)
	{
		delete _busHandler;
		_busHandler = NULL;
		return;
	}
	// capture
	retrieveImages();
	_timer.start();
}

void mainWindowForm::retrieveImages()
{
	if(_busHandler == NULL || _busHandler->returnNumberOfCams() ==0)
		 return;
	//capture

	{
		//QMutexLocker qml(_imageMutex );
		_allImages = _busHandler->retrieveImgsAllParallel(_allImages);
	}

	if(_imagesForm == NULL)
	{
		_imagesForm = new viewSynForm(&_allImages, _widgetForContext/*, _imageMutex*/);
		showImageWindow();
		_wasCapturing = true;
	}
	else
	{
		_imagesForm->setUpImages(_allImages, _widgetForContext);		
		QObject::connect(this, SIGNAL(redrawCameraPoses()), _allImagesForm, SLOT(updateGL()), Qt::UniqueConnection);
		if(!_wasCapturing)
		{
			_allImagesForm->upDateParam();
			_wasCapturing = true;
			emit redrawCameraPoses();	
		}		
		for(int i = 0; i< _imagesForm->_glWidgets.size(); i++)
		{
			QObject::connect(this, SIGNAL(redrawImages()), _imagesForm->_glWidgets[i] , SLOT(updateGL()), Qt::UniqueConnection);
		}
		emit redrawImages();
	}
	//QObject::connect(this, SIGNAL(retrieveOneShot()), this , SLOT(retrieveImages()), Qt::UniqueConnection);	
	//emit retrieveOneShot();
}

void mainWindowForm::openFile_slot()
{	
	QString qFileName = QFileDialog::getOpenFileName(this,
     tr("Open Image"), "C:\\Enliang\\data\\middleBury\\temple", tr("Image List Files (*.txt)"));
	std::string fileName = qFileName.toLocal8Bit().constData();
	std::cout << fileName << std::endl;
	if(fileName.empty())
		return;
	//-------------
	readImages(fileName);

	_wasCapturing = false;
	// show the new window
	if(_imagesForm == NULL)
	{	
		_imagesForm = new viewSynForm(&_allImages, _widgetForContext/*, _imageMutex*/);	
		showImageWindow();
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

	_allImagesForm = new GLWidgetAllImgs(&_allImages, _widgetForContext, _imagesForm->_glWidgets);	
	QMdiSubWindow *subWindow1 = new QMdiSubWindow();	 
    subWindow1->setWidget(_allImagesForm);
	subWindow1->setGeometry(100,100, 400,400);
	subWindow1->setAttribute(Qt::WA_DeleteOnClose);
    ui.mdiArea->addSubWindow(subWindow1);
	subWindow1->show();
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
		{ std::cout<< "images " << im._imageName <<" cannot be found" << std::endl; return;}
		_allImages->push_back(im);
	 }
	 in.close();
}

