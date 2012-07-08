#include "viewSynForm.h"
#include <QRegExp>

#include <iostream>
#include <qfiledialog.h>
#include <sstream>
#include <QTime>

#define NUM_GLWIDGETS 8

viewSynForm::viewSynForm(std::vector<image>** ims, QGLWidget* widgetForContext/*, QMutex *imageMutex */):_imageSavingPath(".\\")//,_imageMutex(imageMutex)
{	
	 _ims = ims;
	
	ui.setupUi(this); 
	_fixedWidth = 240;	
	double aspectRatio = static_cast<double>( (**ims)[0]._image.rows)/static_cast<double>((**ims)[0]._image.cols);
	
	((QWidget*)(ui.gridLayout_Images->parent()))->setGeometry(0,0, _fixedWidth * 4.5, static_cast<int>(_fixedWidth * 2.5 * aspectRatio));
	
	this->setGeometry(((QWidget*)(ui.gridLayout_Images->parent()))->geometry());

	int numOfImages = (*ims)->size();		

	for(int i = 0; i < NUM_GLWIDGETS; i++){
		int j = (i>=numOfImages)? (numOfImages-1):i; 		
		GLWidget *glw = new GLWidget(&((**ims)[j]), i, widgetForContext);
		glw->setFixedHeight(_fixedWidth * aspectRatio);
		glw->setFixedWidth(_fixedWidth);
		if(j!=i)
			glw->hide();
		_glWidgets.append(glw);		
	}		
	for(int i = 0 ; i<NUM_GLWIDGETS; i++){
		int row = i/4;
		int col = i - row * 4;
		ui.gridLayout_Images->addWidget(_glWidgets[i], row, col);		
	}		
	for(int i = 0; i<NUM_GLWIDGETS; i++)
		for(int j = 0; j<NUM_GLWIDGETS; j++){
			if(i != j)
				QObject::connect(_glWidgets[i], SIGNAL(posChanged(int, int, image)),_glWidgets[j], SLOT(setPosValue(int, int, image)));
		}
	ui.button_Capture->setEnabled(false);
	QObject::connect(ui.button_Saving_Path, SIGNAL(clicked()), this, SLOT(getSavingPath()), Qt::QueuedConnection);
	QObject::connect(ui.button_Capture, SIGNAL(clicked()), this, SLOT(saveImage()));
}

void viewSynForm::saveImage()
{	
	//string s = ss.str();
	int numOfImages = (*_ims)->size();

	QTime t = QTime::currentTime();
	QString fileName = t.toString( QString("HH_mm_ss"));
	
		//QMutexLocker qml(_imageMutex );

	for(int i = 0; i<numOfImages; i++)
	{
		std::stringstream ss; 
		ss<<i;
		std::string fullFilePath = _imageSavingPath+ "\\" + ss.str();
		if(!QDir(fullFilePath.c_str()).exists())
		{
			QDir().mkdir(fullFilePath.c_str());
		}

		std::string fullFileName = fullFilePath + "\\" + "camera" + ss.str() + "_" + fileName.toLocal8Bit().constData() + ".jpg";

		if((**_ims)[i]._image.rows != 600)
			std::cout<<"empty images" << std::endl;

		cv::imwrite(fullFileName, (**_ims)[i]._image);
	}
}

void viewSynForm::getSavingPath()
{
	QString s = QFileDialog::getExistingDirectory(this, "select folder", "C:/Enliang/data/own");
	if(!s.isEmpty())
	{
		_imageSavingPath = s.toLocal8Bit().constData();
		ui.button_Capture->setEnabled(true);	
	}
}

void viewSynForm::setUpImages(std::vector<image>* ims, QGLWidget* widgetForContext)
{
	// set images for widgets
	int numOfImages = ims->size();
	int numOfWidgets = _glWidgets.size();

	if(numOfImages < 2)
	{
		std::cout<< "WARNING: number of iamges is less than 2" << std::endl;
		return;
	}

	if(numOfImages > numOfWidgets)
	{
		std::cout<< "WARNING: number of images is bigger than that of glwidgets" << std::endl;
		numOfImages = numOfWidgets;
	}	
	
	for(int i = 0; i<numOfImages; i++)
	{
		double aspectRatio = static_cast<double>((*ims)[i]._image.rows)/static_cast<double>( (*ims)[i]._image.cols);
		_glWidgets[i]->setFixedHeight(_fixedWidth * aspectRatio);
		_glWidgets[i]->_tex.upLoad( (*ims)[i]._image.data, (*ims)[i]._image.cols, (*ims)[i]._image.rows);
		if(_glWidgets[i]->isHidden())
			_glWidgets[i]->show();
	}
	for(int i = numOfImages; i<numOfWidgets; i++)
		_glWidgets[i]->hide();
}

viewSynForm::~viewSynForm(void)
{

}
