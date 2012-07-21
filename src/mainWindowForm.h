#ifndef MAINWINDOWFORM_H
#define MAINWINDOWFORM_H

#include "GLWidgetVirtualView.h"
#include "qwidget.h"
#include "ui_mainWindow.h"
#include "viewSynForm.h"
#include <string>
#include "image.h"
#include "GLWidgetAllImgs.h"
#include <QMdiSubWindow>
#include <iostream>
#include "ptrFlea3.h"
#include <qtimer.h>
#include <qmutex.h>


class mainWindowForm :
	public QMainWindow
{
	Q_OBJECT

public:
	mainWindowForm(void);
	~mainWindowForm(void);

public:
	Ui::MainWindow ui;	
	//
	viewSynForm *_imagesForm;
	GLWidgetAllImgs *_allImagesForm;
	GLWidgetVirtualView *_virtualViewForm;



	QGLWidget *_widgetForContext;

	void readImages(std::string fileName);
	std::vector<image>* _allImages;
	std::vector<image>* _allImagesBackBuffer;

	allImageCaptureManager* _busHandler;
	//QMutex* _imageMutex;

private:
	glm::vec3 _xyzMin;
	glm::vec3 _xyzMax;

	bool _wasCapturing;
	void showImageWindow();
	QTimer _timer;
	
private slots:
	void openFile_slot();
	void startCapture_slot();	
	void startViewSynthesis_slot();	

	void retrieveImages();

signals:
	void redrawImages();
	void redrawCameraPoses();
	

};



#endif