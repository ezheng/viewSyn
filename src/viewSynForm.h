#pragma once
#include <qwidget.h>
#include "GLWidget.h"
#include "ui_mainWidget.h"
#include "texture2D.h"
#include <vector>
#include <QCheckBox>
#include <string>
#include "image.h"
#include <qmutex.h>

class viewSynForm :	public QWidget
{
	Q_OBJECT
public:
	viewSynForm(std::vector<image>** ims, QGLWidget* widgetForContext/*, QMutex* imageMutex*/);
	~viewSynForm(void);
	QList<GLWidget*> _glWidgets;
	QList<QCheckBox> _allCheckBoxes;
	void setUpImages(std::vector<image>* ims, QGLWidget* widgetForContext);
	QMutex* _imageMutex;

private:
	Ui::mainWidget ui;
	int _fixedWidth;		// width of glWidget
	std::string _imageSavingPath;
	std::vector<image>** _ims;


private slots:
	void getSavingPath();
	void saveImage();

};

