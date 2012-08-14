#ifndef PTRFLEA3_H
#define PTRFLEA3_H

#include <FlyCapture2.h>
#include <BusManager.h>
#include <iostream>
#include <string>
#include <qthread.h>
#include <qobject.h>
#include <opencv\highgui.h>
#include "image.h"
#include <qmutex.h>

class oneCame;

class allImageCaptureManager:public QObject
{
	Q_OBJECT
public:
	allImageCaptureManager(std::vector<image>* ims);
	~allImageCaptureManager();
	QThread* _threads;
	FlyCapture2::BusManager _busMgr;

	//std::vector<image>* retrieveImgsAllParallel(std::vector<image>* newBuffer);	
	void startCaptureAll();
	void stopCaptureAll(){};
	void retrieveImgsAll();
	unsigned int returnNumberOfCams();
	void swapBuffer( std::vector<image> ** newBuffer);

private:
	unsigned int _numOfCams;
	std::vector<oneCame*> _allCames;
	std::vector<image> *_allIms;	
	bool allFlagsReady();
	std::vector<int> _indices;
signals:
	void retrieveImgsAllParallel();
	void imageReady_SIGNAL();

public slots:
	void retrieveImgsAllParallel_SLOTS();

};


class oneCame:public QObject
{
	Q_OBJECT
public:
	oneCame(int id, FlyCapture2::BusManager &busMgr, cv::Mat* img, image* myFormatImg);
	~oneCame();
	float getFrameRate();	
	void startCapture();
	
	void stopCapture();
	void retrieveImage();
	void updateCameraPoints(cv::Mat *img){_imgOPENCV = img;}
	bool isReady();

	bool _readyFlag;
	FlyCapture2:: CameraInfo getCamInfo();

private:
	void restartCam();
	bool PollForTriggerReady();
	void tartCaptureParallel();

	FlyCapture2::Format7PacketInfo _fmt7PacketInfo;
	FlyCapture2::Format7ImageSettings _fmt7ImageSettings;
	FlyCapture2::TriggerMode _triggerMode;
//
	FlyCapture2::Camera _cam;
	FlyCapture2::Image _img;
	cv::Mat* _imgOPENCV;
	cv::Mat _tempImgOPENCV;

	image *_myFormatImg;

	unsigned int _cameraId;	// 0 , 1 ,2...
	FlyCapture2::PGRGuid _guid;
	QMutex _mutex;
	
	// for undistortion
	cv::Mat _map1;
	cv::Mat _map2;
	bool _firstTime;
public slots:		
	void retrieveImageParallel();


	


};



#endif