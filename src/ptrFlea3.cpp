#include "ptrFlea3.h"
#include <vector>
#include <algorithm>
#include <QMutexLocker>
#include <algorithm>
#include <numeric> 
#include <opencv/cv.h>

#define PGR_SAFE_CALL(error) _pgrSafeCall(error, __FILE__, __LINE__)
void _pgrSafeCall(FlyCapture2::Error err, std::string fileName, int lineNum)
{
	if(err != FlyCapture2::PGRERROR_OK)
	{	
		std::cout << "File: " << fileName << " Line: " << lineNum << std::endl;
		err.PrintErrorTrace();
		std::cout << std::endl;
	}
}

unsigned int allImageCaptureManager::returnNumberOfCams()
{
	PGR_SAFE_CALL( _busMgr.GetNumOfCameras(&_numOfCams));
	return _numOfCams;
}

std::vector<int> ordered(std::vector<unsigned int> const& values) {
    std::vector<int> indices(values.size());
    std::iota(begin(indices), end(indices), static_cast<int>(0));

    std::sort(
        begin(indices), end(indices),
        [&](int a, int b) { return values[a] < values[b]; }
    );
    return indices;
}

allImageCaptureManager::allImageCaptureManager(std::vector<image>* ims):_numOfCams(0)
{
	PGR_SAFE_CALL( _busMgr.GetNumOfCameras(&_numOfCams));

	if(_numOfCams == 0)
	{
		std::cout<< "None of Cams are detected" << std::endl; 
		return;
	}	
	_allIms = ims;
	_allIms->clear();
	_allIms->resize(_numOfCams);

	for(unsigned int i = 0; i<_numOfCams; i++)
	{
		//ims.push_back(cv::Mat());
		oneCame *oneCam = new oneCame(i, _busMgr, &((*_allIms)[i]._image), &((*_allIms)[i]));
		_allCames.push_back(oneCam);
	}

	for(unsigned int i = 0; i<_numOfCams; i++)
		_allCames[i]->startCapture(); // start capture
	//--------------------------------------------------------------
	std::vector<unsigned int> _serialNumber(_numOfCams);
	for(unsigned int i = 0;i<_numOfCams; i++)
	{
		_serialNumber[i] = _allCames[i]->getCamInfo().serialNumber;
	}
	_indices = ordered(_serialNumber);
	//--------------------------------------------------------------

	_threads = new QThread[_numOfCams];
	for(unsigned int i = 0; i<_numOfCams;i++)
	{
		Q_ASSERT(QObject::connect( this, SIGNAL(retrieveImgsAllParallel()), _allCames[i], SLOT(retrieveImageParallel())));					
	}

	// move to threads and start
	for(unsigned int i = 0; i<_numOfCams; i++)					
		_allCames[i]->moveToThread(&_threads[i]);		
	
	for(unsigned int i = 0; i<_numOfCams; i++)
		_threads[i].start();	// start the new thread	
		
	//emit retrieveImgsAllParallel_SLOTS();
	
}

bool allImageCaptureManager::allFlagsReady()
{ 
	bool allReady = true;
	for(unsigned int i = 0; i< _numOfCams; i++)
	{
		allReady = allReady && _allCames[i]->isReady();
		//std::cout<< "camera " << i << " flag " << bool(_allCames[i]->_readyFlag) << std::cout<< std::endl;
	}
	return allReady;
}

void allImageCaptureManager::swapBuffer( std::vector<image> ** newBuffer)
{

	std::vector<image>* temp;
	temp = _allIms;
	_allIms = *newBuffer;
	*newBuffer = temp;
//----------------------------------------------------------------
	//_allIms->clear();
	//_allIms->resize(_numOfCams);	
	for(unsigned int i = 0; i<_numOfCams; i++)
	{		 
		_allCames[_indices[i]]->updateCameraPoints(&((*_allIms)[i]._image));
	}
}

void allImageCaptureManager:: retrieveImgsAllParallel_SLOTS( )
{
	//while(! allFlagsReady())
	//{/*std::cout<<"waiting " << iii << std::endl; ii++;*/ }
	//std::cout<< std::endl;

	for(unsigned int i = 0; i<_numOfCams; i++)
		_allCames[i]->_readyFlag = false;	
	emit retrieveImgsAllParallel(); 

	// wait here:
	while(! allFlagsReady())
	{/*std::cout<<"waiting " << iii << std::endl; ii++;*/ }
#ifdef _DEBUG 
	std::cout<< std::endl;
#endif
	// emit signal to the main thread to notify 
	emit imageReady_SIGNAL();

}


void allImageCaptureManager::retrieveImgsAll()
{
	for(unsigned int i = 0; i<_allCames.size(); i++)
	{
		_allCames[i]->retrieveImage();
	}
}


void allImageCaptureManager::startCaptureAll()
{
	for(unsigned int i = 0; i<_numOfCams; i++)
	{				
		_allCames[i]->startCapture();
	}
}


allImageCaptureManager::~allImageCaptureManager()
{
	for(unsigned int i = 0; i<_allCames.size(); i++)
	{
		if(_allCames[i] != NULL)
			delete _allCames[i];
	}
	if(_threads != NULL)
	{
		delete []_threads;
		_threads = NULL;
	}
}

//-----------------------------------------------------------------------------------------------------------

bool oneCame::isReady()
{
	QMutexLocker _qmt(&_mutex);
	return _readyFlag;		
}

bool oneCame::PollForTriggerReady()
{
    const unsigned int k_softwareTrigger = 0x62C;
    FlyCapture2::Error error;
    unsigned int regVal = 0;
    do 
    {
        error = _cam.ReadRegister( k_softwareTrigger, &regVal );
        if (error != FlyCapture2::PGRERROR_OK)
        {           
			return false;
        }
    } while ( (regVal >> 31) != 0 );
	return true;
}

float oneCame::getFrameRate()
{
	FlyCapture2::Property frmRate;
    frmRate.type = FlyCapture2::FRAME_RATE;
    PGR_SAFE_CALL( _cam.GetProperty( &frmRate ));
	return frmRate.absValue;
}

void oneCame::startCapture()
{
	PGR_SAFE_CALL(_cam.StartCapture());
}

void oneCame::stopCapture()
{
	PGR_SAFE_CALL(_cam.StopCapture());
}

void oneCame::retrieveImageParallel()
{
	
	retrieveImage();	
}

void oneCame::retrieveImage()
{
	//std::cout<< "camera: " << _cameraId << "start capturing." <<std::endl;
	PGR_SAFE_CALL(_cam.RetrieveBuffer( &_img));
	//std::cout<< "camera: " << _cameraId << "finish capturing." <<std::endl;

	  FlyCapture2::TimeStamp t1 = _img.GetTimeStamp();

#ifdef _DEBUG 
	 std::cout<< "camera: " << _cameraId << " second: " << t1.seconds << " microseconds" << ": " << t1.microSeconds 
		  << " current thread id" << GetCurrentThreadId() << std::endl;
#endif

		int width = _img.GetCols();
		int height = _img.GetRows();
		int stride = _img.GetStride();
		if(width*3 != stride)
			std::cout<< " WARING: width not equals to stride" << __FILE__ << __LINE__ << std::endl;
		
		if(_imgOPENCV->empty())
		{
			_imgOPENCV->create(height, width, CV_8UC3);	
			_tempImgOPENCV.create(height, width, CV_8UC3);			
			
			//cvInitUndistortMap(_myFormatImg->_K, _myFormatImg->_kc, &map1, &map2);
		}
		unsigned char *dataPoint = _img.GetData();

		if(_myFormatImg->_kc.at<double>(0) != 0 || _myFormatImg->_kc.at<double>(1) != 0 || _myFormatImg->_kc.at<double>(2) != 0 ||
			_myFormatImg->_kc.at<double>(3) != 0 || _myFormatImg->_kc.at<double>(4) != 0)
		{
			for(int i = 0; i<height; i++)
			{
				//for(int i
				
				for(int j = 0; j< width; j++)
				{
					int offsetOrig = i*stride + j * 3;
					int offsetDest = (height - 1 - i) * _tempImgOPENCV.step + j * 3;
					_tempImgOPENCV.data[offsetDest + 2] = _myFormatImg->_LUT[dataPoint[offsetOrig]];
					_tempImgOPENCV.data[offsetDest ] = _myFormatImg->_LUT[dataPoint[offsetOrig + 2]];
					_tempImgOPENCV.data[offsetDest + 1] = _myFormatImg->_LUT[dataPoint[offsetOrig + 1]];
				}
			}
	// undistort image		
		//	cv::undistort( _tempImgOPENCV, *_imgOPENCV, _myFormatImg->_K, _myFormatImg->_kc );
			//static bool firstTime = true;
			if(_firstTime)
			{
				_map1.create(height, width, CV_32FC1);
				_map2.create(height, width, CV_32FC1);
				cv::Mat identityMatrix = cv::Mat::eye(3, 3, CV_32F);
				cv::initUndistortRectifyMap(_myFormatImg->_K, _myFormatImg->_kc, identityMatrix, _myFormatImg->_K, cv::Size(width, height), CV_32FC1, _map1, _map2);
				_firstTime = false;
			}
			cv::remap( _tempImgOPENCV, *_imgOPENCV, _map1, _map2, cv::INTER_LINEAR);

		}
		else
		{
			for(int i = 0; i<height; i++)
			{
				//for(int i
				for(int j = 0; j< width; j++)
				{
					int offsetOrig = i*stride + j * 3;
					int offsetDest = (height - 1 - i) * _imgOPENCV->step + j * 3;
					_imgOPENCV->data[offsetDest + 2] = dataPoint[offsetOrig];
					_imgOPENCV->data[offsetDest ] = dataPoint[offsetOrig + 2];
					_imgOPENCV->data[offsetDest + 1] = dataPoint[offsetOrig + 1];
				}
			}
		}	

	//	this->stopCapture();
		{
		QMutexLocker _qmt(&_mutex);
		_readyFlag = true;
		}


}

void oneCame::restartCam()
{
	const unsigned int k_cameraPower = 0x610;
	unsigned int k_powerVal = 0x00000000;
	PGR_SAFE_CALL( _cam.WriteRegister( k_cameraPower, k_powerVal ));
	k_powerVal = 0x80000000;
	PGR_SAFE_CALL( _cam.WriteRegister( k_cameraPower, k_powerVal ));
	unsigned int regVal = 0;
	do 
	{
		Sleep(100); 
		PGR_SAFE_CALL(_cam.ReadRegister(k_cameraPower, &regVal));		
	} while ((regVal & k_powerVal) == 0);

}

oneCame::oneCame(int id, FlyCapture2::BusManager &busMgr, cv::Mat* img, image *myFormatImg):_cameraId(id), _imgOPENCV(img), _readyFlag(false), _myFormatImg(myFormatImg), _firstTime(true)
{
	
	PGR_SAFE_CALL(busMgr.GetCameraFromIndex(_cameraId, &_guid));
	PGR_SAFE_CALL(_cam.Connect(&_guid));

	// restart cam:
	//restartCam();
	
	int centerX = 1328/2;
	int centerY = 1024/2;
	//_allImgs = new FlyCapture2::Image[_numOfCams];
    _fmt7ImageSettings.mode = FlyCapture2::MODE_4;;	  
	_fmt7ImageSettings.width = 656;   
	_fmt7ImageSettings.height = 524;
   // _fmt7ImageSettings.offsetX = centerX - _fmt7ImageSettings.width /2;
   // _fmt7ImageSettings.offsetY = centerY - _fmt7ImageSettings.height /2;
	_fmt7ImageSettings.offsetX = 0;
	_fmt7ImageSettings.offsetY = 0;
    _fmt7ImageSettings.pixelFormat =  FlyCapture2::PIXEL_FORMAT_RGB8;
    // Validate the settings to make sure that they are valid
	bool valid;
	PGR_SAFE_CALL( _cam.ValidateFormat7Settings(&_fmt7ImageSettings,&valid, &_fmt7PacketInfo ));	
	PGR_SAFE_CALL( _cam.SetFormat7Configuration( &_fmt7ImageSettings, _fmt7PacketInfo.recommendedBytesPerPacket ));
	

	FlyCapture2::Property prop;
	// set gain
	prop.type = FlyCapture2::GAIN; prop.autoManualMode = false; prop.absControl = true; prop.absValue = 10; 
	PGR_SAFE_CALL(_cam.SetProperty( &prop ));
	// set shutter
	prop.type = FlyCapture2::SHUTTER; prop.autoManualMode = false; prop.onOff = true;
	prop.absControl = true; prop.absValue = 45; PGR_SAFE_CALL(_cam.SetProperty( &prop ));
	// set frame rate	
	prop.type = FlyCapture2::FRAME_RATE; prop.autoManualMode = false; prop.onOff = true;
	prop.absControl = true; prop.absValue = 20; PGR_SAFE_CALL(_cam.SetProperty( &prop ));
	// set auto white balance
	FlyCapture2::Property prop1;
	prop1.type = FlyCapture2::WHITE_BALANCE; prop1.onOff = true; prop1.autoManualMode = false;
	prop1.valueA = 490; prop1.valueB = 790; PGR_SAFE_CALL(_cam.SetProperty( &prop1 ));
	// disable sharpness
	FlyCapture2::Property sharpnessProp;
	sharpnessProp.type = FlyCapture2::SHARPNESS; sharpnessProp.onOff = false; PGR_SAFE_CALL(_cam.SetProperty( &sharpnessProp ));
	// diable gamma
	FlyCapture2::Property gammaProp;
	gammaProp.type = FlyCapture2::GAMMA; gammaProp.onOff = false; PGR_SAFE_CALL(_cam.SetProperty( &gammaProp ));


	// set trigger mode
	_triggerMode.onOff = true;
	_triggerMode.mode = 15;
	_triggerMode.parameter = 10;
	_triggerMode.source = 0;	 
	PGR_SAFE_CALL(_cam.SetTriggerMode( &_triggerMode));		

	std::cout<< getFrameRate() << std::endl;

}

oneCame::~oneCame()
{					
	_triggerMode.onOff = false;
	PGR_SAFE_CALL(_cam.SetTriggerMode( &_triggerMode));
	PGR_SAFE_CALL(_cam.Disconnect());
	
};

FlyCapture2:: CameraInfo oneCame::getCamInfo()
{
	FlyCapture2::CameraInfo camInfo;
    PGR_SAFE_CALL(_cam.GetCameraInfo( &camInfo ));	

	return camInfo;
}

