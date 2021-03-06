﻿//------------------------------------------------------------------------------
// <copyright file="FTHelper.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

//#include "StdAfx.h"
#include <Windows.h>
#include "FTHelper.h"
//#include "Visualize.h"

#ifdef SAMPLE_OPTIONS
#include "Options.h"
#else
PVOID _opt = NULL;
#endif

FTHelper::FTHelper()
{
    m_pFaceTracker = 0;
    //m_hWnd = NULL;
    m_pFTResult = NULL;
    m_colorImage = NULL;
    m_depthImage = NULL;
    m_ApplicationIsRunning = false;
    m_LastTrackSucceeded = false;
    m_CallBack = NULL;
    m_XCenterFace = 0;
    m_YCenterFace = 0;
 //   m_hFaceTrackingThread = NULL;
    m_DrawMask = TRUE;
    m_depthType = NUI_IMAGE_TYPE_DEPTH;
    m_depthRes = NUI_IMAGE_RESOLUTION_INVALID;
    m_bNearMode = FALSE;
    m_bFallbackToDefault = FALSE;
    m_colorType = NUI_IMAGE_TYPE_COLOR;
    m_colorRes = NUI_IMAGE_RESOLUTION_INVALID;
}

FTHelper::~FTHelper()
{
    //Stop();

	m_pFaceTracker->Release();
    m_pFaceTracker = NULL;

    if(m_colorImage)
    {
        m_colorImage->Release();
        m_colorImage = NULL;
    }

    if(m_depthImage) 
    {
        m_depthImage->Release();
        m_depthImage = NULL;
    }

    if(m_pFTResult)
    {
        m_pFTResult->Release();
        m_pFTResult = NULL;
    }
    m_KinectSensor.Release();
}

HRESULT FTHelper::Init(HWND hWnd, FTHelperCallBack callBack, PVOID callBackParam, 
    NUI_IMAGE_TYPE depthType, NUI_IMAGE_RESOLUTION depthRes, BOOL bNearMode, BOOL bFallbackToDefault, NUI_IMAGE_TYPE colorType, NUI_IMAGE_RESOLUTION colorRes, BOOL bSeatedSkeletonMode)
{
    if (!hWnd || !callBack)
    {
        return E_INVALIDARG;
    }
  //  m_hWnd = hWnd;
    m_CallBack = callBack;
    m_CallBackParam = callBackParam;
    m_ApplicationIsRunning = true;
    m_depthType = depthType;
    m_depthRes = depthRes;
    m_bNearMode = bNearMode;
    m_bFallbackToDefault = bFallbackToDefault;
	m_bSeatedSkeletonMode = bSeatedSkeletonMode;
    m_colorType = colorType;
    m_colorRes = colorRes;
 //   m_hFaceTrackingThread = CreateThread(NULL, 0, FaceTrackingStaticThread, (PVOID)this, 0, 0);
    return S_OK;
}

HRESULT FTHelper::Init(NUI_IMAGE_TYPE depthType, NUI_IMAGE_RESOLUTION depthRes, BOOL bNearMode, BOOL bFallbackToDefault, NUI_IMAGE_TYPE colorType, NUI_IMAGE_RESOLUTION colorRes, BOOL bSeatedSkeletonMode)
{
    
    m_ApplicationIsRunning = true;
    m_depthType = depthType;
    m_depthRes = depthRes;
    m_bNearMode = bNearMode;
    m_bFallbackToDefault = bFallbackToDefault;
	m_bSeatedSkeletonMode = bSeatedSkeletonMode;
    m_colorType = colorType;
    m_colorRes = colorRes;
 //   m_hFaceTrackingThread = CreateThread(NULL, 0, FaceTrackingStaticThread, (PVOID)this, 0, 0);
	//FaceTrackingThread(); by Enliang
    return S_OK;
}

//HRESULT FTHelper::Stop()
//{
    /*m_ApplicationIsRunning = false;
    if (m_hFaceTrackingThread)
    {
        WaitForSingleObject(m_hFaceTrackingThread, 1000);
    }
    m_hFaceTrackingThread = 0;
    return S_OK;*/
//}

#include <iostream>
#ifndef _WIN64
	#include <imdebug.h>
#endif
BOOL FTHelper::SubmitFraceTrackingResult(IFTResult* pResult)
{
    if (pResult != NULL && SUCCEEDED(pResult->GetStatus()))
    {
        if (m_CallBack)
        {
            (*m_CallBack)(m_CallBackParam);
        }

        if (m_DrawMask)
        {
            FLOAT* pSU = NULL;
            UINT numSU;
            BOOL suConverged;
            m_pFaceTracker->GetShapeUnits(NULL, &pSU, &numSU, &suConverged);
            POINT viewOffset = {0, 0};
            FT_CAMERA_CONFIG cameraConfig;
            if (m_KinectSensorPresent)
            {
                m_KinectSensor.GetVideoConfiguration(&cameraConfig);
            }
            else
            {
                cameraConfig.Width = 640;
                cameraConfig.Height = 480;
                cameraConfig.FocalLength = 500.0f;
            }
            IFTModel* ftModel;
            HRESULT hr = m_pFaceTracker->GetFaceModel(&ftModel);
            if (SUCCEEDED(hr))
            {
               // hr = VisualizeFaceModel(m_colorImage, ftModel, &cameraConfig, pSU, 1.0, viewOffset, pResult, 0x00FFFF00);
                ftModel->Release();
				RECT rectFace;
				hr = pResult->GetFaceRect(&rectFace);
                if (SUCCEEDED(hr))
                {
                    POINT leftTop = {rectFace.left, rectFace.top};
                    POINT rightTop = {rectFace.right - 1, rectFace.top};
                    POINT leftBottom = {rectFace.left, rectFace.bottom - 1};
                    POINT rightBottom = {rectFace.right - 1, rectFace.bottom - 1};
					

                    //UINT32 nColor = 0xff00ff;
                   // SUCCEEDED(hr = pColorImg->DrawLine(leftTop, rightTop, nColor, 1)) &&
                    //    SUCCEEDED(hr = pColorImg->DrawLine(rightTop, rightBottom, nColor, 1)) &&
                    //    SUCCEEDED(hr = pColorImg->DrawLine(rightBottom, leftBottom, nColor, 1)) &&
                    //    SUCCEEDED(hr = pColorImg->DrawLine(leftBottom, leftTop, nColor, 1));
                }

            }
        }
    }
    return TRUE;
}

// We compute here the nominal "center of attention" that is used when zooming the presented image.
void FTHelper::SetCenterOfImage(IFTResult* pResult)
{
    float centerX = ((float)m_colorImage->GetWidth())/2.0f;
    float centerY = ((float)m_colorImage->GetHeight())/2.0f;
    if (pResult)
    {
        if (SUCCEEDED(pResult->GetStatus()))
        {
            RECT faceRect;
            pResult->GetFaceRect(&faceRect);
            centerX = (faceRect.left+faceRect.right)/2.0f;
            centerY = (faceRect.top+faceRect.bottom)/2.0f;
        }
        m_XCenterFace += 0.02f*(centerX-m_XCenterFace);
        m_YCenterFace += 0.02f*(centerY-m_YCenterFace);
    }
    else
    {
        m_XCenterFace = centerX;
        m_YCenterFace = centerY;
    }
}

// Get a video image and process it.
void FTHelper::CheckCameraInput()
{
    HRESULT hrFT = E_FAIL;

    if (m_KinectSensorPresent && m_KinectSensor.GetVideoBuffer())
    {
        HRESULT hrCopy = m_KinectSensor.GetVideoBuffer()->CopyTo(m_colorImage, NULL, 0, 0);
        if (SUCCEEDED(hrCopy) && m_KinectSensor.GetDepthBuffer())
        {
            hrCopy = m_KinectSensor.GetDepthBuffer()->CopyTo(m_depthImage, NULL, 0, 0);
        }
		// show the color iamge using imdebug
        // Do face tracking
        if (SUCCEEDED(hrCopy))
        {
            FT_SENSOR_DATA sensorData(m_colorImage, m_depthImage, m_KinectSensor.GetZoomFactor(), m_KinectSensor.GetViewOffSet());

            FT_VECTOR3D* hint = NULL;
            if (SUCCEEDED(m_KinectSensor.GetClosestHint(m_hint3D)))
            {
                hint = m_hint3D;
            }
            if (m_LastTrackSucceeded)
            {
                hrFT = m_pFaceTracker->ContinueTracking(&sensorData, hint, m_pFTResult);
            }
            else
            {
                hrFT = m_pFaceTracker->StartTracking(&sensorData, NULL, hint, m_pFTResult);
            }
        }
    }

    m_LastTrackSucceeded = SUCCEEDED(hrFT) && SUCCEEDED(m_pFTResult->GetStatus());
	RECT rectFace; rectFace.left = -1; rectFace.right = 0; rectFace.bottom = -1; rectFace.top = 0;
	unsigned char* data = m_colorImage->GetBuffer();;
    if (m_LastTrackSucceeded)
    {
		//unsigned int buffersize = m_colorImage->GetBufferSize();
		//std::cout<< "the size of the buffer is: " << buffersize << std::endl;
		//std::cout<< "the height and width of the image is: " << m_colorImage->GetHeight() << " " << m_colorImage->GetWidth() << std::endl;
		//std::cout<< " the stride: " << m_colorImage->GetStride() << std::endl;
		//std::cout<< sizeof(unsigned short) << std::endl;
		
		//imdebug("rgba w=%d h=%d %p", m_colorImage->GetWidth(), m_colorImage->GetHeight(), data);
        //SubmitFraceTrackingResult(m_pFTResult);		
		HRESULT hr = m_pFTResult->GetFaceRect(&rectFace);
		// draw rectangle on the image:
		drawVLines(data, m_colorImage->GetStride(), rectFace.left, rectFace.top, rectFace.bottom, m_colorImage->GetWidth());
		drawVLines(data, m_colorImage->GetStride(), rectFace.right, rectFace.top, rectFace.bottom, m_colorImage->GetWidth());
		drawHLines(data, m_colorImage->GetStride(), rectFace.top, rectFace.left, rectFace.right, m_colorImage->GetHeight());
		drawHLines(data, m_colorImage->GetStride(), rectFace.bottom, rectFace.left, rectFace.right, m_colorImage->GetHeight());
		
	//	if (SUCCEEDED(hr))
    //    {
			/*POINT leftTop = {rectFace.left, rectFace.top};
	POINT rightTop = {rectFace.right - 1, rectFace.top};
	POINT leftBottom = {rectFace.left, rectFace.bottom - 1};
	POINT rightBottom = {rectFace.right - 1, rectFace.bottom - 1};
	std::cout<< "leftTop: " << leftTop.x << " " << leftTop.y << std::endl;
	std::cout<< "rightTop: " << rightTop.x << " " << rightTop.y << std::endl;
	std::cout<< "leftBottom: " << leftBottom.x << " " << leftBottom.y << std::endl;
	std::cout<< "rightBottom: " << rightBottom.x << " " << rightBottom.y << std::endl;*/
			
	//	}
    }
    else
    {		
        m_pFTResult->Reset();
    }
	
	emit drawKinect_SIGNALS( data, m_colorImage->GetWidth(), m_colorImage->GetHeight(), rectFace.left, rectFace.right -1 , rectFace.bottom, rectFace.top - 1 );

    SetCenterOfImage(m_pFTResult);

}

void FTHelper::drawHLines(unsigned char * data, int stride, int row, int colStart, int colEnd, int imageHeight)
{
	for(int j = -1; j <= 1; j++)
	{
		int actualRow = row + j;
		if(actualRow > 0 && actualRow<imageHeight)
		{
			int base = (actualRow) * stride;
			for(int i = colStart+1; i<colEnd; i++)
			{
				data[base + i * 4 + 0] = 255;
				data[base + i * 4 + 1] = 255;
				data[base + i * 4 + 2] = 255;
				data[base + i * 4 + 3] = 255;
			}
		}			
	}		
}

void FTHelper::drawVLines(unsigned char * data, int stride, int col, int rowStart, int rowEnd, int imageWidth)
{
	
	for(int j = -1; j<=1; j++)
	{
		int actualCol = col + j;
		if(actualCol > 0 && actualCol<imageWidth)
		{
			int offset = actualCol * 4;	
			for(int i = rowStart+1; i<rowEnd; i++)
			{
				int base = i * stride;
				data[base + offset + 0] = 255;
				data[base + offset + 1] = 255;
				data[base + offset + 2] = 255;
				data[base + offset + 3] = 255;
			}
		}
	}	
}

//DWORD WINAPI FTHelper::FaceTrackingStaticThread(PVOID lpParam)
//{
//    FTHelper* context = static_cast<FTHelper*>(lpParam);
//    if (context)
//    {
//        return context->FaceTrackingThread();
//    }
//    return 0;
//}

void FTHelper::FaceTrackingThread()
{
	//std::cout<<"current threadId GLWidgetAllImgs" << GetCurrentThreadId() << std::endl;

	this->Init(NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_320x240, true, true, NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, false);

    FT_CAMERA_CONFIG videoConfig;
    FT_CAMERA_CONFIG depthConfig;
    FT_CAMERA_CONFIG* pDepthConfig = NULL;
    
    // Try to get the Kinect camera to work
    HRESULT hr = m_KinectSensor.Init(m_depthType, m_depthRes, m_bNearMode, m_bFallbackToDefault, m_colorType, m_colorRes, m_bSeatedSkeletonMode);
    if (SUCCEEDED(hr))
    {
        m_KinectSensorPresent = TRUE;
        m_KinectSensor.GetVideoConfiguration(&videoConfig);
        m_KinectSensor.GetDepthConfiguration(&depthConfig);
        pDepthConfig = &depthConfig;
        m_hint3D[0] = m_hint3D[1] = FT_VECTOR3D(0, 0, 0);
    }
    else
    {
        m_KinectSensorPresent = FALSE;
        WCHAR errorText[MAX_PATH];
        ZeroMemory(errorText, sizeof(WCHAR) * MAX_PATH);
        wsprintf(errorText, L"Could not initialize the Kinect sensor. hr=0x%x\n", hr);
        //MessageBoxW(m_hWnd, errorText, L"Face Tracker Initialization Error\n", MB_OK);
        return ;
    }

    // Try to start the face tracker.
    m_pFaceTracker = FTCreateFaceTracker(_opt);
    if (!m_pFaceTracker)
    {
      //  MessageBoxW(m_hWnd, L"Could not create the face tracker.\n", L"Face Tracker Initialization Error\n", MB_OK);
        return ;
    }

    hr = m_pFaceTracker->Initialize(&videoConfig, pDepthConfig, NULL, NULL); 
    if (FAILED(hr))
    {
        WCHAR path[512], buffer[1024];
        GetCurrentDirectoryW(ARRAYSIZE(path), path);
        wsprintf(buffer, L"Could not initialize face tracker (%s). hr=0x%x", path, hr);

       // MessageBoxW(m_hWnd, /*L"Could not initialize the face tracker.\n"*/ buffer, L"Face Tracker Initialization Error\n", MB_OK);

        return ;
    }

    hr = m_pFaceTracker->CreateFTResult(&m_pFTResult);
    if (FAILED(hr) || !m_pFTResult)
    {
       // MessageBoxW(m_hWnd, L"Could not initialize the face tracker result.\n", L"Face Tracker Initialization Error\n", MB_OK);
        return;
    }

    // Initialize the RGB image.
    m_colorImage = FTCreateImage();
    if (!m_colorImage || FAILED(hr = m_colorImage->Allocate(videoConfig.Width, videoConfig.Height, FTIMAGEFORMAT_UINT8_B8G8R8X8)))
    {
        return;
    }
    
    if (pDepthConfig)
    {
        m_depthImage = FTCreateImage();
        if (!m_depthImage || FAILED(hr = m_depthImage->Allocate(depthConfig.Width, depthConfig.Height, FTIMAGEFORMAT_UINT16_D13P3)))
        {
            return;
        }
    }

 //   SetCenterOfImage(NULL);
 //   m_LastTrackSucceeded = false;

 //  // while (m_ApplicationIsRunning)
 //  // {
	//while(!m_LastTrackSucceeded)
	//{ 
	//	CheckCameraInput();
	//}
	
	doFaceTracking_SLOT();

       // InvalidateRect(m_hWnd, NULL, FALSE);
      // UpdateWindow(m_hWnd);
    //    Sleep(16);
  //  }

    /*m_pFaceTracker->Release();
    m_pFaceTracker = NULL;

    if(m_colorImage)
    {
        m_colorImage->Release();
        m_colorImage = NULL;
    }

    if(m_depthImage) 
    {
        m_depthImage->Release();
        m_depthImage = NULL;
    }

    if(m_pFTResult)
    {
        m_pFTResult->Release();
        m_pFTResult = NULL;
    }
    m_KinectSensor.Release();*/
    return;
}

void FTHelper::doFaceTracking_SLOT()
{
	 SetCenterOfImage(NULL);
	 
	 static bool firstStart = true;
	 if(firstStart)
	 {
		 firstStart = false;
		m_LastTrackSucceeded = false;
	 }

   // while (m_ApplicationIsRunning)
   // {
	//while(!m_LastTrackSucceeded)
//	{ 
		CheckCameraInput();
		Sleep(16);
//	}
}

HRESULT FTHelper::GetCameraConfig(FT_CAMERA_CONFIG* cameraConfig)
{
    return m_KinectSensorPresent ? m_KinectSensor.GetVideoConfiguration(cameraConfig) : E_FAIL;
}
