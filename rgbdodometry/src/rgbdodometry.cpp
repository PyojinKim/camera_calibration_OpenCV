#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <unistd.h>

using namespace cv;
using namespace std;


static void cvtDepth2Cloud( const Mat& depth, Mat& cloud, const Mat& cameraMatrix )
{
    const float inv_fx = 1.f/cameraMatrix.at<float>(0,0);
    const float inv_fy = 1.f/cameraMatrix.at<float>(1,1);
    const float ox = cameraMatrix.at<float>(0,2);
    const float oy = cameraMatrix.at<float>(1,2);
    cloud.create( depth.size(), CV_32FC3 );
    for( int y = 0; y < cloud.rows; y++ )
    {
        Point3f* cloud_ptr = (Point3f*)cloud.ptr(y);
        const float* depth_prt = (const float*) depth.ptr(y);
        for( int x = 0; x < cloud.cols; x++ )
        {
            float z = depth_prt[x];
            cloud_ptr[x].x = (x - ox) * z * inv_fx;
            cloud_ptr[x].y = (y - oy) * z * inv_fy;
            cloud_ptr[x].z = z;
        }
    }
}


template<class ImageElemType>
static void warpImage( const Mat& image, const Mat& depth,
                       const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
                       Mat& warpedImage )
{
    const Rect rect = Rect(0, 0, image.cols, image.rows);

    vector<Point2f> points2d;
    Mat cloud, transformedCloud;

    cvtDepth2Cloud( depth, cloud, cameraMatrix );
    perspectiveTransform( cloud, transformedCloud, Rt );
    projectPoints( transformedCloud.reshape(3,1), Mat::eye(3,3,CV_64FC1), Mat::zeros(3,1,CV_64FC1), cameraMatrix, distCoeff, points2d );

    Mat pointsPositions( points2d );
    pointsPositions = pointsPositions.reshape( 2, image.rows );

    warpedImage.create( image.size(), image.type() );
    warpedImage = Scalar::all(0);

    Mat zBuffer( image.size(), CV_32FC1, FLT_MAX );
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            const Point3f p3d = transformedCloud.at<Point3f>(y,x);
            const Point p2d = pointsPositions.at<Point2f>(y,x);
            if( !cvIsNaN(cloud.at<Point3f>(y,x).z) && cloud.at<Point3f>(y,x).z > 0 &&
                rect.contains(p2d) && zBuffer.at<float>(p2d) > p3d.z )
            {
                warpedImage.at<ImageElemType>(p2d) = image.at<ImageElemType>(y,x);
                zBuffer.at<float>(p2d) = p3d.z;
            }
        }
    }
}

int main(int argc, char** argv)
{
    float vals[] = {525., 0., 3.1950000000000000e+02,
                    0., 525., 2.3950000000000000e+02,
                    0., 0., 1.};

    const Mat cameraMatrix = Mat(3,3,CV_32FC1,vals);
    const Mat distCoeff(1,5,CV_32FC1,Scalar(0));

    if(1)
    {
        cout << "Format: image0 depth1 image1 depth2 [transformationType]" << endl;
        cout << "Depth file must be 16U image stored depth in mm." << endl;
        cout << "Transformation types:" << endl;
        cout << "   -rbm - rigid body motion (default)" << endl;
        cout << "   -r   - rotation rotation only" << endl;
        cout << "   -t   - translation only" << endl;
    }


    Mat colorImage1 = imread("../rgbdodometry/image_00001.png");
    Mat depth1 = imread("../rgbdodometry/depth_00001.png", -1);

    // for debug
    //std::cout << depth1.at<unsigned short>(100,200) << std::endl;
    //usleep(10000000);


    Mat colorImage2 = imread("../rgbdodometry/image_00002.png");
    Mat depth2 = imread("../rgbdodometry/depth_00002.png", -1);


    int transformationType = RIGID_BODY_MOTION;
    if( argc == 6 )
    {
        string ttype = "-rbm";
        if( ttype == "-rbm" )
        {
            transformationType = RIGID_BODY_MOTION;
        }
        else if ( ttype == "-r")
        {
            transformationType = ROTATION;
        }
        else if ( ttype == "-t")
        {
            transformationType = TRANSLATION;
        }
        else
        {
            cout << "Unsupported transformation type." << endl;
            return -1;
        }
    }


    Mat grayImage1, grayImage2, depthFlt1, depthFlt2; /*in meters*/
    cvtColor( colorImage1, grayImage1, COLOR_BGR2GRAY );
    cvtColor( colorImage2, grayImage2, COLOR_BGR2GRAY );
    depth1.convertTo( depthFlt1, CV_32FC1, 1./1000 );
    depth2.convertTo( depthFlt2, CV_32FC1, 1./1000 );


    // for debug
    //std::cout << depthFlt1.at<float>(312,510) << std::endl;
    //usleep(10000000);


    TickMeter tm;
    Mat T_21;

    std::vector<int> iterCounts(4);
    iterCounts[0] = 7;
    iterCounts[1] = 7;
    iterCounts[2] = 7;
    iterCounts[3] = 10;

    std::vector<float> minGradMagnitudes(4);
    minGradMagnitudes[0] = 12;
    minGradMagnitudes[1] = 5;
    minGradMagnitudes[2] = 3;
    minGradMagnitudes[3] = 1;

    const float minDepth = 0.f;       //  in meters
    const float maxDepth = 4.f;       //  in meters
    const float maxDepthDiff = 0.07f; //  in meters

    tm.start();
    bool isFound = cv::RGBDOdometry( T_21, Mat(),
                                     grayImage1, depthFlt1, Mat(),
                                     grayImage2, depthFlt2, Mat(),
                                     cameraMatrix, minDepth, maxDepth, maxDepthDiff,
                                     iterCounts, minGradMagnitudes, transformationType );
    tm.stop();

    cout << "T_21 = " << T_21 << endl;
    cout << "T_21.inv() = " << T_21.inv() << endl;
    cout << "Time = " << tm.getTimeSec() << " sec." << endl;

    if( !isFound )
    {
        cout << "Rigid body motion cann't be estimated for given RGBD data."  << endl;
        return -1;
    }

    Mat warpedImage0;
    warpImage<Point3_<uchar> >( colorImage1, depthFlt1, T_21, cameraMatrix, distCoeff, warpedImage0 );

    imshow( "image0", colorImage1 );
    imshow( "warped_image0", warpedImage0 );
    imshow( "image1", colorImage2 );
    waitKey();

    return 0;
}
