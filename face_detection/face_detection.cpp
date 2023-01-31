#include<iostream>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>

int main(){
	// Frame scaling
	double scalingFactor = 4.0;

	// Define classifier and load haarCascade model file
	cv::CascadeClassifier faceCascade;
	faceCascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml");

	// Initialize the camera and check status
	cv::VideoCapture cap(0);
	if (!cap.isOpened()){
		std::cout << "Error opening camera!";
		return -1;
	}

	while (true){	
		// Capture frames from camera
		cv::Mat frame;
		cap >> frame;

		// Convert captured frames to greyscale and resize by scaling factor
		cv::Mat frameGray;
		cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
		cv::resize(frameGray, frameGray, cv::Size(frameGray.size().width / scalingFactor, frameGray.size().height / scalingFactor));

		// Detect faces of different sizes using cascade classifier
		std::vector<cv::Rect> faces;
		faceCascade.detectMultiScale(frameGray, faces, 1.3, 4);

		// Draw rectangle around the detected faces
		for (size_t i = 0; i < faces.size(); i++)
			cv::rectangle(frame, (faces[i].tl() * scalingFactor), (faces[i].br() * scalingFactor), cv::Scalar(0, 255, 0), 2);
		
		// Display the frame with detected faces then check if the user pressed the 'q' key
		imshow("Face Detection", frame);
		if (cv::waitKey(1) == 'q')
			break;
	}

	// Release the camera and close all windows
	cap.release();
	cv::destroyAllWindows();
}
