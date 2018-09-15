#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

#define debug_enable	 1  //开启debug，过程中的图像处理会被打开，
#define using_hsv		 0  //使用HSV进行二值化
#define using_video		 1  //1: 使用视频进行测试, 0: 使用图片进行测试

/// 全局变量的声明与初始化
int threholdMin = 200, threholdMax = 255;  //第一次提取白板的阈值
float alpha = 0.0;

//调节阈值
int threholdMinr = 150;
int threholdMing = 200;
int threholdMinb = 200;
int threholdMaxr = 255;
int threholdMaxg = 255;
int threholdMaxb = 255;

int threholdMinh = 70;
int threholdMins = 0;
int threholdMinv = 197;
int threholdMaxh = 170;
int threholdMaxs = 140;
int threholdMaxv = 255;

float kdiff_threshold = 1;         //判断白板矩形要求对边要平行
int rect_area_threshold = 1000;       //要求白板的面积不能小于这个阈值，能够过滤掉大多数的小方块噪点
int approx_poly_threshold = 20;      //用来拟合多边形，提高此值会降低多边形的边数，降低此值会拟合的边数变多

int possibleBoardSize = 0;
bool secJudge = false;  //用于第二次判断board是否合格，筛选

vector<vector<Point>>contours;   //存放二值化之后的轮廓
vector<Vec4i>hierarchy;			//存放轮廓之间的关系
vector<RotatedRect> rectCandi;
Rect_<float> safeBoundRect;  //安全的矩形

RNG rng(12345);
VideoCapture cap;
Mat frame;
Mat board;
Mat srcImage, grayImage, binImage;

int totalFrameNumber;

void Init(); 
Mat getImage();
Mat binaryImage(Mat image);   //对图像进行二值化处理，方便后面提取白板
bool detectBoard(Mat image);  //主要的提取白板模块
bool comp(const cv::RotatedRect &a, const cv::RotatedRect &b);
bool isRect(const vector<Point> poly);  //判断是否是矩形，过滤白板

void DrawRotatedRect(Mat &image, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness);
int filterBoard(Mat image, int cntBoard);  //第二次过滤白板，提高稳定性
//void selectFinalBoard(vector<>);
void on_trackbar(int, void*);
/****************************************************/
bool calcSafeRect(const RotatedRect& roi_rect, const Mat& src, Rect_<float>& safeBoundRect);
bool rotation(Mat& in, Mat& out, const Size rect_size, const Point2f center, double angle);
void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name);
Mat equalizeHist(Mat image);

int main()
{
	int currentFrame = 0;
	Init();

	//**************main loop************************
	try {
		while (totalFrameNumber--)
		{
			double start = static_cast<double>(getTickCount());

			//获取图像
			currentFrame++;
			//cout << "the current frame is: " << currentFrame << endl;
			frame = getImage();
			//frame = equalizeHist(frame);   //直方图均衡化，但是初步试验效果不是太好，有待测试

			binImage = binaryImage(frame);
			//detectBoard(binImage);
			if (detectBoard(binImage))
				filterBoard(frame, possibleBoardSize);


			double time = ((double)getTickCount() - start) / getTickFrequency();
			//cout << "所用时间为：" << time << "秒" << endl;
 			waitKey(2);
		}
	}
	catch (exception &ex)
	{
		std::cout << ex.what() << endl;
		std::cout << totalFrameNumber << endl;
		exit(0);
	}
	waitKey(0);
	return 0;
}

/*******************************************************************************/
void Init()
{
	///////////////////////////采集图像/////////////////////////
#if !using_video
	//读取照片信息
	totalFrameNumber = 1;
#endif

#if using_video
	cap.open("video25.mp4");
	if (!cap.isOpened())
		std::cout << "fail to open!" << endl;
	//获取整个帧数  
	totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
	std::cout << "整个视频共" << totalFrameNumber << "帧" << endl;
#endif // video
}

Mat getImage()
{
	Mat tempImage;
#if using_video
	cap >> tempImage;
#else 
	tempImage = imread("25.jpg");
	//resize(tempImage, tempImage, Size(tempImage.cols / 2, tempImage.rows / 2), 0, 0, INTER_LINEAR);  //截频照片太大了，适当缩小了一下，没啥用
#endif
	
#if debug_enable
	imshow("srcImage", tempImage);
#endif

	return tempImage.clone();
}

Mat binaryImage(Mat image)
{
	if (secJudge)
	{
		//第二次的时候过滤白板，最低阈值会降低，会让白板中间的数字更突出，可以过滤一些纯白板
		threholdMinr = 100;		threholdMing = 160;		threholdMinb = 160;
		threholdMaxr = 255;		threholdMaxg = 255;		threholdMaxb = 255;
	}
	else
	{
		threholdMinr = 140;		threholdMing = 200;		threholdMinb = 200;
		threholdMaxr = 255;		threholdMaxg = 255;		threholdMaxb = 255;		
	}
#if using_hsv
	cvtColor(image, image, CV_BGR2HSV);
	//GaussianBlur(image, image, Size(3, 3), 0, 0);
	inRange(image, Scalar(threholdMinh, threholdMins, threholdMinv), Scalar(threholdMaxh, threholdMaxs, threholdMaxv), binImage);
#else
	inRange(image, Scalar(threholdMinb, threholdMing, threholdMinr), Scalar(threholdMaxb, threholdMaxg, threholdMaxr), binImage);
#endif

	if (!secJudge)
	{
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));   //7*7的核稍微有点大，不过提取白板的时候大了会有利一些
		//dilate(binImage, binImage, element, Point(-1, -1), 1);
		//erode(binImage, binImage, element, Point(-1, -1), 1);     //膨胀腐蚀
		//Canny(binImage, binImage, 3, 9, 3);   //提取边缘
	}
#if debug_enable
	//imshow("Edge Detection", binImage);
	if(!secJudge) cv::imshow("first erosioning", binImage);
#endif
	return binImage.clone();
}

bool detectBoard(Mat image)
{
	findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point());  //找轮廓
	vector<vector<Point>>convexsHull(contours.size());
	vector<vector<Point>>poly(contours.size());
	vector<RotatedRect> rectCandidates;  //候选矩形
	
	Mat polyImage, boardImage, hullImage;
	frame.copyTo(polyImage);
	frame.copyTo(boardImage);
	frame.copyTo(hullImage);
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), poly[i], approx_poly_threshold, true);  //拟合多边形
		convexHull(poly[i], convexsHull[i], true);
		
		if (!isRect(convexsHull[i])) continue;
		if (hierarchy[i].val[2] == -1 || hierarchy[i].val[3] != -1) continue;       //因为KT板中间有黑色部分，外面的轮廓下会有一个子轮廓，没有子轮廓就跳过

		RotatedRect rect = minAreaRect(contours[i]);   //最小外接矩形，方便用长宽比来筛选
		float rectWidth = max(rect.size.width, rect.size.height);    //规定长边是width
		float rectHeight = min(rect.size.width, rect.size.height);	 //规定短边是height
		float cntArea = fabs(contourArea(contours[i], true));
	
 		if (rect.size.area() < 1000 || rect.size.area() > 5000) continue;

		//cout << "********************************" << endl;
		//cout << "poly.size: " << poly[i].size() << endl;
		//cout << "cntArea radio: " << cntArea << endl;
		//cout << "rect.size.area: " << rect.size.area() << endl;
		//cout << "rect.size.width: " << rect.size.width << endl;
		//cout << "rect.size.height: " << rect.size.height << endl;
		//cout << "radio: " << rectWidth / rectHeight << endl;

		if (!(cntArea / rect.size.area() > 0.5			&&	  //比较轮廓和矩形面积的比，理论上来说应该是1:1， 照片0.8左右
			cntArea / rect.size.area() < 1.0
			)) continue;


		if (rectWidth / rectHeight < 1.8 && rectWidth / rectHeight > 1.3) 	//对白板的外接矩形进行一个长宽比的限制，应该是1.5左右
		{
			cout << "cntArea radio: " << cntArea << endl;
			cout << "rect.size.area: " << rect.size.area() << endl;
			rectCandidates.push_back(rect);    //满足条件就进入候选

			std::cout << "there are possible boards!!!"<< endl;
			cout << "convexsHull.size: " << convexsHull[i].size() << endl;
		}
#if debug_enable
		if (!secJudge)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::drawContours(polyImage, poly, i, color, 2, 8, vector<Vec4i>(), 0, Point());
			cv::drawContours(hullImage, convexsHull, i, color, 2, 8, vector<Vec4i>(), 0, Point());
			//cv::imshow("ployImage", polyImage);
			//cv::imshow("hullImage", hullImage);
		}
#endif
	}
	if (rectCandidates.size() == 0)   //同时处理多个，对rectCandidates所有的对象都处理一下
	{
		//std::cout << "there is no board!" << endl;
		return false;
	}
	else possibleBoardSize = rectCandidates.size();
	rectCandi.clear();
	for (int i = 0; i < possibleBoardSize; i++)
	{
		rectCandi.push_back(rectCandidates[i]);
	}

#if 0
	Mat drawing = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}
	namedWindow("Hull demo",CV_WINDOW_AUTOSIZE);
	imshow("Hull demo", drawing);
#endif
	return true;
}


bool isRect(const vector<Point> poly)
{
	//cout << "poly.size(): " << poly.size() << endl;
	if (poly.size() == 4)
	{
		//if (poly.size() != 4) return false;
		double k1 = (poly.at(0).y - poly.at(1).y) / (poly.at(0).x - poly.at(1).x + 0.001);
		double k2 = (poly.at(1).y - poly.at(2).y) / (poly.at(1).x - poly.at(2).x + 0.001);
		double k3 = (poly.at(2).y - poly.at(3).y) / (poly.at(2).x - poly.at(3).x + 0.001);
		double k4 = (poly.at(3).y - poly.at(0).y) / (poly.at(3).x - poly.at(0).x + 0.001);
		//std::cout << "k1 - k2: " << k1 - k2 << "		k3 - k4: " << k3 - k4 << endl;
		if (fabs(k1 - k3) > kdiff_threshold && fabs(k2 - k4) > kdiff_threshold)
		{
			return false;
		}
		//std::cout << "k1 - k3: " << k1 - k3 << "		k2 - k4: " << k2 - k4 << endl;
	}
	else
	{
		if (poly.size() < 4) return false;
	}
	return true;
}

void DrawRotatedRect(Mat &image, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness)
{
	cv::Point2f vertex[4];
	rect.points(vertex);
	for (int i = 0; i < 4; i++)
		cv::line(image, vertex[i], vertex[(i + 1) % 4], color, thickness);
}
int filterBoard(Mat image, int cntBoard)
{
	secJudge = true;
	for (int i = 0; i < cntBoard; i++)
	{	
		calcSafeRect(rectCandi[i], frame, safeBoundRect);
		board = frame(Rect(safeBoundRect.x, safeBoundRect.y, safeBoundRect.width, safeBoundRect.height));
		//rotation(board, board, Size(rectCandi[i].size.width, rectCandi[i].size.height),
		//	Point2f(board.cols / 2, board.rows / 2), rectCandi[i].angle);
		
		//board = equalizeHist(board);
		cvtColor(board, board, CV_BGR2GRAY);
		threshold(board, binImage, 0, 255, CV_THRESH_OTSU);

		if (detectBoard(binImage))
		{
			imshow("second board", binImage);
			imshow("board", board);
			cout << "!!!!!!!!!!!!!!!!!!!!!the board[" << i << "]is successed!!!" << endl;
		}
		else cout << "the board[" << i << "]is failed!!!" << endl;
	}
	secJudge = false;
	return 1;
}


/*******************************************************/
//!roi_rect：一个旋转矩形     ！src：矩形所在的原图像    ！safeBoundRect: 一个安全的Rect矩形
//! 计算一个安全的Rect，如果不存在，返回false
bool calcSafeRect(const RotatedRect& roi_rect, const Mat& src, Rect_<float>& safeBoundRect)
{
	Rect_<float> boudRect = roi_rect.boundingRect();
	boudRect.x -= 5; boudRect.y -= 5;
	boudRect.width += 10; boudRect.height += 10;
	// boudRect的左上的x和y有可能小于0
	float tl_x = boudRect.x > 0 ? boudRect.x : 0;
	float tl_y = boudRect.y > 0 ? boudRect.y : 0;
	// boudRect的右下的x和y有可能大于src的范围
	float br_x = boudRect.x + boudRect.width < src.cols ?
		boudRect.x + boudRect.width - 1 : src.cols - 1;
	float br_y = boudRect.y + boudRect.height < src.rows ?
		boudRect.y + boudRect.height - 1 : src.rows - 1;

	float roi_width = br_x - tl_x;
	float roi_height = br_y - tl_y;

	if (roi_width <= 0 || roi_height <= 0)
		return false;

	// 新建一个mat，确保地址不越界，以防mat定位roi时抛异常
	safeBoundRect = Rect_<float>(tl_x, tl_y, roi_width, roi_height);

	return true;
}

//! 旋转操作
//!in：输入要旋转的图像		out：输出被旋转后的图像，输出图像会扩大1.5倍		
//!rect_size：原图的大小		center：原图的中心坐标		angle：需要旋转的角度
bool rotation(Mat& in, Mat& out, const Size rect_size, const Point2f center, double angle)
{
	Mat in_large;
	if (angle > 0) angle = 90.0 - angle;
	else angle = 90.0 + angle;

	in_large.create(in.rows*1.5, in.cols*1.5, in.type());

	int x = in_large.cols / 2 - center.x > 0 ? in_large.cols / 2 - center.x : 0;
	int y = in_large.rows / 2 - center.y > 0 ? in_large.rows / 2 - center.y : 0;

	int width = (x + in.cols) < in_large.cols ? in.cols : in_large.cols - x;
	int height = (y + in.rows) < in_large.rows ? in.rows : in_large.rows - y;

	/*assert(width == in.cols);
	assert(height == in.rows);*/

	if (width != in.cols || height != in.rows)
		return false;

	Mat imageRoi = in_large(Rect(x, y, width, height));
	addWeighted(imageRoi, 0, in, 1, 0, imageRoi);

	//imshow("in_copy", in_large);
	//waitKey(0);

	Point2f center_diff(in.cols / 2, in.rows / 2);
	Point2f new_center(in_large.cols / 2, in_large.rows / 2);

	Mat rot_mat = getRotationMatrix2D(new_center, angle, 1);

	Mat mat_rotated;
	warpAffine(in_large, mat_rotated, rot_mat, Size(in_large.cols, in_large.rows), CV_INTER_CUBIC);

	//imshow("mat_rotated", mat_rotated);
	//waitKey(4000);

	Mat img_crop;
	getRectSubPix(mat_rotated, Size(rect_size.height, rect_size.width), new_center, img_crop);

	out = img_crop;

	//imshow("img_crop", img_crop);
	//waitKey(0);
	return true;
}
bool comp(const cv::RotatedRect &a, const cv::RotatedRect &b)
{
	return(a.size.area() > b.size.area());
}
//用来动态调二值化的阈值
void on_trackbar(int, void*)
{
	//namedWindow("Edge Detection", CV_WINDOW_NORMAL);
	//createTrackbar("threholdMinh: 255", "Edge Detection", &threholdMinh, 255, on_trackbar);
	//createTrackbar("threholdMins: 255", "Edge Detection", &threholdMins, 255, on_trackbar);
	//createTrackbar("threholdMinv: 255", "Edge Detection", &threholdMinv, 255, on_trackbar);
	//createTrackbar("threholdMaxh: 255", "Edge Detection", &threholdMaxh, 255, on_trackbar);
	//createTrackbar("threholdMaxs: 255", "Edge Detection", &threholdMaxs, 255, on_trackbar);
	//createTrackbar("threholdMaxv: 255", "Edge Detection", &threholdMaxv, 255, on_trackbar);
	binaryImage(frame);
}


//绘制直方图，src为输入的图像，histImage为输出的直方图，name是输出直方图的窗口名称
void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name)
{
	const int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	cv::MatND hist;
	int channels[] = { 0 };

	calcHist(&src, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);

	double maxValue;
	minMaxLoc(hist, 0, &maxValue, 0, 0);
	int scale = 1;
	int histHeight = 256;

	for (int i = 0; i < bins; i++)
	{
		float binValue = hist.at<float>(i);
		int height = cvRound(binValue*histHeight / maxValue);
		rectangle(histImage, cv::Point(i*scale, histHeight), cv::Point((i + 1)*scale, histHeight - height), cv::Scalar(255));

		//imshow(name, histImage);
	}
}

Mat equalizeHist(Mat image)
{
	Mat dst;
	Mat srcHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
	Mat dstHistImage = cv::Mat::zeros(256, 256, CV_8UC1);

	Mat YCC;
	cvtColor(image, YCC, CV_RGB2YCrCb);
	vector<cv::Mat> channels;
	split(YCC, channels);

	//drawHistImg(channels[0], srcHistImage, "srcHistImage");
	equalizeHist(channels[0], channels[0]);//对Y通道进行均衡化

	merge(channels, YCC);
	cvtColor(YCC, dst, CV_YCrCb2RGB);//重新转换到RGB颜色域

	//drawHistImg(channels[0], dstHistImage, "dstHistImage");

	return dst;

}





