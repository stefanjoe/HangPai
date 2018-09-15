主体思路：
	1、先不考虑天井（理论上来说，有天井更有利），图像处理主要分为两部分，一部分是识别出KT板，另一部分是识别出数字。
		前者想要从地面识别到KT板最大的困难在于，地面的情况比较复杂，地面物体比较复杂，很容易出现于KT板
		特征相似的物体，还有就是从天上看，KT板上的数字非常小，而且由于图传的问题，会导致数字不是连续的，有时候会断开。
		这是主要的难题（前提是能有一个良好的图像质量，质量越好，处理越方便越准确）。
	2、对于地面的KT板识别，主要由以下几个特征点：
		a.KT板式白色的，相对比较纯
			针对这一点，目前的想法是，先尽量提取出白板，利用KT板纯白色的特征，周围又没有白色的东西紧挨着，可以提取出一个完整的轮廓
		b.KT板的长宽比是固定的，形状是矩形
			这一点很关键，通过长宽比和形状，可以过滤很多的类似物体，对于形状，目前只要求了是平行四边形即可，后期也可以加上一点角度判断
		c.KT板的中间是由数字的
			地面上有很多的白色区域（不一定本身物体是白色的，光线的照射下也有可能出现跟KT一样的形状），但是由于KT板中间有数字，即黑色区域
			因此，可以在轮廓的基础上加上包含的层级关系，这样可以过滤掉上面出现的跟KT板一样的白板
		经过上面三个过滤基本可以筛选出白板，但这时也不一定能保证就是我们需要的KT板，对于第一次筛选出的board，我们可以进行第二次处理，
		降低二值化的阈值，这样一些重新没有那么黑的白板就能被过滤掉，第二阶段的过滤基本步骤少于第一阶段，主要是对于中间黑色的判断，
		也可以适当的判断一下形状
	3、数字的识别，比较难：
		a.收到各种外部条件的影响，天气、图传质量、飞行的路线等都会直接的影响图像的质量，导致数字的识别要更难一些
			尽量保证图像的质量
		b.由于在室外，数字很容易出现断的现象，这对特征提取来说是个大麻烦，所以传统的找特征的方法可能适用性不是太高
			推荐使用深度学习相关的技能，opencv上有现成的分类器，如KNN，在源文件里的OCR.cpp中就用的简单的监督学习，来对数字进行分类
			试着用了一下，对新的图片识别的效果不是太好，但是训练时加上自己的数据集会有所改善，可以收集一下照片多多尝试，难度不是太大
			以后可以为深度学习先入入手
			也可以尝试yoloV3（推荐）、caffe、Tensor* 等，相关的CNN框架的深度学习，难度更高点，但是准确率上也更高。

版本说明：V2.0版本
实现功能：
1、函数：
	void Init(); 
	Mat getImage();
	Mat binaryImage(Mat image);   //对图像进行二值化处理，方便后面提取白板
	bool detectBoard(Mat image);  //主要的提取白板模块
	bool comp(const cv::RotatedRect &a, const cv::RotatedRect &b);
	bool isRect(const vector<Point> poly);  //判断是否是矩形，过滤白板

	void DrawRotatedRect(Mat &image, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness);

	/**************************  新添加函数  **************************/
	int filterBoard(Mat image, int cntBoard);  //第二次过滤白板，提高稳定性
	void on_trackbar(int, void*);   //动态调节HSV或者RGB的阈值

	bool calcSafeRect(const RotatedRect& roi_rect, const Mat& src, Rect_<float>& safeBoundRect);  //计算一个安全距离
	bool rotation(Mat& in, Mat& out, const Size rect_size, const Point2f center, double angle);	  //对倾斜的矩形进行一个矫正
	void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name);	//绘制均衡化之后的图像
	Mat equalizeHist(Mat image);	//对图像进行均衡化操作

待测功能：
	1、HSV对于提取颜色目标来说，比RGB是有优势的，但是这个版本是大多数用RGB进行测试的，HSV二值化的代码也都写在里面了，开启using_hsv就能使用
		具体的效果还需要更多的测试
	2、目前对于白板的还有待完善，现在的筛选分为两部分：detectBoard()，filterBoard()这两个函数，
		detectBoard()中的筛选条件有：
		if (!isRect(convexsHull[i])) continue;   //是否是矩形  
			//isRect()函数里面对四边形的判断还有潜力提升，可以考虑对轮廓进行处理，
			//将轮廓再一步简化为四边形再进行判断对边是否平行以及临边角度是否合适，应该是90度左右
		if (hierarchy[i].val[2] == -1 || hierarchy[i].val[3] != -1) continue;  
			 //因为KT板中间有黑色部分，外面的轮廓下会有一个子轮廓，没有子轮廓就跳过，并且白板本身没有父轮廓
		if (rect.size.area() < 1000 || rect.size.area() > 5000) continue;
			//对面积的限制，这个数据还是要多测试，我只是从视频里面大致选的一个数值，会根据飞行高度和相机焦距有所变化
		if (!(cntArea / rect.size.area() > 0.5			&&	  //比较轮廓和矩形面积的比，理论上来说应该是1:1， 照片0.8左右
			cntArea / rect.size.area() < 1.0
			)) continue;
		if (rectWidth / rectHeight < 1.8 && rectWidth / rectHeight > 1.3) continue;
			//对白板的外接矩形进行一个长宽比的限制，视频里主要是1.5左右，待多测试

		filterBoard()中的筛选条件有：
		1、前面函数会将possibleBoard可能在的区域进行提取，生成一个新的图片，这个函数是在这个图片上进行的
		2、再一次判断是否是白板，此时颜色的阈值会降低，对四边形的判断也更严格
		3、在此之后将会提供一个标准的图片，传给数字识别那部分函数，在ocr.cpp里面


















