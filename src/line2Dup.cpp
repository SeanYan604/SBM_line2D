#include "../include/line2Dup.h"
#include "../include/utils_.h"
#include <iostream>

using namespace std;
using namespace cv;

#include <chrono>

namespace linemod
{
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:
        return 0;
    case 2:
        return 1;
    case 4:
        return 2;
    case 8:
        return 3;
    case 16:
        return 4;
    case 32:
        return 5;
    case 64:
        return 6;
    case 128:
        return 7;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

static Rect cropTemplates(std::vector<Template> &templates)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();

    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }

    /// @todo Why require even min_x, min_y?
    if (min_x % 2 == 1)
        --min_x;
    if (min_y % 2 == 1)
        --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y >> templ.pyramid_level;

        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            templ.features[j].x -= templ.tl_x;   ///   temp中features的x,y是相对与temp的左上角的起始点算的, 其实算偏移量dx,dy
            templ.features[j].y -= templ.tl_y;
        }
    }

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y);   ///  返回一个BoundingBox
}

/****************************************************************************************\
*                                                              Modality interfaces       *
\****************************************************************************************/
bool QuantizedPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                   std::vector<Feature> &features,
                                                   size_t num_features, float distance)   // 选取散点特征
{
    features.clear();
    float distance_sq = distance * distance;
    int i = 0;
    while (features.size() < num_features)
    {
        Candidate c = candidates[i];

        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep)
            features.push_back(c.f);

        if (++i == (int)candidates.size())
        {
            // Start back at beginning, and relax required distance
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
        }
    }
    if (features.size() == num_features)      //   同样散点特征不够的话,也返回false, 并输出没有足够的特征
    {
        return true;
    }
    else
    {
        std::cout << "this templ has no enough features" << std::endl;
        return false;
    }
}

Ptr<Modality> Modality::create(const String& modality_type)
{
  if (modality_type == "ColorGradient")
    return makePtr<ColorGradient>();
  else if (modality_type == "DepthNormal")
    return makePtr<DepthNormal>();
  else
    return Ptr<Modality>();
}

Ptr<Modality> Modality::create(const FileNode& fn)
{
  String type = fn["type"];
  Ptr<Modality> modality = create(type);
  modality->read(fn);
  return modality;
}

void colormap(const Mat& quantized, Mat& dst)
{
  std::vector<Vec3b> lut(8);
  lut[0] = Vec3b(  0,   0, 255);
  lut[1] = Vec3b(  0, 170, 255);
  lut[2] = Vec3b(  0, 255, 170);
  lut[3] = Vec3b(  0, 255,   0);
  lut[4] = Vec3b(170, 255,   0);
  lut[5] = Vec3b(255, 170,   0);
  lut[6] = Vec3b(255,   0,   0);
  lut[7] = Vec3b(255,   0, 170);

  dst = Mat::zeros(quantized.size(), CV_8UC3);
  for (int r = 0; r < dst.rows; ++r)
  {
    const uchar* quant_r = quantized.ptr(r);
    Vec3b* dst_r = dst.ptr<Vec3b>(r);
    for (int c = 0; c < dst.cols; ++c)
    {
      uchar q = quant_r[c];
      if (q)
        dst_r[c] = lut[getLabel(q)];
    }
  }
}
/****************************************************************************************\
*                                                         Color gradient ColorGradient                                                                        *
\****************************************************************************************/

void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,
                        Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);           // 16/360 为alpha比值, convert将angle中的值从0-360转到0-16,并存入quantized中去

    // Zero out top and bottom rows
    /// @todo is this necessary, or even correct?
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);                                  // 对quantized的图像外围一圈像素的内存进行初始化0
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;                                   // 在该循环内进行二进制与的操作
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);  // 使用指针来遍历Mat

        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);   // 从该像素的左上角那个像素开始, 1x3一行一行来进行增加
                histogram[patch3x3_row[0]]++;                                // patch从quantized读到的值为0-7, 试对应的histogram的值++
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();                //其中step1()是该指针指向向量一维的元素个数, patch指针向下移了一行
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;
                histogram[patch3x3_row[1]]++;
                histogram[patch3x3_row[2]]++;

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i)     // 当该像素的mag大于阈值的时候  计算他周边3X3的区域, 进行梯度直方图统计, 查找出计数最多的那个方向, 得到方向序号index
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;         // 设置梯度计数阈值为5   超过5才将该点计数修改为量化的八位二进制形式
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);   //  将1左移index位赋值给该像素的quantized_angle
            }
        }
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,
                                  Mat &angle, float threshold)     // 获取方向量化
{
    Mat smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

    if(src.channels() == 1){                                                     // 输入图像只有一个通道的时候
        Mat sobel_dx, sobel_dy, magnitude, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, sobel_ag, true);                               // 通过dx, dy得到每个像素的角度 sobel_ag   True表示使用角度进行表示
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);   // 对该角度图进行量化, 获取量化后的角度图, 为输入的参数angle, threshold为量化阈值,用来限制magnitude

    }else{

        magnitude.create(src.size(), CV_32F);    // 相当于对magnitude做初始化, 设定大小和数据类型,  magnitude为最后的输出数组

        // Allocate temporary buffers
        Size size = src.size();     //size = [cols, rows]
        Mat sobel_3dx;              // per-channel horizontal derivative
        Mat sobel_3dy;              // per-channel vertical derivative
        Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        Mat sobel_ag;               // final gradient orientation (unquantized)
        
        Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

        short *ptrx = (short *)sobel_3dx.data;
        short *ptry = (short *)sobel_3dy.data;
        float *ptr0x = (float *)sobel_dx.data;
        float *ptr0y = (float *)sobel_dy.data;
        float *ptrmg = (float *)magnitude.data;

        const int length1 = static_cast<const int>(sobel_3dx.step1());           // 三通道图像的step1是cols的3倍
        const int length2 = static_cast<const int>(sobel_3dy.step1());
        const int length3 = static_cast<const int>(sobel_dx.step1());
        const int length4 = static_cast<const int>(sobel_dy.step1());
        const int length5 = static_cast<const int>(magnitude.step1());
        const int length0 = sobel_3dy.cols * 3;

        for (int r = 0; r < sobel_3dy.rows; ++r)
        {
            int ind = 0;

            for (int i = 0; i < length0; i += 3)     // 因为包括3个rgb通道, 所以需要每次加3
            {
                // Use the gradient orientation of the channel whose magnitude is largest
                int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];     // 分别对BGR通道求magnitude
                int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
                int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];

                if (mag1 >= mag2 && mag1 >= mag3)                // 对比三个通道 取梯度最大的这个通道的值, dx 和 dy分别放进ptr0x和ptr0y, 将mag放进ptrmg
                {
                    ptr0x[ind] = ptrx[i];
                    ptr0y[ind] = ptry[i];
                    ptrmg[ind] = (float)mag1;
                }
                else if (mag2 >= mag1 && mag2 >= mag3)
                {
                    ptr0x[ind] = ptrx[i + 1];
                    ptr0y[ind] = ptry[i + 1];
                    ptrmg[ind] = (float)mag2;
                }
                else
                {
                    ptr0x[ind] = ptrx[i + 2];
                    ptr0y[ind] = ptry[i + 2];
                    ptrmg[ind] = (float)mag3;
                }
                ++ind;
            }
            ptrx += length1;             // 计算完一行的数据之后对指针做操作,初始位置向下移动一行
            ptry += length2;
            ptr0x += length3;
            ptr0y += length4;
            ptrmg += length5;
        }

        // Calculate the final gradient orientations
        phase(sobel_dx, sobel_dy, sobel_ag, true);            // 得到了单一通道的梯度方向图, 使用与channel =1时一样的方法进行处理 , 提取关键强梯度信息
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
    }


}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, weak_threshold);
}

void ColorGradientPyramid::pyrDown()              // 对图像进行降采样, 边长缩小到一半,同时把mask进行缩小
{
    // Some parameters need to be adjusted
    num_features /= 2; /// @todo Why not 4?
    ++pyramid_level;

    // Downsample the current inputs
    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;

    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }

    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask);
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    // Want features on the border to distinguish from background
    Mat local_mask;
    if (!mask.empty())
    {
        erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
//        subtract(mask, local_mask, local_mask);
    }

    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();
    float threshold_sq = strong_threshold * strong_threshold;

    for (int r = 0; r < magnitude.rows; ++r)
    {
        const uchar* angle_r = angle.ptr<uchar>(r);
        const float* magnitude_r = magnitude.ptr<float>(r);
        const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

        for (int c = 0; c < magnitude.cols; ++c)
        {
        if (no_mask || mask_r[c])
        {
            uchar quantized = angle_r[c];
            if (quantized > 0)
            {
            float score = magnitude_r[c];
            if (score > threshold_sq)
            {
                candidates.push_back(Candidate(c, r, getLabel(quantized), score));
            }
            }
        }
        }
    }

    // int nms_kernel_size = 5;
    // cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));

    // for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r)
    // {
    //     const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

    //     for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c)
    //     {
    //         if (no_mask || mask_r[c])
    //         {
    //             float score = 0;
    //             if(magnitude_valid.at<uchar>(r, c)>0){
    //                 score = magnitude.at<float>(r, c);
    //                 bool is_max = true;
    //                 for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){    ///  对图像做了一次5X5区域内的非极大抑制
    //                     for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
    //                         if(r_offset == 0 && c_offset == 0) continue;

    //                         if(score < magnitude.at<float>(r+r_offset, c+c_offset)){
    //                             score = 0;
    //                             is_max = false;
    //                             break;
    //                         }
    //                     }
    //                 }

    //                 if(is_max){                                                                           ///  如果该像素是最大的, 则对mgnitude_valid把边上的值进行清零
    //                     for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++){
    //                         for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++){
    //                             if(r_offset == 0 && c_offset == 0) continue;
    //                             magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
    //                         }
    //                     }
    //                 }
    //             }

    //             if (score > threshold_sq && angle.at<uchar>(r, c) > 0)            /// 如果该点的mag大于thresh且存在梯度, 则把该点标记为一个candidate
    //             {
    //                 candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));   //  getLabel将angle的二进制转换为0-7的编号
    //             }
    //         }
    //     }
    // }
    // We require a certain number of features
    if (candidates.size() < num_features)                  ///  如果提取的candidates数量小于要求的特征点数, 则返回false 此时的addTemplate失败
        return false;
    // NOTE: Stable sort to agree with old code, which used std::list::sort()
    std::stable_sort(candidates.begin(), candidates.end());

    // Use heuristic(启发式的) based on surplus(剩余的) of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }

    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;

    return true;
}

ColorGradient::ColorGradient()
    : weak_threshold(10.0f),      ///  ori : 10.0f
      num_features(63),
      strong_threshold(55.0f)    ///  ori : 55.0f
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";

cv::String ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);

    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}
/****************************************************************************************\
*                                                                 Response maps                                                                                    *
\****************************************************************************************/

static void orUnaligned8u(const uchar *src, const int src_stride,      //  src为一个像素, src_stride为图一行的像素数
                          uchar *dst, const int dst_stride,            // width, height为 图像中以src这个像素为左上角点的矩形的宽高, 直接到图像边缘
                          const int width, const int height)
{
    for (int r = 0; r < height; ++r)
        {
            int c = 0;

            // not aligned, which will happen because we move 1 bytes a time for spreading
            while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) {
                dst[c] |= src[c];
                c++;
            }


            for(; c<width; c++)
                dst[c] |= src[c];

            // Advance to next row
            src += src_stride;
            dst += dst_stride;
    }
}

static void spread(const Mat &src, Mat &dst, int T)
{
    // Allocate and zero-initialize spread (OR'ed) image
    dst = Mat::zeros(src.size(), CV_8U);

    // Fill in spread gradient image (section 2.3)
    for (int r = 0; r < T; ++r)
    {
        int height = src.rows - r;
        for (int c = 0; c < T; ++c)
        {
            orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
                          static_cast<const int>(dst.step1()), src.cols - c, height);
        }
    }
}

// 1,2-->0 3-->1
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4};

static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);

    // Allocate response maps
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)   // 这两个循环把src中的值分为8为的低四位和高四位来进行存储,分别为lsb和msb
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);

        for (int c = 0; c < src.cols; ++c)
        {
            // Least significant 4 bits of spread image pixel
            lsb4_r[c] = src_r[c] & 15;
            // Most significant 4 bits, right-shifted to be in [0, 16)
            msb4_r[c] = (src_r[c] & 240) >> 4;
        }
    }

    // For each of the 8 quantized orientations...
    for (int ori = 0; ori < 8; ++ori)
    {
        uchar *map_data = response_maps[ori].ptr<uchar>();
        uchar *lsb4_data = lsb4.ptr<uchar>();
        uchar *msb4_data = msb4.ptr<uchar>();
        const uchar *lut_low = SIMILARITY_LUT + 32 * ori;  //SIMILARITY_LUT为相似度数组, 高位低位各16位,分别表示0000-1111,从0方向到7方向,对应位为1则计分为4,邻近方向计分为1,其余计分为0,相当于table查表了
        const uchar *lut_hi = lut_low + 16;

        for (int i = 0; i < src.rows * src.cols; ++i)
        {
            map_data[i] = std::max(lut_low[lsb4_data[i]], lut_hi[msb4_data[i]]); // 求出src低4位和高4位对应table中分数的max
        }
    }

}

static void linearize(const Mat &response_map, Mat &linearized, int T)
{
    CV_Assert(response_map.rows % T == 0);
    CV_Assert(response_map.cols % T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    
    linearized.create(T * T, mem_width * mem_height, CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
    {
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;

            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T)
                    *memory++ = response_data[c];
            }
        }
    }
}
/****************************************************************************************\
*                                                             Linearized similarities                                                                    *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,
                                               const Feature &f, int T, int W)    //该函数直接返回在线性表中, 对应目标模板位置的值
{
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T.
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;
}

static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,
                       Mat &dst, Size size, int T)
{
    // we only have one modality, so 8192*2
    CV_Assert(templ.features.size() < 16384);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    int template_positions = span_y * W + span_x + 1; // why add 1?

    dst = Mat::zeros(H, W, CV_16U);
    short *dst_ptr = dst.ptr<short>();

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {

        Feature f = templ.features[i];

        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;
#if CV_SSE2
        if (haveSSE2)
        {
            __m128i const zero = _mm_setzero_si128();
            // Fall back to MOVDQU
            for (; j < template_positions - 7; j += 8)   // 偏移步长为8
            {
                __m128i responses = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(lm_ptr + j));
                __m128i *dst_ptr_sse = reinterpret_cast<__m128i *>(dst_ptr + j);
                responses = _mm_unpacklo_epi8(responses, zero);
                *dst_ptr_sse = _mm_add_epi16(*dst_ptr_sse, responses);
            }
        }
        else
#endif
        for (; j < template_positions; ++j)                      // 可注释可留  影响不大
            dst_ptr[j] = short(dst_ptr[j] + short(lm_ptr[j]));
    }
}

static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    CV_Assert(templ.features.size() < 16384);

    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);

    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
    __m128i *dst_ptr_sse = dst.ptr<__m128i>();
#endif

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;                  /// 从点center开始进行模板滑窗

        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
#if CV_SSE2
        if (haveSSE2)
        {
            __m128i const zero = _mm_setzero_si128();
            for (int row = 0; row < 16; ++row)     /// 这个是要取一个块的相似度值
            {
                __m128i aligned_low = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(lm_ptr));
                __m128i aligned_high = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(lm_ptr + 8));
                aligned_low = _mm_unpacklo_epi8(aligned_low, zero);
                aligned_high = _mm_unpacklo_epi8(aligned_high, zero);
                dst_ptr_sse[2 * row] = _mm_add_epi16(dst_ptr_sse[2 * row], aligned_low);
                dst_ptr_sse[2 * row + 1] = _mm_add_epi16(dst_ptr_sse[2 * row + 1], aligned_high);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
        {
            short *dst_ptr = dst.ptr<short>();
            for (int row = 0; row < 16; ++row)
            {
                for (int col = 0; col < 16; ++col)
                    dst_ptr[col] = short(dst_ptr[col] + short(lm_ptr[col]));
                dst_ptr += 16;
                lm_ptr += W;
            }
        }
    }
}

static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
                          Mat &dst, Size size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() <= 63);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;    // 为了保证划窗在不溢出, 需要算多一个划窗位, 要么能被T整除,要么多一个
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;   // 模板可以滑动的空间 x,y,   尽量少了一个窗口位
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?    
    
    //   加一是包含第一个位置, 这个position是计算所有可以移动temp的位置个数, 
    //   因为是线性表, 每一行的长度都是W*H, 而最开始的模板在W方向上已经占了wf个长度了,H方向上占了hf长度,  对于总长度W*H来说是缩短了 (hf - 1)*W + wf, 
    //   W*H - (hf - 1)*W - wf = W*(H - hf) + W - wf = W*span_y + span_x   (再加一是包含起始位置)

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    //  Nope, due to the linear structure, it's more reasonable to be a buffer, 
    //     rather than a (Mat), it is a little different from a convolutional operation.

    dst = Mat::zeros(H, W, CV_8U);
    uchar *dst_ptr = dst.ptr<uchar>();

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;
#if CV_SSE2
            if (haveSSE2)
        {
            // Fall back to MOVDQU
            for (; j < template_positions - 15; j += 16)  ///  18为temp滑动的步长
            {
                __m128i responses = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lm_ptr + j));
                __m128i *dst_ptr_sse = reinterpret_cast<__m128i *>(dst_ptr + j);
                *dst_ptr_sse = _mm_add_epi8(*dst_ptr_sse, responses);
            }
        }
        else
#endif
        for (; j < template_positions; ++j)                  // 可注释可留  影响不大
            dst_ptr[j] = uchar(dst_ptr[j] + lm_ptr[j]);
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() <= 63);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
    volatile bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
    volatile bool haveSSE3 = checkHardwareSupport(CV_CPU_SSE3);
#endif
    __m128i *dst_ptr_sse = dst.ptr<__m128i>();
#endif

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
#if CV_SSE2
#if CV_SSE3
        if (haveSSE3)
        {
            // LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(lm_ptr));
                dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
            if (haveSSE2)
        {
            // Fall back to MOVDQU
            for (int row = 0; row < 16; ++row)
            {
                __m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lm_ptr));
                dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
                lm_ptr += W; // Step to next row
            }
        }
        else
#endif
        {
            uchar *dst_ptr = dst.ptr<uchar>();
            for (int row = 0; row < 16; ++row)
            {
                for (int col = 0; col < 16; ++col)
                    dst_ptr[col] = uchar(dst_ptr[col] + lm_ptr[col]);
                dst_ptr += 16;
                lm_ptr += W;
            }
        }
    }
}

/****************************************************************************************\
*                                                             High-level Detector API                                                                    *
\****************************************************************************************/

Detector::Detector()
{
    this->detect_mode = "Line2D";
    this->CG_modality = makePtr<ColorGradient>();
    pyramid_levels = 2;
    T_at_level.push_back(5);
    T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T, std::string mode)
{
    this->detect_mode = mode;
    this->CG_modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T, std::string mode)
{
    this->detect_mode = mode;
    this->CG_modality = makePtr<ColorGradient>(10.0f, num_features, 55.0f);
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(const std::vector< cv::Ptr<Modality> >& modalities, const std::vector<int>& T_pyramid, std::string mode)
{
    this->modalities = modalities;
    this->detect_mode = mode;
    pyramid_levels = T_pyramid.size();
    T_at_level = T_pyramid;
}

std::vector<Match> Detector::match(std::vector<Mat>& sources, float threshold,
                                   const std::vector<std::string> &class_ids, const Mat mask) const
{
    // Timer timer;
    std::vector<Match> matches;    //  定义最后得到的matches向量,   输入source待处理图片, threshold阈值, ids类的名称,  mask为一个空的cv::Mat

    // Initialize each ColorGradient with our sources
    std::vector< Ptr<QuantizedPyramid> > quantizers;     //  量化图像的vector
    if(detect_mode == "Line2D")
        quantizers.push_back(CG_modality->process(sources[0], mask));   // 对source和mask做整合,开始构建ColorGradientPyramid, 开始对图像做量化了,返回量化后的CGP列表
    else
    {
        for (int i = 0; i < (int)modalities.size(); ++i){
            Mat mask, source;
            source = sources[i];
            CV_Assert(mask.empty() || mask.size() == source.size());
            quantizers.push_back(modalities[i]->process(source, mask));
        }
    }

    
    // pyramid level -> ColorGradient -> quantization
    int mod_size;
    if(detect_mode == "Line2D")
        mod_size = 1;
    else
        mod_size = modalities.size();

    
    LinearMemoryPyramid lm_pyramid(pyramid_levels,
                                   std::vector<LinearMemories>(mod_size, LinearMemories(8)));   // 定义线性金字塔空间,第一维是金字塔的层数,第二维为1,第三维为8,  这8个分别为8个方向向量的模板

    // For each pyramid level, precompute linear memories for each ColorGradient
    std::vector<Size> sizes;
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T = T_at_level[l];
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];
        
        if (l > 0)
        {
            for (int i = 0; i < (int)quantizers.size(); ++i)
                quantizers[i]->pyrDown();
        }

        Mat quantized, spread_quantized;
        std::vector<Mat> response_maps;
        for (int i = 0; i < (int)quantizers.size(); ++i)
        {
            quantizers[i]->quantize(quantized);
            spread(quantized, spread_quantized, T);
            computeResponseMaps(spread_quantized, response_maps);

            LinearMemories &memories = lm_level[i];
            for (int j = 0; j < 8; ++j)
                linearize(response_maps[j], memories[j], T);
        }

        sizes.push_back(quantized.size());
    }

    // timer.out("construct response map");

    if (class_ids.empty())
    {
        // Match all templates
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it)
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
    else
    {
        // Match only templates for the requested class IDs
        for (int i = 0; i < (int)class_ids.size(); ++i)
        {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
    }

    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement

    std::sort(matches.begin(), matches.end());    // 先对matches 按similarity排序, 去重标准三步骤
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());  ///  通过unique对相邻的相同元素去重,并将重复的移到vector最后面,返回的是去重好的最后一位的位置指针
    matches.erase(new_end, matches.end());                        ///  将去重后的最后一个元素地址一直到vector的最后删去
 


    // timer.out("templ match");

    return matches;
}

// Used to filter out weak matches
struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,   // 该函数实现Class的匹配
                          const std::vector<Size> &sizes,
                          float threshold, std::vector<Match> &matches,
                          const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids) const
{
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)
    {
        const TemplatePyramid &tp = template_pyramids[template_id];
        // First match over the whole image at the lowest pyramid level
        /// @todo Factor this out into separate function
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();

        // Compute similarity maps for each ColorGradient at lowest pyramid level
        Mat similarities;
        float angle = tp.first[0];
        float scale = tp.first[1];
        int lowest_start = static_cast<int>(tp.second.size() - 1);     // 计算金字塔的层数,从最上面一层开始
        int lowest_T = T_at_level.back();
        int num_features = 0;
        int feature_64 = -1;
        {
            const Template &templ = tp.second[lowest_start];
            num_features += static_cast<int>(templ.features.size());
            if (feature_64 <= 0)
            {
                if (templ.features.size() < 64)
                {
                    feature_64 = 1;
                }
                else if (templ.features.size() < 16384)
                {
                    feature_64 = 2;
                }
            }
            if (feature_64 == 1)
            {
                similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
            }
            else if (feature_64 == 2)
            {
                similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
            }
        }

        if (feature_64 == 1)
        {
            similarities.convertTo(similarities, CV_16U);
        }

        // Find initial matches
        std::vector<Match> candidates;
        for (int r = 0; r < similarities.rows; ++r)
        {
            ushort *row = similarities.ptr<ushort>(r);
            for (int c = 0; c < similarities.cols; ++c)
            {
                int raw_score = row[c];
                float score = (raw_score * 100.f) / (4 * num_features);   ///  对score进行统计 

                if (score > threshold)    //这个threshold传进来是90
                {
                    int offset = lowest_T / 2 + (lowest_T % 2 - 1);   /// 这个偏移是给定的, 定在T滑块的中间位置
                    int x = c * lowest_T + offset;                    /// 通过T的大小进行反算回去坐标
                    int y = r * lowest_T + offset;
                    candidates.push_back(Match(x, y, angle, scale, score, class_id, static_cast<int>(template_id)));
                }
            }
        }

        // Locally refine each match by marching up the pyramid   (如果只有一层金字塔,则无需这一步骤)
        for (int l = pyramid_levels - 2; l >= 0; --l)
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l];
            int T = T_at_level[l];
            int start = static_cast<int>(l);
            Size size = sizes[l];
            int border = 8 * T;
            int offset = T / 2 + (T % 2 - 1);
            int max_x = size.width - tp.second[start].width - border;
            int max_y = size.height - tp.second[start].height - border;

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; /// @todo Support other pyramid distance
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);            ///       边缘上留一个8*T的边框

                // Compute local similarity maps for each ColorGradient
                int numFeatures = 0;
                feature_64 = -1;
                {
                    const Template &templ = tp.second[start];
                    numFeatures += static_cast<int>(templ.features.size());
                    if (feature_64 <= 0)
                    {
                        if (templ.features.size() < 64)
                        {
                            feature_64 = 1;
                        }
                        else if (templ.features.size() < 16384)
                        {
                            feature_64 = 2;
                        }
                    }
                    if (feature_64 == 1)
                    {
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                    }
                    else if (feature_64 == 2)
                    {
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));        ///上一层金字塔是对特征点操作, 这次是传点进去,对点周围的一块区域进行匹配
                    }
                }

                if (feature_64 == 1)
                {
                    similarities2.convertTo(similarities2, CV_16U);
                }

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r)             /// 对提取出的块, 计算其中相似度最高的点
                {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c)
                    {
                        int score_int = row[c];
                        float score = (score_int * 100.f) / (4 * numFeatures);

                        if (score > best_score)
                        {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                    }
                }
                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
            }

            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
                                                                  MatchPredicate(threshold));    /// similarity < threshold 为true 则将要删的元素移到后面,返回不要删的最后一个元素的地址
            candidates.erase(new_end, candidates.end());
        }
        matches.insert(matches.end(), candidates.begin(), candidates.end()); ///最后将得到的candidate插入matches的最后, 再到下一个模板的匹配
    }
}

int Detector::addTemplate(const Mat source, const std::string &class_id,
                          const Mat &object_mask, float angle, float scale, int num_features)
{
    int num_modalities = 1;
    // if(detect_mode == "LineMod")
    //     int num_modalities = static_cast<int>(modalities.size());
        
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.first.push_back (angle);
    tp.first.push_back (scale);
    tp.second.resize(pyramid_levels);

    for(int i = 0; i < num_modalities; i++)
    {
        // Extract a template at each pyramid level
        Ptr<QuantizedPyramid> qp;
        if(detect_mode == "Line2D")
            qp = CG_modality->process(source, object_mask);
        else
            qp = modalities[i]->process(source, object_mask);

        if(num_features > 0)
        qp->num_features = num_features;

        for (int l = 0; l < pyramid_levels; ++l)
        {
            /// @todo Could do mask subsampling here instead of in pyrDown()
            if (l > 0)
                qp->pyrDown();

            bool success = qp->extractTemplate(tp.second[l]);
            if (!success)
                return -1;
        }
    }
    //    Rect bb =
    cropTemplates(tp.second);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}

int Detector::addTemplate (const std::vector<cv::Mat>& sources, const std::string& class_id,
                                        const cv::Mat& object_mask, float angle, float scale, int num_features)
{
    int num_modalities = static_cast<int>(modalities.size());
    std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());

    TemplatePyramid tp;
    tp.first.push_back (angle);
    tp.first.push_back (scale);
    tp.second.resize(num_modalities * pyramid_levels);

    // For each modality...
    for (int i = 0; i < num_modalities; i++)
    {   
        cout << "num_mod :" << i << "  " << num_modalities <<endl;
        imshow("src", sources[i]);

        // Extract a template at each pyramid level
        Ptr<QuantizedPyramid> qp = modalities[i]->process(sources[i], object_mask);
        for (int l = 0; l < pyramid_levels; ++l)
        {
        /// @todo Could do mask subsampling here instead of in pyrDown()
        if (l > 0)
            qp->pyrDown();

        bool success = qp->extractTemplate(tp.second[l*num_modalities + i]);
        if (!success)
            return -1;
        }
    }
    cropTemplates(tp.second);

    /// @todo Can probably avoid a copy of tp here with swap
    template_pyramids.push_back(tp);
    return template_id;
}


int Detector::addSyntheticTemplate(const std::vector<Template>& templates, const std::string& class_id)
{
  std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
  int template_id = static_cast<int>(template_pyramids.size());
  std::vector<float> ang_scal = {0,0};
  TemplatePyramid new_tp;
  new_tp.first = ang_scal;
  new_tp.second = templates;

  template_pyramids.push_back(new_tp);
  return template_id;
}

const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id].second;
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    if(detect_mode == "Line2D")
        CG_modality = makePtr<ColorGradient>();
    else
    {
        modalities.clear();
        FileNode modalities_fn = fn["modalities"];
        FileNodeIterator it = modalities_fn.begin(), it_end = modalities_fn.end();
        for ( ; it != it_end; ++it)
        {
            modalities.push_back(Modality::create(*it));
        }
    }
    
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    if(detect_mode == "Line2D")
        CG_modality->write(fs);
    else
    {
        fs << "modalities" << "[";
        for (int i = 0; i < (int)modalities.size(); ++i)
        {
            fs << "{";
            modalities[i]->write(fs);
            fs << "}";
        }
        fs << "]"; // modalities
    }
    
}

void Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    if(detect_mode == "LineMod")
    {
          // Verify compatible with Detector settings
        FileNode mod_fn = fn["modalities"];
        CV_Assert(mod_fn.size() == modalities.size());
        FileNodeIterator mod_it = mod_fn.begin(), mod_it_end = mod_fn.end();
        int i = 0;
        for ( ; mod_it != mod_it_end; ++mod_it, ++i)
            CV_Assert(modalities[i]->name() == (String)(*mod_it));
        CV_Assert((int)fn["pyramid_levels"] == pyramid_levels);
    }
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        tps[template_id].first.push_back((*tps_it)["template_angle"]);   /// 在这里增加读入时候的template的angle 和scale
        tps[template_id].first.push_back((*tps_it)["template_scale"]);

        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].second.resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id].second[idx++].read(*templ_it);
        }
    }

    class_templates.insert(v);
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    if(detect_mode == "LineMod")
    {
        fs << "modalities" << "[:";
        for (size_t i = 0; i < modalities.size(); ++i)
            fs << modalities[i]->name(); 
        fs << "]"; // modalities  
    }
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "template_angle" << tp.first[0];
        fs << "template_scale" << tp.first[1];
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.second.size(); ++j)
        {
            fs << "{";
            tp.second[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,
                           const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}




/// with normal

/****************************************************************************************\
*                                                             Depth normal modality      *
\****************************************************************************************/

// Contains GRANULARITY and NORMAL_LUT
#include "../include/normal_lut.i"

static void accumBilateral(long delta, long i, long j, long * A, long * b, int threshold)
{
  long f = std::abs(delta) < threshold ? 1 : 0;

  const long fi = f * i;
  const long fj = f * j;

  A[0] += fi * i;
  A[1] += fi * j;
  A[3] += fj * j;
  b[0]  += fi * delta;
  b[1]  += fj * delta;
}

/**
 * \brief Compute quantized normal image from depth image.
 *
 * Implements section 2.6 "Extension to Dense Depth Sensors."
 *
 * \param[in]  src  The source 16-bit depth image (in mm).
 * \param[out] dst  The destination 8-bit image. Each bit represents one bin of
 *                  the view cone.
 * \param distance_threshold   Ignore pixels beyond this distance.
 * \param difference_threshold When computing normals, ignore contributions of pixels whose
 *                             depth difference with the central pixel is above this threshold.
 *
 * \todo Should also need camera model, or at least focal lengths? Replace distance_threshold with mask?
 */
static void quantizedNormals(const Mat& src, Mat& dst, int distance_threshold,
                      int difference_threshold)
{
  dst = Mat::zeros(src.size(), CV_8U);

  const unsigned short * lp_depth   = src.ptr<ushort>();
  unsigned char  * lp_normals = dst.ptr<uchar>();

  const int l_W = src.cols;
  const int l_H = src.rows;

  const int l_r = 5; // used to be 7
  const int l_offset0 = -l_r - l_r * l_W;
  const int l_offset1 =    0 - l_r * l_W;
  const int l_offset2 = +l_r - l_r * l_W;
  const int l_offset3 = -l_r;
  const int l_offset4 = +l_r;
  const int l_offset5 = -l_r + l_r * l_W;
  const int l_offset6 =    0 + l_r * l_W;
  const int l_offset7 = +l_r + l_r * l_W;

  const int l_offsetx = GRANULARITY / 2;
  const int l_offsety = GRANULARITY / 2;

  for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
  {
    const unsigned short * lp_line = lp_depth + (l_y * l_W + l_r);
    unsigned char * lp_norm = lp_normals + (l_y * l_W + l_r);

    for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
    {
      long l_d = lp_line[0];

      if (l_d < distance_threshold)
      {
        // accum
        long l_A[4]; l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
        long l_b[2]; l_b[0] = l_b[1] = 0;
        accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset1] - l_d,    0, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset3] - l_d, -l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset4] - l_d, +l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset6] - l_d,    0, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

        // solve
        long l_det =  l_A[0] * l_A[3] - l_A[1] * l_A[1];
        long l_ddx =  l_A[3] * l_b[0] - l_A[1] * l_b[1];
        long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

        /// @todo Magic number 1150 is focal length? This is something like
        /// f in SXGA mode, but in VGA is more like 530.
        float l_nx = static_cast<float>(1150 * l_ddx);
        float l_ny = static_cast<float>(1150 * l_ddy);
        float l_nz = static_cast<float>(-l_det * l_d);

        float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

        if (l_sqrt > 0)
        {
          float l_norminv = 1.0f / (l_sqrt);

          l_nx *= l_norminv;
          l_ny *= l_norminv;
          l_nz *= l_norminv;

          //*lp_norm = fabs(l_nz)*255;

          int l_val1 = static_cast<int>(l_nx * l_offsetx + l_offsetx);
          int l_val2 = static_cast<int>(l_ny * l_offsety + l_offsety);
          int l_val3 = static_cast<int>(l_nz * GRANULARITY + GRANULARITY);

          *lp_norm = NORMAL_LUT[l_val3][l_val2][l_val1];
        }
        else
        {
          *lp_norm = 0; // Discard shadows from depth sensor
        }
      }
      else
      {
        *lp_norm = 0; //out of depth
      }
      ++lp_line;
      ++lp_norm;
    }
  }
  medianBlur(dst, dst, 5);
}

DepthNormalPyramid::DepthNormalPyramid(const Mat& src, const Mat& _mask,
                                       int distance_threshold, int difference_threshold, size_t _num_features,
                                       int _extract_threshold)
  : mask(_mask),
    pyramid_level(0),
    num_features(_num_features),
    extract_threshold(_extract_threshold)
{
   quantizedNormals(src, normal, distance_threshold, difference_threshold);
}

void DepthNormalPyramid::pyrDown()
{
  // Some parameters need to be adjusted
  num_features /= 2; /// @todo Why not 4?
  extract_threshold /= 2;
  ++pyramid_level;

  // In this case, NN-downsample the quantized image
  Mat next_normal;
  Size size(normal.cols / 2, normal.rows / 2);
  resize(normal, next_normal, size, 0.0, 0.0, INTER_NEAREST);
  normal = next_normal;
  if (!mask.empty())
  {
    Mat next_mask;
    resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
    mask = next_mask;
  }
}

void DepthNormalPyramid::quantize(Mat& dst) const
{
  dst = Mat::zeros(normal.size(), CV_8U);
  normal.copyTo(dst, mask);
}

bool DepthNormalPyramid::extractTemplate(Template& templ) const
{
  // Features right on the object border are unreliable
  Mat local_mask;
  if (!mask.empty())
  {
    erode(mask, local_mask, Mat(), Point(-1,-1), 2, BORDER_REPLICATE);
  }

  // Compute distance transform for each individual quantized orientation
  Mat temp = Mat::zeros(normal.size(), CV_8U);
  Mat distances[8];
  for (int i = 0; i < 8; ++i)
  {
    temp.setTo(1 << i, local_mask);
    bitwise_and(temp, normal, temp);
    // temp is now non-zero at pixels in the mask with quantized orientation i
    distanceTransform(temp, distances[i], DIST_C, 3);
  }

  // Count how many features taken for each label
  int label_counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create sorted list of candidate features
  std::vector<Candidate> candidates;
  bool no_mask = local_mask.empty();
  for (int r = 0; r < normal.rows; ++r)
  {
    const uchar* normal_r = normal.ptr<uchar>(r);
    const uchar* mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);

    for (int c = 0; c < normal.cols; ++c)
    {
      if (no_mask || mask_r[c])
      {
        uchar quantized = normal_r[c];

        if (quantized != 0 && quantized != 255) // background and shadow
        {
          int label = getLabel(quantized);

          // Accept if distance to a pixel belonging to a different label is greater than
          // some threshold. IOW, ideal feature is in the center of a large homogeneous
          // region.
          float score = distances[label].at<float>(r, c);
          if (score >= extract_threshold)
          {
            candidates.push_back( Candidate(c, r, label, score) );
            ++label_counts[label];
          }
        }
      }
    }
  }
  // We require a certain number of features
  if (candidates.size() < num_features)
    return false;

  // Prefer large distances, but also want to collect features over all 8 labels.
  // So penalize labels with lots of candidates.
  for (size_t i = 0; i < candidates.size(); ++i)
  {
    Candidate& c = candidates[i];
    c.score /= (float)label_counts[c.f.label];
  }
  std::stable_sort(candidates.begin(), candidates.end());

  // Use heuristic based on object area for initial distance threshold
  float area = no_mask ? (float)normal.total() : (float)countNonZero(local_mask);
  float distance = sqrtf(area) / sqrtf((float)num_features) + 1.5f;
  selectScatteredFeatures(candidates, templ.features, num_features, distance);

  // Size determined externally, needs to match templates for other modalities
  templ.width = -1;
  templ.height = -1;
  templ.pyramid_level = pyramid_level;

  return true;
}

DepthNormal::DepthNormal()
  : distance_threshold(2000),
    difference_threshold(50),
    num_features(63),
    extract_threshold(2)
{
}

DepthNormal::DepthNormal(int _distance_threshold, int _difference_threshold, size_t _num_features,
                         int _extract_threshold)
  : distance_threshold(_distance_threshold),
    difference_threshold(_difference_threshold),
    num_features(_num_features),
    extract_threshold(_extract_threshold)
{
}

static const char DN_NAME[] = "DepthNormal";

cv::String DepthNormal::name() const
{
  return DN_NAME;
}

void DepthNormal::read(const FileNode& fn)
{
  String type = fn["type"];
  CV_Assert(type == DN_NAME);

  distance_threshold = fn["distance_threshold"];
  difference_threshold = fn["difference_threshold"];
  num_features = int(fn["num_features"]);
  extract_threshold = fn["extract_threshold"];
}

void DepthNormal::write(FileStorage& fs) const
{
  fs << "type" << DN_NAME;
  fs << "distance_threshold" << distance_threshold;
  fs << "difference_threshold" << difference_threshold;
  fs << "num_features" << int(num_features);
  fs << "extract_threshold" << extract_threshold;
}

static const int T_DEFAULTS[] = {5, 8};

// Ptr<Detector> getDefaultLINE()
// {
//   std::vector< Ptr<Modality> > modalities;
//   modalities.push_back(makePtr<ColorGradient>());
//   return makePtr<Detector>(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
// }

Ptr<Detector> getDefaultLINEMOD( std::string mode)
{
  std::vector< Ptr<Modality> > modalities;
  modalities.push_back(makePtr<ColorGradient>());
  modalities.push_back(makePtr<DepthNormal>());
  return makePtr<Detector>(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2), mode);
}


} // namespace linemod