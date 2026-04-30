#include <iostream>
#include <filesystem>
#include <onnxruntime_cxx_api.h>
// #include <vector>
#include <opencv2/opencv.hpp>
extern "C" {
#include <libswscale/swscale.h>
}
int fileExists(const std::string& path){
    if (path.empty()) {
        std::cerr << "Error: Path is null." << std::endl;
        return -1;
    }

    std::filesystem::path filePath(path);

    if (!std::filesystem::exists(filePath)) {
        std::cerr << "Error: File does not exist at path: " << path << std::endl;
        return -1;
    }

    if (!std::filesystem::is_regular_file(filePath)) {
        std::cerr << "Error: Path is not a regular file: " << path << std::endl;
        return -1;
    }
    return 0;
}

cv::Mat BgrMatToPaddedRgbMat(SwsContext* sws_ctx, AVFrame* frame, int pad_w, int pad_h) {
    //// 1. 原始尺寸
    int src_w = frame->width;
    int src_h = frame->height;

    //// 2. 计算填充后的尺寸（32 的倍数）
    //int align = 32;
    //pad_w = ((src_w + align - 1) / align) * align;
    //pad_h = ((src_h + align - 1) / align) * align;

    // 3. 初始化格式转换上下文 (YUV420P -> RGB24)，保持原始尺寸，不缩放
    //SwsContext* sws_ctx = sws_getContext(
    //    src_w, src_h, (AVPixelFormat)frame->format,
    //    src_w, src_h, AV_PIX_FMT_RGB24,
    //    SWS_BILINEAR, nullptr, nullptr, nullptr
    //);
    if (!sws_ctx) {
        throw std::runtime_error("Cannot initialize SwsContext");
    }

    // 4. 分配存储 RGB 数据的缓冲区（原始尺寸）
    cv::Mat rgb_frame(src_h, src_w, CV_8UC3);

    // 5. 准备转换所需的指针
    uint8_t* dest_data[1] = { rgb_frame.data };
    int dest_linesize[1] = { src_w * 3 };

    // 6. 执行像素格式转换（YUV -> RGB）
    sws_scale(sws_ctx, frame->data, frame->linesize, 0, src_h,
        dest_data, dest_linesize);
    // sws_freeContext(sws_ctx);

    // 创建填充后的图像（pad_h x pad_w，黑色背景）
    //cv::Mat padded_frame(pad_h, pad_w, CV_8UC3, cv::Scalar(0, 0, 0));
    //rgb_frame.copyTo(padded_frame(cv::Rect(0, 0, src_w, src_h)));
    // 优化：创建填充黑图，然后将原图拷贝到黑图左上角
    //Padding (使用 OpenCV copyMakeBorder，比手动创建 Mat + copyTo 更快且内存连续)
    cv::Mat padded_frame;
    cv::copyMakeBorder(rgb_frame, padded_frame, 0, pad_h - src_h, 0, pad_w - src_w, 
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // 调整 out_data 大小并填充数据 (CHW, float [0,1])
    // out_data.resize(3 * pad_h * pad_w);
    // for (int h = 0; h < pad_h; ++h) {
    //     for (int w = 0; w < pad_w; ++w) {
    //         cv::Vec3b pixel = padded_frame.at<cv::Vec3b>(h, w);
    //         float r = pixel[0] / 255.0f;
    //         float g = pixel[1] / 255.0f;
    //         float b = pixel[2] / 255.0f;

    //         // CHW 索引
    //         out_data[(0 * pad_h + h) * pad_w + w] = r;
    //         out_data[(1 * pad_h + h) * pad_w + w] = g;
    //         out_data[(2 * pad_h + h) * pad_w + w] = b;
    //     }
    // }
    // 优化： Normalize & Convert to CHW Float (关键优化点)
    // 避免逐像素循环，使用 OpenCV 批量操作
    //out_data.resize(size_t(3) * pad_h * pad_w);
    //float* dst = out_data.data();

    //const size_t channel_size = size_t(pad_h) * pad_w;
    cv::Mat fp32;
    padded_frame.convertTo(fp32, CV_32F, 1.0f / 255.0f);
    return fp32;

    //for (int y = 0; y < pad_h; y++) {
    //    const cv::Vec3f* ptr = fp32.ptr<cv::Vec3f>(y);
    //    for (int x = 0; x < pad_w; x++) {
    //        auto pix = ptr[x];
    //        dst[channel_size * 0 + y * pad_w + x] = pix[0]; // R
    //        dst[channel_size * 1 + y * pad_w + x] = pix[1]; // G
    //        dst[channel_size * 2 + y * pad_w + x] = pix[2]; // B
    //    }
    //}
}

// 传入 BGR Mat
void getBufferByMat(const cv::Mat& bgr, std::vector<float>& out_data) {
    cv::Mat fp32;
    fp32.convertTo(bgr, CV_32FC3, 1.0f / 255.0f);
    cv::cvtColor(fp32, fp32, cv::COLOR_BGR2RGB);
    cv::Mat padded_frame;
    // 需要填充到的宽度. 每次都重新计算？还是外面传入？
    int pad_w = ((fp32.cols + 32 - 1) / 32) * 32;
    int pad_h = ((fp32.rows + 32 - 1) / 32) * 32;
    cv::copyMakeBorder(fp32, padded_frame, 0, pad_h - fp32.rows, 0, pad_w - fp32.cols,
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));


    out_data.resize(size_t(3) * pad_h * pad_w);
    auto dst = out_data.data();
    const size_t channel_size = size_t(pad_h) * pad_w;
    for (int y = 0; y < pad_h; y++) {
       const cv::Vec3f* ptr = fp32.ptr<cv::Vec3f>(y);
       for (int x = 0; x < pad_w; x++) {
           auto pix = ptr[x];
           dst[channel_size * 0 + y * pad_w + x] = pix[0]; // R
           dst[channel_size * 1 + y * pad_w + x] = pix[1]; // G
           dst[channel_size * 2 + y * pad_w + x] = pix[2]; // B
       }
    }
}

cv::Mat TensorToMat(Ort::Value& tensor, int width, int height) {
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    int64_t out_h = shape[2];
    int64_t out_w = shape[3];

    float* data = tensor.GetTensorMutableData<float>();
    int hw = out_h * out_w;

    cv::Mat result(out_h, out_w, CV_8UC3);

    for (int h = 0; h < out_h; ++h) {
        // 取行指针（比 at<> 快 5~10 倍）
        cv::Vec3b* row = result.ptr<cv::Vec3b>(h);

        for (int w = 0; w < out_w; ++w) {
            // 极简索引，无冗余计算（和你逻辑完全一致）
            float r = data[0 * hw + h * out_w + w];
            float g = data[1 * hw + h * out_w + w];
            float b = data[2 * hw + h * out_w + w];

            // 直接写入，无任何多余操作
            row[w][0] = (uchar)(b * 255);
            row[w][1] = (uchar)(g * 255);
            row[w][2] = (uchar)(r * 255);
        }
    }

    if (out_h > height || out_w > width) {
        result = result(cv::Rect(0, 0, width, height)).clone();
    }

    return result;
}

// 将frame 转换为的 cv::Mat 转为 vector<float>
void matToVector(cv::Mat& mat, std::vector<float>& out_data, int pad_w, int pad_h) {
    out_data.resize(size_t(3) * pad_h * pad_w);
    float* dst = out_data.data();

    const size_t channel_size = size_t(pad_h) * pad_w;
    for (int y = 0; y < pad_h; y++) {
        const cv::Vec3f* ptr = mat.ptr<cv::Vec3f>(y);
        for (int x = 0; x < pad_w; x++) {
            auto pix = ptr[x];
            dst[channel_size * 0 + y * pad_w + x] = pix[0]; // R
            dst[channel_size * 1 + y * pad_w + x] = pix[1]; // G
            dst[channel_size * 2 + y * pad_w + x] = pix[2]; // B
        }
    }
}

double compute_ssim_rgb_f32(
    const cv::Mat& img1,
    const cv::Mat& img2,
    int window = 4, // 局部窗口大小, 每次用8x8的块计算一次SSIM 4-8 8-11 16+
    int step = 1    // 步长, 1 2 4 8 step 越小 → 采样越密 → 越精确 → 越慢. 外部图片缩小了再对比的，这里把值设小一点
) {
    CV_Assert(img1.type() == CV_32FC3);
    CV_Assert(img2.type() == CV_32FC3);
    CV_Assert(img1.size() == img2.size());

    const int width  = img1.cols;
    const int height = img1.rows;

    // 对应 [0,1] 范围
    const double C1 = (0.01 * 1.0) * (0.01 * 1.0);
    const double C2 = (0.03 * 1.0) * (0.03 * 1.0);

    double ssim_sum = 0.0;
    int count = 0;

    for (int y = 0; y <= height - window; y += step) {
        for (int x = 0; x <= width - window; x += step) {

            // 每个通道分别统计
            double ssim_c[3] = {0, 0, 0};

            for (int c = 0; c < 3; c++) {
                double sum_x = 0, sum_y = 0;
                double sum_x2 = 0, sum_y2 = 0, sum_xy = 0;

                for (int j = 0; j < window; j++) {
                    const cv::Vec3f* row1 = img1.ptr<cv::Vec3f>(y + j);
                    const cv::Vec3f* row2 = img2.ptr<cv::Vec3f>(y + j);

                    for (int i = 0; i < window; i++) {
                        float a = row1[x + i][c];
                        float b = row2[x + i][c];

                        sum_x  += a;
                        sum_y  += b;
                        sum_x2 += a * a;
                        sum_y2 += b * b;
                        sum_xy += a * b;
                    }
                }

                const int N = window * window;

                double mu_x = sum_x / N;
                double mu_y = sum_y / N;

                double sigma_x2 = sum_x2 / N - mu_x * mu_x;
                double sigma_y2 = sum_y2 / N - mu_y * mu_y;
                double sigma_xy = sum_xy / N - mu_x * mu_y;

                double numerator =
                    (2 * mu_x * mu_y + C1) *
                    (2 * sigma_xy + C2);

                double denominator =
                    (mu_x * mu_x + mu_y * mu_y + C1) *
                    (sigma_x2 + sigma_y2 + C2);

                ssim_c[c] = numerator / denominator;
            }

            // 三通道平均
            double ssim = (ssim_c[0] + ssim_c[1] + ssim_c[2]) / 3.0;

            ssim_sum += ssim;
            count++;
        }
    }

    if (count == 0) return 1.0;
    return ssim_sum / count;
}