#include <iostream>
#include <filesystem>
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