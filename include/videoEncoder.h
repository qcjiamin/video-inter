#pragma once
#pragma once

#include <string>
#include <cstdint>

// FFmpeg headers (必须使用 extern "C")
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavformat/avformat.h>
}

// OpenCV (用于 cv::Mat)
#include <opencv2/opencv.hpp>

/**
 * @brief 基于 FFmpeg 的视频编码器，支持将 RGB 帧编码为 H.264 并写入 MP4 文件
 *
 * 用法：
 *   VideoEncoder encoder("output.mp4", 1920, 1080, 30.0);
 *   cv::Mat rgb_frame; // RGB24 格式
 *   encoder.WriteFrame(rgb_frame);
 *   // 析构时自动写入文件尾并释放资源
 */
class VideoEncoder {
public:
    /**
     * @brief 构造编码器并初始化输出文件
     * @param filename 输出文件名（扩展名决定封装格式，如 .mp4）
     * @param width 视频宽度（像素）
     * @param height 视频高度（像素）
     * @param fps 帧率（帧/秒）
     * @throws std::runtime_error 如果编码器初始化失败（不支持 H.264、无法打开文件等）
     */
    VideoEncoder(const std::string& filename, int width, int height, double fps);

    /**
     * @brief 析构函数，自动刷新编码器缓冲区并写入文件尾
     */
    ~VideoEncoder();

    /**
     * @brief 写入一帧 RGB 图像
     * @param rgb_frame OpenCV Mat 对象，格式必须为 CV_8UC3，颜色空间为 RGB
     * @throws std::runtime_error 如果编码或写入失败
     */
    void WriteFrame(const cv::Mat& rgb_frame);

    // 禁止拷贝和赋值
    VideoEncoder(const VideoEncoder&) = delete;
    VideoEncoder& operator=(const VideoEncoder&) = delete;

private:
    /**
     * @brief 发送 AVFrame 到编码器，并循环接收所有可用 packet 写入文件
     * @param frame 要编码的帧（可以为 nullptr，用于刷新编码器）
     */
    void EncodeFrame(AVFrame* frame);

private:
    int width_;                     // 视频宽度
    int height_;                    // 视频高度
    double fps_;                    // 帧率
    int64_t frame_index_;           // 当前帧索引（用于 PTS）

    AVFormatContext* fmt_ctx_;      // 输出格式上下文
    AVCodecContext* codec_ctx_;     // 编码器上下文
    AVStream* video_stream_;        // 视频流
    SwsContext* sws_ctx_;           // 颜色空间转换上下文（RGB -> YUV420P）
    AVFrame* yuv_frame_;            // 转换后的 YUV 帧（用于编码）

    // 新增：硬件设备上下文
    AVBufferRef* hw_device_ctx_ = nullptr;
    bool is_hw_encoder_ = false;
};