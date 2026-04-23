#include "VideoEncoder.h"
#include <iostream>  // 可选，用于调试输出，但异常处理更常用

VideoEncoder::VideoEncoder(const std::string& filename, int width, int height, double fps)
    : width_(width), height_(height), fps_(fps), frame_index_(0), hw_device_ctx_(nullptr), is_hw_encoder_(false) {
    // 1. 分配输出格式上下文（根据文件扩展名自动选择格式）
    if (avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, filename.c_str()) < 0 ||
        !fmt_ctx_) {
        throw std::runtime_error("Failed to allocate output format context");
    }

    // 2. 尝试查找硬件编码器（优先级：NVENC -> D3D11VA -> QSV -> AMF -> 软件 H.264）
    const AVCodec* codec = nullptr;
    std::string encoder_name;
    AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
    is_hw_encoder_ = true;
    codec = avcodec_find_encoder_by_name("h264_mf");

    // 如果没找到可用的硬件编码器，回退到软件编码器
    if (!codec) {
        is_hw_encoder_ = false;
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        if (!codec) {
            avformat_free_context(fmt_ctx_);
            throw std::runtime_error("H.264 encoder not found (neither hardware nor software)");
        }
        std::cout << "[Encoder] Using software encoder: libx264" << std::endl;
    }


    // // 2. 查找 H.264 编码器（如果不可用，可回退到 MPEG4，但这里直接抛出异常）
    // const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    // if (!codec) {
    //     avformat_free_context(fmt_ctx_);
    //     fmt_ctx_ = nullptr;
    //     throw std::runtime_error("H.264 encoder not found. Please install libx264 or use a different codec.");
    // }

    // 3. 分配编码器上下文
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        throw std::runtime_error("Failed to allocate encoder context");
    }

    // 4. 设置编码参数
    codec_ctx_->width = width;
    codec_ctx_->height = height;
    // time_base: 1/fps（秒），使用有理数表示
    codec_ctx_->time_base = av_d2q(1.0 / fps, 1000000);
    codec_ctx_->framerate = av_d2q(fps, 1000000);
    codec_ctx_->pix_fmt = is_hw_encoder_ ? AV_PIX_FMT_NV12  : AV_PIX_FMT_YUV420P;   // H.264 要求 YUV420P
    codec_ctx_->bit_rate = 4000000;             // 4 Mbps，可根据需要调整
    codec_ctx_->gop_size = 12;                  // 每 12 帧一个关键帧
    codec_ctx_->max_b_frames = 2;
    codec_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;     // 关键：h264_mf 专用配置

    // // 可选：设置编码器预设（仅当编码器为 libx264 时有效）
    // av_opt_set(codec_ctx_->priv_data, "preset", "medium", 0);
    // av_opt_set(codec_ctx_->priv_data, "crf", "23", 0);

     if (is_hw_encoder_) {
        // 为硬件编码器设置通用参数（如 NVENC 的 preset）
        av_opt_set(codec_ctx_->priv_data, "preset", "p4", 0);  // NVENC 预设
        av_opt_set(codec_ctx_->priv_data, "rc", "vbr", 0);
     }
     else {
         av_opt_set(codec_ctx_->priv_data, "preset", "medium", 0);
         av_opt_set(codec_ctx_->priv_data, "crf", "23", 0);
     }

    // 5. 绑定硬件设备上下文（如果是硬件编码器）
    if (is_hw_encoder_ && hw_device_ctx_) {
        codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
        //todo: 需要了解
        // 可选：设置 hw_frames_ctx（如果编码器需要硬件帧输入）
        // 但我们仍会输入软件帧（NV12），由编码器内部处理上传，因此不需要设置 hw_frames_ctx
    }


    // 5. 打开编码器
    int ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret != 0) {
        throw std::runtime_error("Failed to avcodec_open2");
    }

    // 6. 创建视频流
    video_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!video_stream_) {
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        throw std::runtime_error("Failed to create video stream");
    }
    video_stream_->time_base = codec_ctx_->time_base;
    // 将编码器参数复制到流中
    if (avcodec_parameters_from_context(video_stream_->codecpar, codec_ctx_) < 0) {
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        throw std::runtime_error("Failed to copy encoder parameters to stream");
    }

    // 7. 打开输出文件并写入头部
    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx_->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0) {
            avcodec_free_context(&codec_ctx_);
            avformat_free_context(fmt_ctx_);
            fmt_ctx_ = nullptr;
            throw std::runtime_error("Could not open output file: " + filename);
        }
    }
    if (avformat_write_header(fmt_ctx_, nullptr) < 0) {
        avcodec_free_context(&codec_ctx_);
        if (fmt_ctx_ && !(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
            avio_closep(&fmt_ctx_->pb);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        throw std::runtime_error("Failed to write header to output file");
    }

    // 8. 分配颜色转换上下文（RGB24 → YUV420P）
    // sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_RGB24,
    //     width, height, AV_PIX_FMT_YUV420P,
    //     SWS_BILINEAR, nullptr, nullptr, nullptr);

    // 设置目标像素格式（用于颜色转换）
    AVPixelFormat target_pix_fmt = is_hw_encoder_ ? AV_PIX_FMT_NV12 : AV_PIX_FMT_YUV420P;
    sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_RGB24,
                              width, height, target_pix_fmt,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!sws_ctx_) {
        // 清理已分配资源
        av_write_trailer(fmt_ctx_);
        if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
            avio_closep(&fmt_ctx_->pb);
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        throw std::runtime_error("Cannot initialize sws context for color conversion");
    }

    // 9. 分配 YUV 帧（用于存储转换后的数据）
    yuv_frame_ = av_frame_alloc();
    if (!yuv_frame_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
        av_write_trailer(fmt_ctx_);
        if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
            avio_closep(&fmt_ctx_->pb);
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        throw std::runtime_error("Failed to allocate YUV frame");
    }
    //todo why?
    // yuv_frame_->format = codec_ctx_->pix_fmt;
    yuv_frame_->format = target_pix_fmt;
    yuv_frame_->width = width;
    yuv_frame_->height = height;
    if (av_frame_get_buffer(yuv_frame_, 0) < 0) {
        av_frame_free(&yuv_frame_);
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
        av_write_trailer(fmt_ctx_);
        if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
            avio_closep(&fmt_ctx_->pb);
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
        throw std::runtime_error("Failed to allocate buffer for YUV frame");
    }
}

VideoEncoder::~VideoEncoder() {
    // 1. 刷新编码器缓冲区（发送空帧）
    if (codec_ctx_) {
        EncodeFrame(nullptr);
    }
    // 2. 写入文件尾部
    if (fmt_ctx_) {
        av_write_trailer(fmt_ctx_);
    }
    // 3. 释放资源
    if (yuv_frame_) {
        av_frame_free(&yuv_frame_);
    }
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    // 释放硬件设备上下文
    av_buffer_unref(&hw_device_ctx_);
    if (fmt_ctx_) {
        if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE) && fmt_ctx_->pb) {
            avio_closep(&fmt_ctx_->pb);
        }
        avformat_free_context(fmt_ctx_);
    }
}

void VideoEncoder::WriteFrame(const cv::Mat& bgr_frame) {
    if (bgr_frame.empty() || bgr_frame.cols != width_ || bgr_frame.rows != height_) {
        throw std::runtime_error("Invalid frame size or empty frame");
    }
    //std::cout << "write frame" << frame_index_ << std::endl;
    // 1. 创建 RGB AVFrame
    AVFrame* rgb_avframe = av_frame_alloc();
    if (!rgb_avframe) throw std::runtime_error("Failed to allocate RGB AVFrame");

    rgb_avframe->format = AV_PIX_FMT_RGB24;
    rgb_avframe->width = width_;
    rgb_avframe->height = height_;

    // 分配缓冲区，注意：av_frame_get_buffer 会自动处理 linesize 对齐
    if (av_frame_get_buffer(rgb_avframe, 32) < 0) {
        av_frame_free(&rgb_avframe);
        throw std::runtime_error("Failed to allocate buffer for RGB AVFrame");
    }

    // 2. 关键修复：逐行拷贝以处理 linesize 差异，并进行 BGR -> RGB 转换
    const uint8_t* src_data = bgr_frame.data;
    int src_linesize = bgr_frame.step; // OpenCV Mat 的步长

    uint8_t* dst_data = rgb_avframe->data[0];
    int dst_linesize = rgb_avframe->linesize[0];

    for (int y = 0; y < height_; ++y) {
        const uint8_t* src_row = src_data + y * src_linesize;
        uint8_t* dst_row = dst_data + y * dst_linesize;

        for (int x = 0; x < width_; ++x) {
            // OpenCV 是 BGR，FFmpeg RGB24 需要 RGB
            // 源: B, G, R
            // 目: R, G, B
            dst_row[x * 3 + 0] = src_row[x * 3 + 2]; // R
            dst_row[x * 3 + 1] = src_row[x * 3 + 1]; // G
            dst_row[x * 3 + 2] = src_row[x * 3 + 0]; // B
        }
    }

    // 3. 颜色空间转换 RGB -> YUV420P X  将 RGB 转换为 yuv_frame_ 的格式√
    sws_scale(sws_ctx_,
        rgb_avframe->data, rgb_avframe->linesize,
        0, height_,
        yuv_frame_->data, yuv_frame_->linesize);

    av_frame_free(&rgb_avframe);

    // 4. 设置 PTS (注意：这里需要根据实际写入的帧数递增)
    yuv_frame_->pts = frame_index_++;

    // 5. 编码并写入
    EncodeFrame(yuv_frame_);
}

void VideoEncoder::EncodeFrame(AVFrame* frame) {
    int ret = avcodec_send_frame(codec_ctx_, frame);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        throw std::runtime_error("Error sending frame to encoder: " + std::string(errbuf));
    }

    while (ret >= 0) {
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) {
            throw std::runtime_error("Failed to allocate AVPacket");
        }
        ret = avcodec_receive_packet(codec_ctx_, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            break;
        }
        else if (ret < 0) {
            av_packet_free(&pkt);
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, sizeof(errbuf));
            throw std::runtime_error("Error receiving packet from encoder: " + std::string(errbuf));
        }

        // 调整时间基（从编码器上下文的时间基转换为流的时间基）
        av_packet_rescale_ts(pkt, codec_ctx_->time_base, video_stream_->time_base);
        pkt->stream_index = video_stream_->index;

        // 写入文件
        ret = av_interleaved_write_frame(fmt_ctx_, pkt);
        if (ret < 0) {
            av_packet_free(&pkt);
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, sizeof(errbuf));
            throw std::runtime_error("Error writing packet to file: " + std::string(errbuf));
        }
        av_packet_free(&pkt);
    }
}