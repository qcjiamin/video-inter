#include <iostream>
#include <videoDecoder.h>


VideoDecoder::VideoDecoder(const std::string& filename): hw_device_ctx_(nullptr), is_hw_accel_(false) {
        // 1. 打开文件并查找视频流
        if (avformat_open_input(&fmt_ctx_, filename.c_str(), nullptr, nullptr) < 0) {
            throw std::runtime_error("Could not open input file.");
        }
        if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
            throw std::runtime_error("Could not find stream info.");
        }
        video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_idx_ < 0) {
            throw std::runtime_error("Could not find video stream.");
        }

        // 查找硬件解码器
        AVCodecParameters* codecpar = fmt_ctx_->streams[video_stream_idx_]->codecpar;

        // 先查找普通解码器（如 h264, hevc）
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            throw std::runtime_error("Unsupported codec.");
        }

        //// 尝试创建 D3D11VA 硬件设备上下文
        if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0) == 0) {
            std::cout << "[Decoder] D3D11VA hardware device created successfully." << std::endl;
            is_hw_accel_ = true;
        }
        else {
            std::cout << "[Decoder] Failed to create D3D11VA device, falling back to software." << std::endl;
        }

        // 分配解码器上下文
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            throw std::runtime_error("Could not allocate codec context.");
        }
        //// 如果是硬件解码，需要将硬件设备上下文传递给 codec_ctx
        if (is_hw_accel_ && hw_device_ctx_) {
            //告知解码器使用该硬件设备。
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            // 对于某些硬件解码器，可能需要设置 get_format 回调，但 FFmpeg 新版通常自动处理
        }

        //将流中的编码参数复制到 codec_ctx_（如宽度、高度、extradata 等）
        if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
            throw std::runtime_error("Could not copy codec params.");
        }
        // 打开解码器
        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            throw std::runtime_error("Could not open codec.");
        }


        //const AVCodec* codec = nullptr;
        //std::string hw_decoder_name = "";
        //switch (codecpar->codec_id){
        //    case AV_CODEC_ID_H264:  hw_decoder_name = "d3dva2"; break;
        //    // case AV_CODEC_ID_H264:  hw_decoder_name = "h264_d3d11va"; break;
        //    // case AV_CODEC_ID_H264:  hw_decoder_name = "h264_cuvid"; break;
        //    case AV_CODEC_ID_HEVC:  hw_decoder_name = "hevc_d3d11va"; break;
        //    case AV_CODEC_ID_VP9:   hw_decoder_name = "vp9_d3d11va"; break;
        //    case AV_CODEC_ID_AV1:   hw_decoder_name = "av1_d3d11va"; break;
        //    default: break;
        //}
        //if(!hw_decoder_name.empty()){
        //    codec = avcodec_find_decoder_by_name(hw_decoder_name.c_str());
        //}
        //
        //// 如果找不到硬解码器，则使用软解码器
        //if (!codec) {
        //    std::cout << "[Decoder] Hardware decoder not found or not supported, falling back to software decoding." << std::endl;
        //    codec = avcodec_find_decoder(codecpar->codec_id);
        //}else{
        //    std::cout << "[Decoder] Using hardware decoder: " << hw_decoder_name << std::endl;
        //    is_hw_accel_ = true;
        //    
        //    // 创建硬件设备上下文 (D3D11)
        //    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0) < 0) {
        //        std::cerr << "[Decoder] Failed to create D3D11VA device context. Falling back to software." << std::endl;
        //        is_hw_accel_ = false;
        //        hw_device_ctx_ = nullptr;
        //        codec = avcodec_find_decoder(codecpar->codec_id); // Fallback to SW
        //    }
        //}

        //if(!codec){
        //    throw std::runtime_error("Unsupported codec.");
        //}
        //codec_ctx_ = avcodec_alloc_context3(codec);
        //if (!codec_ctx_) {
        //    throw std::runtime_error("Could not allocate codec context.");
        //}
        //// 如果是硬件解码，需要将硬件设备上下文传递给 codec_ctx
        //if (is_hw_accel_ && hw_device_ctx_) {
        //    //告知解码器使用该硬件设备。
        //    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
        //    // 对于某些硬件解码器，可能需要设置 get_format 回调，但 FFmpeg 新版通常自动处理
        //}
        ////将流中的编码参数复制到 codec_ctx_（如宽度、高度、extradata 等）
        //if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
        //    throw std::runtime_error("Could not copy codec params.");
        //}

        //if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        //    throw std::runtime_error("Could not open codec.");
        //}


        // 2. 初始化解码器
        // const AVCodec* codec = avcodec_find_decoder(fmt_ctx_->streams[video_stream_idx_]->codecpar->codec_id);
        // if (!codec) throw std::runtime_error("Unsupported codec.");
        // codec_ctx_ = avcodec_alloc_context3(codec);
        // if (avcodec_parameters_to_context(codec_ctx_, fmt_ctx_->streams[video_stream_idx_]->codecpar) < 0) {
        //     throw std::runtime_error("Could not copy codec params.");
        // }
        // if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        //     throw std::runtime_error("Could not open codec.");
        // }
        // 3. 为 Packet 和 Frame 分配内存
        pkt_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        // 用于存储从 GPU 下载后的软件帧
        sw_frame_ = av_frame_alloc(); 
        if (!pkt_ || !frame_ || !sw_frame_) {
            throw std::runtime_error("Could not allocate packet or frame.");
        }
    }
VideoDecoder::~VideoDecoder() {
    // 确保所有资源都被释放
    av_frame_free(&sw_frame_);
    av_frame_free(&frame_);
    av_packet_free(&pkt_);
    avcodec_free_context(&codec_ctx_);
    av_buffer_unref(&hw_device_ctx_);
    avformat_close_input(&fmt_ctx_);
}
// 获取下一帧，成功返回 true，读取完毕或出错返回 false
bool VideoDecoder::GetNextFrame() {

        // 基于一个pkt_可能多帧的情况，首先尝试从解码器输出缓冲中直接取帧（处理之前 packet 遗留的输出）
        if (receive_frame_internal()) {
            return true;
        }

        while (true) {
            int ret = av_read_frame(fmt_ctx_, pkt_);
            //当 av_read_frame 返回 <0（文件结束或出错）时，发送 nullptr 包给解码器，通知它输出缓冲中剩余的帧，
            //然后调用 receive_frame_internal() 尝试接收最后一帧。
            if (ret < 0) {
                // EOF or error
                // Flush decoder
                avcodec_send_packet(codec_ctx_, nullptr);
                if (receive_frame_internal()) {
                    return true;
                }
                return false;
            }
            // 只读取视频流的包，其他的立即释放
            if (pkt_->stream_index == video_stream_idx_) {
                ret = avcodec_send_packet(codec_ctx_, pkt_);
                av_packet_unref(pkt_); // Release packet reference immediately

                if (ret < 0) {
                    std::cerr << "Error sending packet to decoder: " << ret << std::endl;
                    continue;
                }
                // 如果能拿到一帧就返回true, 否则继续读下一个包
                if (receive_frame_internal()) {
                    return true;
                }
            } else {
                av_packet_unref(pkt_);
            }
        }
    }
// 获取当前帧 (始终返回 CPU 内存中的 AVFrame，格式为 YUV420P 或 NV12 等软件格式)
AVFrame* VideoDecoder::GetFrame() const{ 
    // 如果使用了硬件加速，sw_frame_ 包含下载后的数据
    // 如果是软件解码，frame_ 本身就是软件数据
    return is_hw_accel_ ? sw_frame_ : frame_; 
}
int VideoDecoder::GetWidth() const {
    return codec_ctx_ ? codec_ctx_->width : 0;
}
int VideoDecoder::GetHeight() const {
    return codec_ctx_ ? codec_ctx_->height : 0;
}
double VideoDecoder::GetFPS() const {
    if (!fmt_ctx_ || video_stream_idx_ < 0) return 0.0;
    AVRational fps = fmt_ctx_->streams[video_stream_idx_]->avg_frame_rate;
    // 如果 avg_frame_rate 为 0/0，尝试 r_frame_rate
    if (fps.num == 0 || fps.den == 0) {
        fps = fmt_ctx_->streams[video_stream_idx_]->r_frame_rate;
    }
    return av_q2d(fps);
}
int64_t VideoDecoder::GetFrameCount() const {
    if (!fmt_ctx_ || video_stream_idx_ < 0) return 0;
    return fmt_ctx_->streams[video_stream_idx_]->nb_frames;
}
// 返回 fourcc 代码（整数），可用于比较或打印
unsigned int VideoDecoder::GetFourCC() const {
    if (!fmt_ctx_ || video_stream_idx_ < 0) return 0;
    AVCodecParameters* codecpar = fmt_ctx_->streams[video_stream_idx_]->codecpar;
    // 对于视频流，codec_tag 通常就是 fourcc
    return codecpar->codec_tag;
}
// 辅助函数：将 fourcc 转为可读字符串（如 "avc1"）
std::string VideoDecoder::GetFourCCString() const {
    unsigned int fourcc = GetFourCC();
    if (fourcc == 0) return "unknown";
    char str[5] = { 0 };
    str[0] = fourcc & 0xFF;
    str[1] = (fourcc >> 8) & 0xFF;
    str[2] = (fourcc >> 16) & 0xFF;
    str[3] = (fourcc >> 24) & 0xFF;
    return std::string(str);
}
double VideoDecoder::GetDuration() const {
    if (!fmt_ctx_) return 0.0;
    // duration 的单位是 AV_TIME_BASE (微秒)
    double duration_sec = fmt_ctx_->duration / (double)AV_TIME_BASE;
    return duration_sec;
}
bool VideoDecoder::receive_frame_internal() {
    // 从上下文中取出一帧
    int ret = avcodec_receive_frame(codec_ctx_, frame_);
    if (ret < 0) {
        // 如果返回 EAGAIN（需要更多数据）或 EOF，返回 false
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return false;
        }
        //todo: 输出错误信息
        std::cerr << "Error receiving frame from decoder: " << ret << std::endl;
        return false;
    }
    // 如果启用了硬件加速，且收到的帧是硬件帧，则需要下载到 CPU
    if (is_hw_accel_ && frame_->format == AV_PIX_FMT_D3D11) {
        std::cout << "HW frame received" << std::endl;
        // 将硬件帧传输到软件帧。将数据从 GPU 拷贝到 CPU 内存，结果存入 sw_frame_
        ret = av_hwframe_transfer_data(sw_frame_, frame_, 0);
        if (ret < 0) {
            std::cerr << "Error transferring the data from the hardware frame to software: " << ret << std::endl;
            av_frame_unref(frame_);
            return false;
        }
        // 复制属性（如 PTS, key_frame 等）
        av_frame_copy_props(sw_frame_, frame_);
    } else if (!is_hw_accel_) {
        // 如果是软件解码，直接使用 frame_，但为了接口统一，也可以拷贝到 sw_frame_ 
        // 或者修改 GetFrame 逻辑。这里为了简单，如果非 HW，GetFrame 返回 frame_
    }
    
    // 注意：调用者使用完 GetFrame() 返回的指针后，不需要手动 unref，
    // 因为下一次 GetNextFrame 会覆盖它。但如果需要长期持有，必须 av_frame_clone。
    
    return true;
}
