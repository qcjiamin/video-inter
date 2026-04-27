// video_pipeline.cpp
// Compile: g++ -std=c++17 -O3 video_pipeline.cpp -o video_pipeline \
//   -lavcodec -lavformat -lavutil -lswscale -lpthread -lstdc++fs
// (若需要 OpenCV 显示信息可链接 opencv4，非必须)

#include <iostream>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

// ========================== 帧数据结构 ==========================
enum class PixelFormat {
    YUV420P,
    RGB24,
};

struct VideoFrame {
    int64_t pts = 0;                // 原始时间戳 (输入流时间基)
    int width = 0;
    int height = 0;
    PixelFormat format = PixelFormat::YUV420P;
    std::shared_ptr<uint8_t> data[4]; // 每个plane的智能指针
    int linesize[4] = {0};

    // 工厂方法：分配新的帧缓冲区
    static VideoFrame* alloc(int w, int h, PixelFormat fmt) {
        auto* frame = new VideoFrame();
        frame->width = w;
        frame->height = h;
        frame->format = fmt;
        int bytes = 0;
        if (fmt == PixelFormat::YUV420P) {
            bytes = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, w, h, 1);
            frame->linesize[0] = w;
            frame->linesize[1] = w/2;
            frame->linesize[2] = w/2;
            frame->linesize[3] = 0;
        } else if (fmt == PixelFormat::RGB24) {
            bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, w, h, 1);
            frame->linesize[0] = w * 3;
            for (int i=1; i<4; ++i) frame->linesize[i]=0;
        } else {
            delete frame;
            return nullptr;
        }
        uint8_t* buf = new uint8_t[bytes];
        frame->data[0] = std::shared_ptr<uint8_t>(buf, std::default_delete<uint8_t[]>());
        if (fmt == PixelFormat::YUV420P) {
            // 对于 planar 格式，需要设置各 plane 指针
            frame->data[1] = std::shared_ptr<uint8_t>(buf + w*h, [buf](uint8_t*){});
            frame->data[2] = std::shared_ptr<uint8_t>(buf + w*h + (w/2)*(h/2), [buf](uint8_t*){});
        }
        return frame;
    }

    ~VideoFrame() = default;
};
using FramePtr = std::shared_ptr<VideoFrame>;

// ========================== 有界阻塞队列 ==========================
template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity) : capacity_(capacity) {}
    
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        // 等待直到队列不满或 shutdown
        // wait 函数会自动执行以下原子操作：
        //    a. 释放 lock (让其他线程有机会 pop 数据)
        //    b. 将当前线程挂起（休眠），直到被 notify 唤醒
        //    c. 唤醒后，重新获取 lock
        //    d. 检查 lambda 表达式: queue_.size() < capacity_ || shutdown_
        //       如果仍为 false (队列还是满的)，继续回到步骤 a 休眠
        not_full_.wait(lock, [this] { return queue_.size() < capacity_ || shutdown_; });
        if (shutdown_) return;
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }
    
    bool pop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_ms < 0) {
            // 等待直到队列非空或 shutdown
            not_empty_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        } else {
            auto timeout = std::chrono::milliseconds(timeout_ms);
            if (!not_empty_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; }))
                return false;
        }
        if (shutdown_ && queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
private:
    mutable std::mutex mutex_;  // 互斥锁，只有一个线程能修改或读取队列
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t capacity_;
    bool shutdown_ = false;
};

// ========================== 处理器基类 ==========================
struct ProcessorConfig {
    std::string model_path;  // 示例配置字段
    int extra_param = 0;
};

class Processor {
public:
    virtual ~Processor() = default;
    virtual bool init(const ProcessorConfig& cfg) = 0;
    // 输入一帧，输出零至多帧（例如插帧可产生多个）
    virtual std::vector<FramePtr> process(const FramePtr& input) = 0;
    // 当输入流结束时，输出剩余帧
    virtual std::vector<FramePtr> flush() = 0;
    void setOutputQueue(BlockingQueue<FramePtr>* q) { out_queue_ = q; }
protected:
    BlockingQueue<FramePtr>* out_queue_ = nullptr;
};

// ========================== FFmpeg 解码器 ==========================
class FfmpegDecoder : public Processor {
public:
    ~FfmpegDecoder() { close(); }
    
    bool init(const ProcessorConfig& cfg) override {
        input_filename_ = cfg.model_path;  // 使用 model_path 作为文件名
        int ret = avformat_open_input(&fmt_ctx_, input_filename_.c_str(), nullptr, nullptr);
        if (ret < 0) return false;
        ret = avformat_find_stream_info(fmt_ctx_, nullptr);
        if (ret < 0) return false;
        video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_idx_ < 0) return false;
        AVCodecParameters* codecpar = fmt_ctx_->streams[video_stream_idx_]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) return false;
        codec_ctx_ = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(codec_ctx_, codecpar);
        ret = avcodec_open2(codec_ctx_, codec, nullptr);
        if (ret < 0) return false;
        time_base_ = fmt_ctx_->streams[video_stream_idx_]->time_base;
        packet_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        sws_ctx_ = sws_getContext(codec_ctx_->width, codec_ctx_->height, codec_ctx_->pix_fmt,
                                  codec_ctx_->width, codec_ctx_->height, AV_PIX_FMT_YUV420P,
                                  SWS_BILINEAR, nullptr, nullptr, nullptr);
        return true;
    }
    
    std::vector<FramePtr> process(const FramePtr& input) override {
        // 解码器没有输入，它从文件中读取。这里忽略传入的 input，实际由调度器特殊处理
        // 但为了统一接口，我们直接在这里读取下一帧
        std::vector<FramePtr> result;
        while (true) {
            int ret = av_read_frame(fmt_ctx_, packet_);
            if (ret < 0) {
                if (ret == AVERROR_EOF) break;
                continue;
            }
            if (packet_->stream_index != video_stream_idx_) {
                av_packet_unref(packet_);
                continue;
            }
            ret = avcodec_send_packet(codec_ctx_, packet_);
            av_packet_unref(packet_);
            if (ret < 0) continue;
            while (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx_, frame_);
                if (ret == AVERROR(EAGAIN)) break;
                if (ret < 0) break;
                // 转换为 YUV420P
                auto out_frame = VideoFrame::alloc(codec_ctx_->width, codec_ctx_->height, PixelFormat::YUV420P);
                uint8_t* dst_data[4] = {out_frame->data[0].get(), out_frame->data[1].get(), out_frame->data[2].get(), nullptr};
                int dst_linesize[4] = {out_frame->linesize[0], out_frame->linesize[1], out_frame->linesize[2], 0};
                sws_scale(sws_ctx_, frame_->data, frame_->linesize, 0, codec_ctx_->height,
                          dst_data, dst_linesize);
                out_frame->pts = av_rescale_q(frame_->pts, time_base_, AVRational{1, 1000000}); // 转为微秒
                result.push_back(std::move(out_frame));
                av_frame_unref(frame_);
            }
            if (!result.empty()) break; // 一次返回一帧，保证流水线节奏
        }
        return result;
    }
    
    std::vector<FramePtr> flush() override { return {}; }
    
private:
    void close() {
        if (sws_ctx_) sws_freeContext(sws_ctx_);
        if (frame_) av_frame_free(&frame_);
        if (packet_) av_packet_free(&packet_);
        if (codec_ctx_) avcodec_free_context(&codec_ctx_);
        if (fmt_ctx_) avformat_close_input(&fmt_ctx_);
    }
    std::string input_filename_;
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    int video_stream_idx_ = -1;
    AVRational time_base_;
    AVPacket* packet_ = nullptr;
    AVFrame* frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
};

// ========================== 插帧器 (简化：帧重复实现目标帧率) ==========================
class DummyInterpolator : public Processor {
public:
    bool init(const ProcessorConfig& cfg) override {
        target_fps_ = cfg.extra_param; // 使用 extra_param 传递目标帧率
        return target_fps_ > 0;
    }
    
    std::vector<FramePtr> process(const FramePtr& input) override {
        std::vector<FramePtr> result;
        // 缓存前一帧，用于计算需要插入的帧数
        if (prev_frame_) {
            // 简单策略：假设输入帧率固定 (通过传入的 pts 计算)
            // 这里示例简化：强制每输入两帧之间插入 (target_fps_/input_fps_ -1) 帧
            // 实际应从 pts 差值计算，为简化我们使用固定倍数: 假设输入 30fps, 目标 60fps -> 插入1帧
            // 更通用的实现应该基于时间戳，此处演示简单重复
            if (target_fps_ > 30) { // 硬编码假设原始帧率30
                int insert_cnt = target_fps_ / 30 - 1;
                for (int i = 0; i < insert_cnt; ++i) {
                    auto dup = copyFrame(prev_frame_);
                    result.push_back(std::move(dup));
                }
            }
        }
        result.push_back(input); // 输出当前帧
        prev_frame_ = input;
        return result;
    }
    
    std::vector<FramePtr> flush() override {
        return {}; // 无剩余帧
    }
    
private:
    FramePtr copyFrame(const FramePtr& src) {
        auto dst = VideoFrame::alloc(src->width, src->height, src->format);
        int bytes = 0;
        if (src->format == PixelFormat::YUV420P) {
            bytes = src->width * src->height + 2 * (src->width/2)*(src->height/2);
        } else if (src->format == PixelFormat::RGB24) {
            bytes = src->width * src->height * 3;
        }
        memcpy(dst->data[0].get(), src->data[0].get(), bytes);
        dst->pts = src->pts; // 简化：保持相同pts；实际应更新
        return dst;
    }
    FramePtr prev_frame_;
    int target_fps_ = 60;
};

// ========================== 超分器 (简化：双线性放大) ==========================
class DummySuperResolution : public Processor {
public:
    bool init(const ProcessorConfig& cfg) override {
        scale_factor_ = cfg.extra_param;
        if (scale_factor_ <= 0) scale_factor_ = 2;
        return true;
    }
    
    std::vector<FramePtr> process(const FramePtr& input) override {
        int new_w = input->width * scale_factor_;
        int new_h = input->height * scale_factor_;
        auto out = VideoFrame::alloc(new_w, new_h, input->format);
        // 简单双线性缩放 (使用 sws)
        SwsContext* sws = sws_getContext(input->width, input->height, 
                                         (input->format==PixelFormat::YUV420P?AV_PIX_FMT_YUV420P:AV_PIX_FMT_RGB24),
                                         new_w, new_h,
                                         (input->format==PixelFormat::YUV420P?AV_PIX_FMT_YUV420P:AV_PIX_FMT_RGB24),
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
        uint8_t* src_data[4] = {input->data[0].get(), input->data[1].get(), input->data[2].get(), nullptr};
        int src_linesize[4] = {input->linesize[0], input->linesize[1], input->linesize[2], 0};
        uint8_t* dst_data[4] = {out->data[0].get(), out->data[1].get(), out->data[2].get(), nullptr};
        int dst_linesize[4] = {out->linesize[0], out->linesize[1], out->linesize[2], 0};
        sws_scale(sws, src_data, src_linesize, 0, input->height, dst_data, dst_linesize);
        sws_freeContext(sws);
        out->pts = input->pts;
        return {std::move(out)};
    }
    
    std::vector<FramePtr> flush() override { return {}; }
private:
    int scale_factor_ = 2;
};

// ========================== FFmpeg 编码器 ==========================
class FfmpegEncoder : public Processor {
public:
    ~FfmpegEncoder() { close(); }
    
    bool init(const ProcessorConfig& cfg) override {
        output_filename_ = cfg.model_path;
        // 编码参数从 config 或硬编码
        width_ = 1920;   // 会在process第一帧时动态设置
        height_ = 1080;
        fps_ = 30;
        // 打开输出上下文
        int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, output_filename_.c_str());
        if (ret < 0) return false;
        const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        if (!codec) return false;
        stream_ = avformat_new_stream(fmt_ctx_, nullptr);
        codec_ctx_ = avcodec_alloc_context3(codec);
        codec_ctx_->width = width_;
        codec_ctx_->height = height_;
        codec_ctx_->time_base = AVRational{1, fps_};
        codec_ctx_->framerate = AVRational{fps_, 1};
        codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
        codec_ctx_->bit_rate = 4000000;
        codec_ctx_->gop_size = 12;
        av_opt_set(codec_ctx_->priv_data, "preset", "medium", 0);
        if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER)
            codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        ret = avcodec_open2(codec_ctx_, codec, nullptr);
        if (ret < 0) return false;
        ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
        if (ret < 0) return false;
        stream_->time_base = codec_ctx_->time_base;
        ret = avio_open(&fmt_ctx_->pb, output_filename_.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) return false;
        ret = avformat_write_header(fmt_ctx_, nullptr);
        if (ret < 0) return false;
        
        sws_ctx_ = sws_getContext(width_, height_, AV_PIX_FMT_YUV420P,
                                  width_, height_, AV_PIX_FMT_YUV420P,
                                  SWS_BILINEAR, nullptr, nullptr, nullptr);
        packet_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        frame_->width = width_;
        frame_->height = height_;
        frame_->format = AV_PIX_FMT_YUV420P;
        av_frame_get_buffer(frame_, 0);
        return true;
    }
    
    std::vector<FramePtr> process(const FramePtr& input) override {
        // 第一次调用时动态设置尺寸和fps
        if (first_frame_) {
            width_ = input->width;
            height_ = input->height;
            // 重新配置编码器 (简单处理：关闭原编码器重新打开)
            // 为简化，这里假设输入分辨率与init时一致，实际应动态重建编码器。
            first_frame_ = false;
        }
        // 拷贝数据到编码帧
        uint8_t* src_data[4] = {input->data[0].get(), input->data[1].get(), input->data[2].get(), nullptr};
        int src_linesize[4] = {input->linesize[0], input->linesize[1], input->linesize[2], 0};
        sws_scale(sws_ctx_, src_data, src_linesize, 0, height_, frame_->data, frame_->linesize);
        frame_->pts = next_pts_++;
        int ret = avcodec_send_frame(codec_ctx_, frame_);
        if (ret < 0) return {};
        std::vector<FramePtr> empty_result;
        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx_, packet_);
            if (ret == AVERROR(EAGAIN)) break;
            if (ret < 0) break;
            av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
            av_write_frame(fmt_ctx_, packet_);
            av_packet_unref(packet_);
        }
        return {};
    }
    
    std::vector<FramePtr> flush() override {
        avcodec_send_frame(codec_ctx_, nullptr);
        int ret = 0;
        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx_, packet_);
            if (ret < 0) break;
            av_write_frame(fmt_ctx_, packet_);
            av_packet_unref(packet_);
        }
        av_write_trailer(fmt_ctx_);
        return {};
    }
    
private:
    void close() {
        if (packet_) av_packet_free(&packet_);
        if (frame_) av_frame_free(&frame_);
        if (sws_ctx_) sws_freeContext(sws_ctx_);
        if (codec_ctx_) avcodec_free_context(&codec_ctx_);
        if (fmt_ctx_) {
            if (fmt_ctx_->pb) avio_close(fmt_ctx_->pb);
            avformat_free_context(fmt_ctx_);
        }
    }
    std::string output_filename_;
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVPacket* packet_ = nullptr;
    AVFrame* frame_ = nullptr;
    int width_ = 0, height_ = 0;
    int fps_ = 30;
    int64_t next_pts_ = 0;
    bool first_frame_ = true;
};

// ========================== 流水线调度器 ==========================
class Pipeline {
public:
    void addProcessor(std::unique_ptr<Processor> proc, size_t queue_capacity = 10) {
        processors_.push_back(std::move(proc));
        queues_.push_back(std::make_unique<BlockingQueue<FramePtr>>(queue_capacity));
    }
    
    // 设置源（解码器）的特殊处理：它主动读取文件，不从队列取数据
    void setSource(std::unique_ptr<Processor> src) {
        source_ = std::move(src);
    }
    
    void setSink(std::unique_ptr<Processor> sink) {
        sink_ = std::move(sink);
    }
    
    void run() {
        // 连接队列：第i个处理器的输出队列 = 第i+1个处理器的输入队列
        // 但需要特殊处理：第一个处理器的输入为空（源），最后一个处理器的输出不连接
        for (size_t i = 0; i < processors_.size(); ++i) {
            if (i > 0) {
                processors_[i]->setOutputQueue(queues_[i-1].get()); // 注意方向: 前一阶段输出 -> 后一阶段输入队列
            }
        }
        // 如果设置了源和汇，将其加入处理链
        // 简化：假设 processors_ 包含了解码器、插帧器、超分器、编码器。并且 source_/sink_未使用
        // 这里直接使用 processors_ 首尾
        
        // 启动线程: 每个处理器一个工作线程
        for (size_t i = 0; i < processors_.size(); ++i) {
            threads_.emplace_back([this, i]() {
                auto* proc = processors_[i].get();
                BlockingQueue<FramePtr>* in_queue = nullptr;
                if (i > 0) {
                    in_queue = queues_[i-1].get(); // 输入队列来自上一个队列
                }
                // 特殊处理: 解码器 (i==0) 没有输入队列，直接读取文件
                if (i == 0) {
                    while (running_) {
                        auto frames = proc->process(nullptr); // 解码器忽略输入
                        if (frames.empty()) {
                            // 解码结束，通知下游
                            if (proc->getOutputQueue()) proc->getOutputQueue()->shutdown();
                            break;
                        }
                        for (auto& f : frames) {
                            proc->getOutputQueue()->push(f);
                        }
                    }
                } else {
                    while (running_) {
                        FramePtr frame;
                        bool ok = in_queue->pop(frame, 100);
                        if (!ok && in_queue->empty() && !running_) break;
                        if (!frame) continue;
                        auto outputs = proc->process(frame);
                        for (auto& out : outputs) {
                            proc->getOutputQueue()->push(out);
                        }
                    }
                    // flush
                    auto remaining = proc->flush();
                    for (auto& f : remaining) {
                        proc->getOutputQueue()->push(f);
                    }
                    if (proc->getOutputQueue()) proc->getOutputQueue()->shutdown();
                }
            });
        }
        
        // 等待所有线程结束
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }
    
    void stop() {
        running_ = false;
        for (auto& q : queues_) q->shutdown();
    }
    
private:
    std::vector<std::unique_ptr<Processor>> processors_;
    std::vector<std::unique_ptr<BlockingQueue<FramePtr>>> queues_;
    std::vector<std::thread> threads_;
    std::unique_ptr<Processor> source_;
    std::unique_ptr<Processor> sink_;
    std::atomic<bool> running_{true};
};

// ========================== 主函数示例 ==========================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video>" << std::endl;
        return 1;
    }
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    // 初始化 FFmpeg 网络/全局（可选）
    avformat_network_init();
    
    Pipeline pipeline;
    
    // 1. 解码器
    auto decoder = std::make_unique<FfmpegDecoder>();
    ProcessorConfig dec_cfg;
    dec_cfg.model_path = input_file;
    if (!decoder->init(dec_cfg)) {
        std::cerr << "Failed to init decoder" << std::endl;
        return 1;
    }
    
    // 2. 插帧器 (目标帧率 60)
    auto interpolator = std::make_unique<DummyInterpolator>();
    ProcessorConfig interp_cfg;
    interp_cfg.extra_param = 60; // 目标 fps
    if (!interpolator->init(interp_cfg)) {
        std::cerr << "Failed to init interpolator" << std::endl;
        return 1;
    }
    
    // 3. 超分器 (2倍)
    auto sr = std::make_unique<DummySuperResolution>();
    ProcessorConfig sr_cfg;
    sr_cfg.extra_param = 2;
    if (!sr->init(sr_cfg)) {
        std::cerr << "Failed to init super-resolution" << std::endl;
        return 1;
    }
    
    // 4. 编码器
    auto encoder = std::make_unique<FfmpegEncoder>();
    ProcessorConfig enc_cfg;
    enc_cfg.model_path = output_file;
    if (!encoder->init(enc_cfg)) {
        std::cerr << "Failed to init encoder" << std::endl;
        return 1;
    }
    
    // 按顺序添加到流水线
    pipeline.addProcessor(std::move(decoder), 8);
    pipeline.addProcessor(std::move(interpolator), 8);
    pipeline.addProcessor(std::move(sr), 8);
    pipeline.addProcessor(std::move(encoder), 8);
    
    pipeline.run();
    std::cout << "Processing finished." << std::endl;
    return 0;
}