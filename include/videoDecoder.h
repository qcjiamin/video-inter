#include <string>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
class VideoDecoder{
public:
    VideoDecoder(const std::string& filename);
    ~VideoDecoder();
    bool GetNextFrame();
    AVFrame* GetFrame() const;
    int GetWidth() const;
    int GetHeight() const;
    double GetFPS() const;
    int64_t GetFrameCount() const;
    unsigned int GetFourCC() const;
    std::string GetFourCCString() const;
    double GetDuration() const;
    // 凡是持有系统资源、指针、句柄的类，必须加上这两行禁止拷贝
    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;
    // 允许移动 std::move()
    VideoDecoder(VideoDecoder&&) = default;
    VideoDecoder& operator=(VideoDecoder&&) = default;
private:
    bool receive_frame_internal();
private:
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVFrame* sw_frame_ = nullptr;   // 总是存放 SW 帧 (用于 HW 下载后存储)
    int video_stream_idx_ = -1;
    AVBufferRef* hw_device_ctx_ = nullptr;
    bool is_hw_accel_ = false;
};