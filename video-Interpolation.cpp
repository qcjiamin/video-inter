// video-Interpolation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <filesystem>
#include <dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}
#include <vector>
#include <opencv2/opencv.hpp>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <videoEncoder.h>
#include <videoDecoder.h>
//tensorRT
// #include <NvInfer.h>
// #include <fstream>
// #include <cuda_runtime.h>

// using namespace nvinfer1;

/**
 * 列出当前 FFmpeg 库中所有的解码器
 */
void list_all_decoders() {
    void* opaque = nullptr;
    const AVCodec* codec = nullptr;
    int decoder_count = 0;
    
    std::cout << "========================================" << std::endl;
    std::cout << "   Available FFmpeg Decoders List" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(25) << "Decoder Name" 
              << std::setw(30) << "Long Name" 
              << "Type" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // 遍历所有已注册的编解码器
    while ((codec = av_codec_iterate(&opaque))) {
        // 只关注解码器 (支持解码能力的编解码器)
        //if (av_codec_is_encoder(codec) && (codec->capabilities & AV_CODEC_CAP_HARDWARE)) {
        if (av_codec_is_encoder(codec)) {
            std::string media_type = "Unknown";
            if (codec->type == AVMEDIA_TYPE_VIDEO) {
                media_type = "Video";
            } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
                media_type = "Audio";
            } else if (codec->type == AVMEDIA_TYPE_SUBTITLE) {
                media_type = "Subtitle";
            }
            
            std::cout << std::left << std::setw(25) << codec->name
                      << std::setw(30) << (codec->long_name ? codec->long_name : "N/A")
                      << media_type << std::endl;
            decoder_count++;
        }
    }
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total decoders found: " << decoder_count << std::endl;
}


void printModelInputInfo(Ort::Session& session) {
    size_t numInputNodes = session.GetInputCount();
    std::cout << "Number of inputs: " << numInputNodes << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;

    for (size_t i = 0; i < numInputNodes; i++) {
        // 输入名字
        //char* inputName = session.GetInputName(i, allocator);
        //std::cout << "Input " << i << " name: " << inputName << std::endl;

        // 类型信息
        Ort::TypeInfo typeInfo = session.GetInputTypeInfo(i);

        if (typeInfo.GetONNXType() == ONNX_TYPE_TENSOR) {
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

            // shape
            auto shape = tensorInfo.GetShape();

            std::cout << "Shape: [ ";
            for (auto dim : shape) {
                if (dim == -1)
                    std::cout << "dynamic ";
                else
                    std::cout << dim << " ";
            }
            std::cout << "]" << std::endl;

            // 数据类型
            ONNXTensorElementDataType type = tensorInfo.GetElementType();
            std::cout << "Element type: " << type << std::endl;
        }

        std::cout << "------------------------" << std::endl;

        //allocator.Free(inputName);
    }
}


// 函数: 将 AVFrame (假设为 YUV420P 格式) 转换为模型的输入张量
// 参数: frame - FFmpeg 解码出的视频帧
//       target_width - 模型期望的输入宽度
//       target_height - 模型期望的输入高度
// 返回: Ort::Value 类型的张量，可直接用于推理
//Ort::Value AVFrameToTensor(AVFrame* frame, int target_width, int target_height) {
//
//    // 1. 初始化格式转换上下文 (YUV420P -> RGB24)
//    SwsContext* sws_ctx = sws_getContext(
//        frame->width, frame->height, (AVPixelFormat)frame->format,
//        target_width, target_height, AV_PIX_FMT_RGB24,
//        SWS_BILINEAR, nullptr, nullptr, nullptr
//    );
//
//    if (!sws_ctx) {
//        throw std::runtime_error("Cannot initialize SwsContext");
//    }
//
//    // 2. 分配存储 RGB 数据的缓冲区
//    cv::Mat rgb_frame(target_height, target_width, CV_8UC3);
//
//    // 3. 准备转换所需的指针 (指向 RGB 缓冲区)
//    uint8_t* dest_data[1] = { rgb_frame.data };
//    int dest_linesize[1] = { target_width * 3 }; // RGB24 每行字节数
//
//    // 4. 执行像素格式转换和缩放
//    sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
//        dest_data, dest_linesize);
//    sws_freeContext(sws_ctx);
//
//    // 5. 数据重排 (HWC -> CHW) 与归一化 (uint8 [0,255] -> float [0,1])
//    // 创建存放 float 数据的 vector，大小为 C * H * W
//    std::vector<float> input_tensor_values(3 * target_height * target_width);
//    for (int h = 0; h < target_height; ++h) {
//        for (int w = 0; w < target_width; ++w) {
//            // 在 OpenCV 中，RGB 数据的存储顺序是 B、G、R
//            cv::Vec3b pixel = rgb_frame.at<cv::Vec3b>(h, w);
//            // 根据模型要求，可能需要将通道顺序从 BGR 转为 RGB
//            float r = pixel[2] / 255.0f;
//            float g = pixel[1] / 255.0f;
//            float b = pixel[0] / 255.0f;
//
//            // 计算在 CHW 格式中的索引: index = (c * H + h) * W + w
//            input_tensor_values[(0 * target_height + h) * target_width + w] = r;
//            input_tensor_values[(1 * target_height + h) * target_width + w] = g;
//            input_tensor_values[(2 * target_height + h) * target_width + w] = b;
//        }
//    }
//
//    // 6. 构建 ONNX Runtime 张量
//    std::vector<int64_t> input_node_dims = { 1, 3, target_height, target_width };
//    size_t input_tensor_size = input_tensor_values.size();
//
//    // 创建内存信息
//    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
//        OrtArenaAllocator, OrtMemTypeDefault
//    );
//
//    // 创建 Ort::Value 张量
//    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//        memory_info,
//        input_tensor_values.data(),
//        input_tensor_size,
//        input_node_dims.data(),
//        input_node_dims.size()
//        );
//
//    return input_tensor;
//}

Ort::Value AVFrameToTensor(AVFrame* frame, int /*target_width*/, int /*target_height*/) {
    // 1. 原始尺寸
    int src_w = frame->width;
    int src_h = frame->height;

    // 2. 计算填充后的尺寸（32 的倍数）
    int align = 32;
    int pad_w = ((src_w + align - 1) / align) * align;
    int pad_h = ((src_h + align - 1) / align) * align;

    // 3. 初始化格式转换上下文 (YUV420P -> RGB24)，保持原始尺寸，不缩放
    SwsContext* sws_ctx = sws_getContext(
        src_w, src_h, (AVPixelFormat)frame->format,
        src_w, src_h, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );

    if (!sws_ctx) {
        throw std::runtime_error("Cannot initialize SwsContext");
    }

    // 4. 分配存储 RGB 数据的缓冲区（原始尺寸）
    cv::Mat rgb_frame(src_h, src_w, CV_8UC3);

    // 5. 准备转换所需的指针
    uint8_t* dest_data[1] = { rgb_frame.data };
    int dest_linesize[1] = { src_w * 3 }; // RGB24 每行字节数

    // 6. 执行像素格式转换（YUV -> RGB）
    sws_scale(sws_ctx, frame->data, frame->linesize, 0, src_h,
        dest_data, dest_linesize);
    sws_freeContext(sws_ctx);

    // 7. 创建填充后的图像（pad_h x pad_w，黑色背景）
    cv::Mat padded_frame(pad_h, pad_w, CV_8UC3, cv::Scalar(0, 0, 0));
    // 将 rgb_frame 复制到 padded_frame 的左上角
    rgb_frame.copyTo(padded_frame(cv::Rect(0, 0, src_w, src_h)));

    // 8. 数据重排 (HWC -> CHW) 与归一化 (uint8 [0,255] -> float [0,1])
    std::vector<float> input_tensor_values(3 * pad_h * pad_w);
    for (int h = 0; h < pad_h; ++h) {
        for (int w = 0; w < pad_w; ++w) {
            // 对于填充区域，pixel 为 (0,0,0)，归一化后为 (0,0,0)
            cv::Vec3b pixel = padded_frame.at<cv::Vec3b>(h, w);
            // BGR -> RGB（OpenCV 存储为 BGR，模型通常需要 RGB）
            float r = pixel[2] / 255.0f;
            float g = pixel[1] / 255.0f;
            float b = pixel[0] / 255.0f;

            // CHW 索引计算
            input_tensor_values[(0 * pad_h + h) * pad_w + w] = r;
            input_tensor_values[(1 * pad_h + h) * pad_w + w] = g;
            input_tensor_values[(2 * pad_h + h) * pad_w + w] = b;
        }
    }

    // 9. 构建 ONNX Runtime 张量
    std::vector<int64_t> input_node_dims = { 1, 3, pad_h, pad_w };
    size_t input_tensor_size = input_tensor_values.size();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_size,
        input_node_dims.data(),
        input_node_dims.size()
        );

    return input_tensor;
}


void SaveCHWDataAsImage(const std::vector<float>& data, int pad_h, int pad_w, const std::string& filename) {
    // 1. 创建空的 BGR Mat
    cv::Mat img(pad_h, pad_w, CV_8UC3);

    // 2. 遍历像素，从 CHW float 还原为 HWC uchar
    for (int h = 0; h < pad_h; ++h) {
        for (int w = 0; w < pad_w; ++w) {
            // 从 out_data 中提取 R, G, B
            // 注意：out_data 布局是 [R_plane, G_plane, B_plane]
            float r = data[(0 * pad_h + h) * pad_w + w];
            float g = data[(1 * pad_h + h) * pad_w + w];
            float b = data[(2 * pad_h + h) * pad_w + w];

            // 反归一化并转换为 uchar
            // 确保值在 0-255 之间
            uchar u_r = static_cast<uchar>(std::min(std::max(r * 255.0f, 0.0f), 255.0f));
            uchar u_g = static_cast<uchar>(std::min(std::max(g * 255.0f, 0.0f), 255.0f));
            uchar u_b = static_cast<uchar>(std::min(std::max(b * 255.0f, 0.0f), 255.0f));

            // OpenCV Mat 存储顺序是 BGR
            img.at<cv::Vec3b>(h, w) = cv::Vec3b(u_b, u_g, u_r);
        }
    }

    // 3. 保存图片
    if (!cv::imwrite(filename, img)) {
        std::cerr << "Error: Could not save image to " << filename << std::endl;
    }
    else {
        std::cout << "Saved debug image: " << filename << std::endl;
    }
}

// ==================== 辅助函数：将 AVFrame 转换为填充后的 RGB 数据（存入 vector<float>）====================
// 输入：frame - 解码后的视频帧
// 输出：out_data - 存储 CHW 格式、归一化 [0,1] 的 float 数据，大小 = 3 * pad_h * pad_w
// 返回：填充后的宽度 pad_w 和高度 pad_h
void FrameToPaddedRGBVector(AVFrame* frame, std::vector<float>& out_data, int pad_w, int pad_h) {
    //// 1. 原始尺寸
    int src_w = frame->width;
    int src_h = frame->height;

    //// 2. 计算填充后的尺寸（32 的倍数）
    //int align = 32;
    //pad_w = ((src_w + align - 1) / align) * align;
    //pad_h = ((src_h + align - 1) / align) * align;

    // 3. 初始化格式转换上下文 (YUV420P -> RGB24)，保持原始尺寸，不缩放
    SwsContext* sws_ctx = sws_getContext(
        src_w, src_h, (AVPixelFormat)frame->format,
        src_w, src_h, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
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
    sws_freeContext(sws_ctx);

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
    out_data.resize(size_t(3) * pad_h * pad_w);
    float* dst = out_data.data();

    const size_t channel_size = size_t(pad_h) * pad_w;
    cv::Mat fp32;
    padded_frame.convertTo(fp32, CV_32F, 1.0f / 255.0f);

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

//def make_inference(I0, I1, n) :
//    global model
//    middle = model.inference(I0, I1, args.scale)
//    if n == 1 :
//        return[middle]
//        first_half = make_inference(I0, middle, n = n // 2)
//            second_half = make_inference(middle, I1, n = n // 2)
//                if n % 2:
//return[*first_half, middle, *second_half]
//                else:
//return[*first_half, *second_half]

//Ort::Value make_inferencce(Ort::Session session, Ort::Value I0, Ort::Value I1, int n) {
//    std::vector<Ort::Value> input_tensors;
//    input_tensors.reserve(2); // 预分配空间，提升性能
//    // 清空并重新添加输入张量
//    input_tensors.clear();
//    // 使用 std::move 转移所有权，避免拷贝
//    input_tensors.push_back(std::move(I0));
//    input_tensors.push_back(std::move(I1));
//
//     auto results = session.Run(
//         Ort::RunOptions{ nullptr },
//         input_names,
//         input_tensors_ptr,   // 注意这里传入的是指针数组，每个元素是 Ort::Value*
//         2,
//         output_names,
//         1
//     );
//}

// 对 cv::Mat 进行右下角填充（0填充）
cv::Mat padImage(const cv::Mat& img, int target_w, int target_h) {
    int top = 0, bottom = target_h - img.rows;
    int left = 0, right = target_w - img.cols;
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return padded;
}

// cv::Mat TensorToMat(Ort::Value& tensor, int width, int height) {
//     auto type_info = tensor.GetTypeInfo();
//     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//     auto shape = tensor_info.GetShape();

//     // 假设输出为 NCHW 格式 [1,3,height,width]
//     int64_t channels = shape[1];
//     int64_t out_h = shape[2];
//     int64_t out_w = shape[3];

//     float* data = tensor.GetTensorMutableData<float>();

//     cv::Mat result(out_h, out_w, CV_8UC3);
//     for (int h = 0; h < out_h; ++h) {
//         for (int w = 0; w < out_w; ++w) {
//             float r = data[(0 * channels + 0) * out_h * out_w + h * out_w + w];
//             float g = data[(0 * channels + 1) * out_h * out_w + h * out_w + w];
//             float b = data[(0 * channels + 2) * out_h * out_w + h * out_w + w];
//             // 反归一化 (假设模型输出范围 [0,1])
//             cv::Vec3b pixel(static_cast<uchar>(b * 255),
//                 static_cast<uchar>(g * 255),
//                 static_cast<uchar>(r * 255));
//             result.at<cv::Vec3b>(h, w) = pixel;
//         }
//     }
//     // 裁剪填充区域 (如果之前做了32对齐)
//     if (out_h > height || out_w > width) {
//         result = result(cv::Rect(0, 0, width, height)).clone();
//     }
//     return result;
// }

// 优化版
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

cv::Mat VectorToMat(const std::vector<float>& vec, int out_w, int out_h, int width, int height) {
    int channels = 3; // 输出是3通道的图像
    cv::Mat result(out_h, out_w, CV_8UC3);
    for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
            float r = vec[(0 * channels + 0) * out_h * out_w + h * out_w + w];
            float g = vec[(0 * channels + 1) * out_h * out_w + h * out_w + w];
            float b = vec[(0 * channels + 2) * out_h * out_w + h * out_w + w];
            // 反归一化 (假设模型输出范围 [0,1])
            cv::Vec3b pixel(static_cast<uchar>(b * 255),
                static_cast<uchar>(g * 255),
                static_cast<uchar>(r * 255));
            // float b = data[(0 * channels + 0) * out_h * out_w + h * out_w + w];
            // float g = data[(0 * channels + 1) * out_h * out_w + h * out_w + w];
            // float r = data[(0 * channels + 2) * out_h * out_w + h * out_w + w];
            // // OpenCV Mat 存储 BGR
            // cv::Vec3b pixel(static_cast<uchar>(b * 255),
            //     static_cast<uchar>(g * 255),
            //     static_cast<uchar>(r * 255));
            result.at<cv::Vec3b>(h, w) = pixel;
        }
    }
    // 裁剪填充区域 (如果之前做了32对齐)
    if (out_h > height || out_w > width) {
        result = result(cv::Rect(0, 0, width, height)).clone();
    }
    return result; 
}


// int SaveAVFrameAsImage(AVFrame* frame, const std::string& filename) {
//     if (!frame) return false;

//     // 1. Initialize SwsContext for conversion (YUV420P -> BGR24)
//     // Note: Check frame->format to ensure it matches AV_PIX_FMT_YUV420P or adjust accordingly
//     SwsContext* sws_ctx = sws_getContext(
//         frame->width, frame->height, (AVPixelFormat)frame->format,
//         frame->width, frame->height, AV_PIX_FMT_BGR24,
//         SWS_BILINEAR, nullptr, nullptr, nullptr
//     );

//     if (!sws_ctx) {
//         std::cerr << "Error: Cannot initialize SwsContext." << std::endl;
//         return -1;
//     }

//     // 2. Allocate buffer for BGR data
//     uint8_t* bgr_data[1];
//     int bgr_linesize[1];

//     // Calculate required buffer size
//     int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
//     uint8_t* buffer = (uint8_t*)av_malloc(num_bytes * sizeof(uint8_t));

//     // Fill the pointers and linesizes
//     av_image_fill_arrays(bgr_data, bgr_linesize, buffer, AV_PIX_FMT_BGR24, frame->width, frame->height, 1);

//     // 3. Perform the conversion
//     sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, bgr_data, bgr_linesize);

//     // 4. Create cv::Mat from the BGR data
//     // cv::Mat constructor shares data with bgr_data[0], so we must clone it if we want to free buffer immediately
//     cv::Mat img(frame->height, frame->width, CV_8UC3, bgr_data[0], bgr_linesize[0]);

//     // Clone to ensure data ownership before freeing buffer
//     cv::Mat img_clone = img.clone();

//     // 5. Save using OpenCV
//     bool success = cv::imwrite(filename, img_clone);

//     // 6. Cleanup
//     av_free(buffer);
//     sws_freeContext(sws_ctx);

//     return 0;
// }

void save_frame_as_jpeg(AVFrame* frame, const char* filename) {
    int ret;
    const AVCodec* codec;
    AVCodecContext* codec_ctx = NULL;
    AVPacket* pkt = NULL; // Change to pointer
    AVFrame* rgb_frame = NULL;
    struct SwsContext* sws_ctx = NULL;

    // 1. Initialize encoder
    codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!codec) return;

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) return;

    // Set encoder parameters
    codec_ctx->width = frame->width;
    codec_ctx->height = frame->height;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
    codec_ctx->time_base.num = 1;
    codec_ctx->time_base.den = 25;

    if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
        avcodec_free_context(&codec_ctx);
        return;
    }

    // 2. Prepare frame for encoding (format conversion)
    rgb_frame = av_frame_alloc();
    if (!rgb_frame) {
        avcodec_free_context(&codec_ctx);
        return;
    }

    rgb_frame->format = AV_PIX_FMT_YUVJ420P;
    rgb_frame->width = frame->width;
    rgb_frame->height = frame->height;

    if (av_frame_get_buffer(rgb_frame, 32) < 0) {
        av_frame_free(&rgb_frame);
        avcodec_free_context(&codec_ctx);
        return;
    }

    sws_ctx = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
        rgb_frame->width, rgb_frame->height, (AVPixelFormat)rgb_frame->format,
        SWS_BILINEAR, NULL, NULL, NULL);

    if (!sws_ctx) {
        av_frame_free(&rgb_frame);
        avcodec_free_context(&codec_ctx);
        return;
    }

    sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
        rgb_frame->data, rgb_frame->linesize);

    rgb_frame->pts = 0;

    // 3. Encode and write to file
    pkt = av_packet_alloc(); // Allocate packet
    if (!pkt) {
        sws_freeContext(sws_ctx);
        av_frame_free(&rgb_frame);
        avcodec_free_context(&codec_ctx);
        return;
    }

    ret = avcodec_send_frame(codec_ctx, rgb_frame);
    if (ret < 0) {
        av_packet_free(&pkt);
        sws_freeContext(sws_ctx);
        av_frame_free(&rgb_frame);
        avcodec_free_context(&codec_ctx);
        return;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0)
            break;

        // Write encoded data to file
        FILE* f;
        fopen_s(&f, filename, "wb");
        if (f) {
            fwrite(pkt->data, 1, pkt->size, f);
            fclose(f);
        }
        av_packet_unref(pkt); // Unref instead of free inside loop
    }

    // 4. Cleanup
    av_packet_free(&pkt); // Free packet
    sws_freeContext(sws_ctx);
    av_frame_free(&rgb_frame);
    avcodec_free_context(&codec_ctx);
}

void SaveTensorAsImage(const Ort::Value& tensor, int pad_h, int pad_w, const std::string& filename) {
    // 1. 获取张量信息
    auto type_info = tensor.GetTypeInfo();
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    // 确保形状符合预期: [1, 3, H, W]
    if (shape.size() != 4 || shape[1] != 3) {
        std::cerr << "Error: Tensor shape is not NCHW with 3 channels." << std::endl;
        return;
    }

    int64_t h = shape[2];
    int64_t w = shape[3];

    // 获取数据指针
    const float* data = tensor.GetTensorData<float>();

    //cv::Mat alpha_mat(h, w, CV_32FC1, &data[3 * h * w]); 
    //alpha_mat.convertTo(alpha_mat, CV_8U, 255.0);
    //cv::imwrite("debug_alpha.png", alpha_mat);

    // 2. 创建空的 BGR Mat (OpenCV 默认使用 BGR)
    cv::Mat img(h, w, CV_8UC3);

    // 3. 遍历像素，从 CHW Float 还原为 HWC UChar
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            // 计算 CHW 索引
            // Channel 0: R, Channel 1: G, Channel 2: B
            float r = data[(0 * h + y) * w + x];
            float g = data[(1 * h + y) * w + x];
            float b = data[(2 * h + y) * w + x];

            // 反归一化 [0, 1] -> [0, 255] 并钳位
            uchar u_r = static_cast<uchar>(std::min(std::max(r * 255.0f, 0.0f), 255.0f));
            uchar u_g = static_cast<uchar>(std::min(std::max(g * 255.0f, 0.0f), 255.0f));
            uchar u_b = static_cast<uchar>(std::min(std::max(b * 255.0f, 0.0f), 255.0f));

            // 存入 OpenCV Mat (BGR 顺序)
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(u_b, u_g, u_r);
        }
    }

    // 4. 保存图片
    if (!cv::imwrite(filename, img)) {
        std::cerr << "Error: Could not save tensor image to " << filename << std::endl;
    }
    else {
        std::cout << "Saved tensor debug image: " << filename << std::endl;
    }
}

//实现一个方法，输入视频地址，输出插帧后的视频
int interpolation(std::string path, std::string modelPath) {
    list_all_decoders();

    //判断path是否存在
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
    std::cout << path << " valid" << std::endl;

    std::cout << "video info" << std::endl;
    // 初始化解码器
    VideoDecoder decoder(path);

    // 从解码器获取视频信息
    int width = decoder.GetWidth();
    int height = decoder.GetHeight();
    double fps = decoder.GetFPS();
    int64_t frame_count = decoder.GetFrameCount();
    double duration = decoder.GetDuration();
    unsigned int fourcc = decoder.GetFourCC();
    std::string fourcc_str = decoder.GetFourCCString();

    // 打印获取到的信息
    std::cout << "视频基本信息:" << std::endl;
    std::cout << "帧速率 (FPS): " << fps << std::endl;
    std::cout << "分辨率: " << width << " x " << height << std::endl;
    std::cout << "总帧数: " << frame_count << std::endl;
    std::cout << "时长: " << duration << " s" << std::endl;
    std::cout << "编码格式 (FOURCC): " << fourcc_str << " (0x" << std::hex << fourcc << std::dec << ")" << std::endl;

    std::cout << "decoder inited" << std::endl;



    // 初始化ort
    std::cout << "Initializing ONNX Runtime..." << std::endl;
    const OrtApi& ortApi = Ort::GetApi();
    const void* dmlApi_ptr = nullptr;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VideoInterpolation");
    // 开启详细日志
    // Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "VideoInterpolation");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.DisableMemPattern();
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    OrtStatus* status = ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, &dmlApi_ptr);
    if (status == nullptr && dmlApi_ptr != nullptr) {
        auto dmlApi = reinterpret_cast<const OrtDmlApi*>(dmlApi_ptr);
        auto re = dmlApi->SessionOptionsAppendExecutionProvider_DML(session_options, 0);
        if(re != nullptr) {
            const char* msg = ortApi.GetErrorMessage(re);
            std::cerr << "Error: Failed to set DML execution provider. Reason: " << msg << std::endl;
            ortApi.ReleaseStatus(re);
            return -1;
        }else{
            std::cout << "use dml" << std::endl;
        }
    }
    try {
        // window系统下使用款字符串处理复杂的路径，linux下可以直接使用char*
        std::wstring widestr = std::wstring(modelPath.begin(), modelPath.end());
        Ort::Session session(env, widestr.c_str(), session_options);
        std::cout << "session inited" << std::endl;

        std::cout << "[model info start" << std::endl;

        printModelInputInfo(session);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "[DEBUG] Number of outputs: " << num_output_nodes << std::endl;
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            std::cout << "[DEBUG] Output " << i << " name: " << output_name.get()
                << std::endl;
        }
        std::cout << "model info end]" << std::endl;

        // 构建输出文件
        // 假设你的输出视频参数
        int output_width = width;   // 输出视频宽度
        int output_height = height;  // 输出视频高度
        //double fps = 30.0;        // 输出视频帧率

        // 初始化视频写入器

        VideoEncoder encoder("output_video.mp4", width, height, fps * 2);

        int pad_w = ((width + 32 - 1) / 32) * 32;
        int pad_h = ((height + 32 - 1) / 32) * 32;


        // 循环调用拉取数据和推理
        int frame_count = 0;
        // Ort::Value prev_tensor = nullptr;

        // 输入输出名称（在循环外初始化一次）
        auto input_name0 = session.GetInputNameAllocated(0, allocator);
        auto input_name1 = session.GetInputNameAllocated(1, allocator);
        auto output_name0 = session.GetOutputNameAllocated(0, allocator);
        auto output_name1 = session.GetOutputNameAllocated(1, allocator);
        auto output_name2 = session.GetOutputNameAllocated(2, allocator);
        const char* input_names[] = { input_name0.get(), input_name1.get() };
        const char* output_names[] = { output_name0.get(), output_name1.get(),output_name2.get(), };
        // 内存信息（复用）
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> dims = { 1, 3, pad_h, pad_w };
        // 持久化存储前一帧和当前帧的数据（避免悬空指针）
        // std::vector<float> prev_data, curr_data;

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(2); // 预分配空间，提升性能

        std::vector<float> frame_buffers[2]; 
        int buf_idx = 0;

        ULONGLONG start = GetTickCount64();
        while (decoder.GetNextFrame()) {
            AVFrame* frame = decoder.GetFrame();
            // 解码数据正常
            // save_frame_as_jpeg(frame, ("frame_" + std::to_string(frame_count) + ".jpg").c_str());

            // cv::Mat curr_rgb_frame = AvFrameToCvMat(frame);
            auto& curr_buf = frame_buffers[buf_idx];
            FrameToPaddedRGBVector(frame, curr_buf, pad_w, pad_h);

            // 构建当前帧张量（引用 curr_data 的数据，注意 curr_data 生命周期贯穿整个循环）
            // Ort::Value 在析构时不会释放外部传入的缓冲区，数据的释放完全由 curr_buf 负责
            Ort::Value curr_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                curr_buf.data(),
                curr_buf.size(),
                dims.data(),
                dims.size()
                );


            if (frame_count > 0) {
                auto& prev_buf = frame_buffers[(buf_idx + 1) % 2];
                Ort::Value prev_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    prev_buf.data(),
                    prev_buf.size(),
                    dims.data(),
                    dims.size()
                    );


                cv::Mat last_frame = TensorToMat(prev_tensor, width, height);
                encoder.WriteFrame(last_frame);

                // 清空并重新添加输入张量
                input_tensors.clear();
                // 使用 std::move 转移所有权，避免拷贝
                input_tensors.push_back(std::move(prev_tensor));
                input_tensors.push_back(std::move(curr_tensor));

                // 获取指向 vector 内部数据的指针，类型为 Ort::Value*
                Ort::Value* input_tensors_ptr = input_tensors.data();

                auto results = session.Run(
                    Ort::RunOptions{ nullptr },
                    input_names,
                    input_tensors_ptr,   // 注意这里传入的是指针数组，每个元素是 Ort::Value*
                    2,
                    output_names,
                    3
                );
                // 流式保存视频
                // 1. 获取输出张量信息
                Ort::Value& output_tensor = results[2];

                //std::string debug_filename2 = "output_tensor_" + std::to_string(frame_count) + ".png";
                //SaveTensorAsImage(output_tensor, pad_h, pad_w, debug_filename2);


                cv::Mat rgb_frame = TensorToMat(output_tensor, width, height);

                //std::string imageName = "frame_" + std::to_string(frame_count) + ".png";
                //if (!cv::imwrite(imageName, rgb_frame)) {
                //    std::cerr << "Warning: Could not save image " << imageName << std::endl;
                //}
                // 6. 写入视频帧
                //videoWriter.write(output_uint8);
                encoder.WriteFrame(rgb_frame);



            }else{
                std::cout << "prev_ is null" << std::endl;
            }

            // 切换缓冲区索引，复用两块内存。 当前的 curr_buf 变成上一帧的数据
            buf_idx = (buf_idx + 1) % 2;
            frame_count++;
        }
        // 写入最后一帧
        if(frame_count > 0){
            // 最后一帧结束时，下标被切换到了下一块缓冲区， 最后一帧的缓冲区在相对位置
            int last_buf_idx = (buf_idx + 1) % 2;
            auto& last_buf = frame_buffers[last_buf_idx];
            
            Ort::Value last_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                last_buf.data(),
                last_buf.size(),
                dims.data(),
                dims.size()
            );
            cv::Mat last_frame = TensorToMat(last_tensor, width, height);
            encoder.WriteFrame(last_frame);
        }

        // if (prev_tensor) {
        //     cv::Mat last_frame = TensorToMat(prev_tensor, width, height);
        //     encoder.WriteFrame(last_frame);
        // }
        std::cout << "Total frames processed: " << frame_count << std::endl;
        ULONGLONG end = GetTickCount64();
        std::cout << "耗时: " << (end - start) << " 毫秒" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing ONNX Runtime: " << e.what() << std::endl;
        return -1;
    }

}


// class Logger : public ILogger
// {
//     void log(Severity severity, const char* msg) noexcept override
//     {
//         // suppress info-level messages
//         if (severity <= Severity::kWARNING)
//             std::cout << msg << std::endl;
//     }
// } gLogger;

// std::vector<char> loadFile(const std::string& path) {
//     std::ifstream file(path, std::ios::binary);
//     assert(file.good());
//     return std::vector<char>((std::istreambuf_iterator<char>(file)),
//         std::istreambuf_iterator<char>());
// }


// class TRTInfer {
// public:
//     TRTInfer(const std::string& enginePath) {
//         auto engineData = loadFile(enginePath);
//         runtime = createInferRuntime(gLogger);
//         engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
//         context = engine->createExecutionContext();

//         cudaStreamCreate(&stream);
//     }
//     ~TRTInfer() {
//         for (auto& kv : deviceBuffers) {
//             cudaFree(kv.second);
//         }
//         cudaStreamDestroy(stream);
//         context->destroy();
//         engine->destroy();
//         runtime->destroy();
//     }

//     void allocate(int batch, int h, int w) {
//         inputShapes["img0"] = Dims4(batch, 3, h, w);
//         inputShapes["img1"] = Dims4(batch, 3, h, w);

//         int nb = engine->getNbIOTensors();

//         for (int i = 0; i < nb; i++) {
//             const char* name = engine->getIOTensorName(i);
//             auto mode = engine->getTensorIOMode(name);

//             if (mode == TensorIOMode::kINPUT) {
//                 context->setInputShape(name, inputShapes[name]);
//             }
//         }

//         for (int i = 0; i < nb; i++) {
//             const char* name = engine->getIOTensorName(i);
//             auto shape = context->getTensorShape(name);

//             int64_t size = 1;
//             for (int j = 0; j < shape.nbDims; j++) {
//                 size *= shape.d[j];
//             }

//             size_t bytes = size * sizeof(float);

//             void* devicePtr;
//             cudaMalloc(&devicePtr, bytes);
//             deviceBuffers[name] = devicePtr;

//             // 🔥 关键：绑定 tensor → GPU 地址
//             context->setTensorAddress(name, devicePtr);

//             //bindings.push_back(devicePtr);

//             shapes[name] = shape;
//         }
//     }

//     std::map<std::string, std::vector<float>> infer(std::map<std::string, std::vector<float>>& inputs) {
//         //H2D
//         for (auto& kv : inputs) {
//             cudaMemcpyAsync(deviceBuffers[kv.first],
//                 kv.second.data(),
//                 kv.second.size() * sizeof(float),
//                 cudaMemcpyHostToDevice,
//                 stream);
//         }

//         //执行
//         context->enqueueV3(stream);

//         //D2H
//         std::map<std::string, std::vector<float>> outputs;
//         for (auto& kv : deviceBuffers) {
//             const std::string& name = kv.first;
//             if (engine->getTensorIOMode(name.c_str()) == TensorIOMode::kOUTPUT) {
//                 auto shape = shapes[name];
//                 int size = 1;
//                 for (int i = 0; i < shape.nbDims; i++) {
//                     size *= shape.d[i];
//                 }
//                 std::vector<float> out(size);
//                 cudaMemcpyAsync(out.data(),
//                     kv.second,
//                     size * sizeof(float),
//                     cudaMemcpyDeviceToHost,
//                     stream);
//                 outputs[name] = std::move(out);
//             };
//         }
//         cudaStreamSynchronize(stream);
//         return outputs;
//     }
// private:
//     IRuntime* runtime{ nullptr };
//     ICudaEngine* engine{ nullptr };
//     IExecutionContext* context{ nullptr };
//     cudaStream_t stream;

//     std::map<std::string, void*> deviceBuffers;
//     std::vector<void*> bindings;
//     std::map<std::string, Dims> shapes;
//     std::map<std::string, Dims> inputShapes;
// };


// int interpolationTensorrt(std::string path, std::string modelPath) {
//     //判断path是否存在
//     if (path.empty()) {
//         std::cerr << "Error: Path is null." << std::endl;
//         return -1;
//     }

//     std::filesystem::path filePath(path);

//     if (!std::filesystem::exists(filePath)) {
//         std::cerr << "Error: File does not exist at path: " << path << std::endl;
//         return -1;
//     }

//     if (!std::filesystem::is_regular_file(filePath)) {
//         std::cerr << "Error: Path is not a regular file: " << path << std::endl;
//         return -1;
//     }
//     std::cout << path << " valid" << std::endl;

//     std::cout << "video info" << std::endl;
//     // 初始化解码器
//     VideoDecoder decoder(path);

//     // 从解码器获取视频信息
//     int width = decoder.GetWidth();
//     int height = decoder.GetHeight();
//     double fps = decoder.GetFPS();
//     int64_t frameCount = decoder.GetFrameCount();
//     double duration = decoder.GetDuration();
//     unsigned int fourcc = decoder.GetFourCC();
//     std::string fourcc_str = decoder.GetFourCCString();
//     // 打印获取到的信息
//     std::cout << "视频基本信息:" << std::endl;
//     std::cout << "帧速率 (FPS): " << fps << std::endl;
//     std::cout << "分辨率: " << width << " x " << height << std::endl;
//     std::cout << "总帧数: " << frameCount << std::endl;
//     std::cout << "时长: " << duration << " s" << std::endl;
//     std::cout << "编码格式 (FOURCC): " << fourcc_str << " (0x" << std::hex << fourcc << std::dec << ")" << std::endl;
//     std::cout << "decoder inited" << std::endl;

//     TRTInfer trt(modelPath);

//     int output_width = width;   // 输出视频宽度
//     int output_height = height;  // 输出视频高度
//     VideoEncoder encoder("output_video_tensor.mp4", width, height, fps * 2);
//     int pad_w = ((width + 32 - 1) / 32) * 32;
//     int pad_h = ((height + 32 - 1) / 32) * 32;

//     std::vector<float> frame_buffers[2];
//     int buf_idx = 0;
//     int frame_count = 0;
//     ULONGLONG start = GetTickCount64();
//     while (decoder.GetNextFrame()) {
//         AVFrame* frame = decoder.GetFrame();
//         auto& curr_buf = frame_buffers[buf_idx];
//         FrameToPaddedRGBVector(frame, curr_buf, pad_w, pad_h);
//         if (frame_count > 0) {
//             // 分配显存空间
//             trt.allocate(1, pad_h, pad_w);
//             auto& prev_buf = frame_buffers[(buf_idx + 1) % 2];

//             std::map<std::string, std::vector<float>> inputs = {
//                 {"img0", prev_buf},
//                 {"img1", curr_buf}
//             };

//             auto outputs = trt.infer(inputs);
//             auto rgb_frame = VectorToMat(outputs["merged"], pad_w, pad_h, width, height);

//             encoder.WriteFrame(rgb_frame);
//         }
//         // 切换缓冲区索引，复用两块内存。 当前的 curr_buf 变成上一帧的数据
//         buf_idx = (buf_idx + 1) % 2;
//         frame_count++;
//     }
//     // 写入最后一帧
//     if (frame_count > 0) {
//         // 最后一帧结束时，下标被切换到了下一块缓冲区， 最后一帧的缓冲区在相对位置
//         int last_buf_idx = (buf_idx + 1) % 2;
//         auto& last_buf = frame_buffers[last_buf_idx];

//         cv::Mat last_frame = VectorToMat(last_buf, pad_w, pad_h, width, height);
//         encoder.WriteFrame(last_frame);
//     }
//     std::cout << "Total frames processed: " << frame_count << std::endl;
//     ULONGLONG end = GetTickCount64();
//     std::cout << "耗时: " << (end - start) << " 毫秒" << std::endl;
//     return 0;
// }

int main()
{
    //onnx dml
    int res = interpolation("./demo5s.mp4", "./flownet.onnx");

    //tensor
    //int res = interpolationTensorrt("./demo5s.mp4", "./flownet.engine");
    // 
    //TRTInfer trt("flownet.engine");

    //int H = 512;
    //int W = 512;

    //trt.allocate(1, H, W);

    //std::vector<float> img0(1 * 3 * H * W);
    //std::vector<float> img1(1 * 3 * H * W);

    //// 填充数据
    //for (auto& v : img0) v = rand() / float(RAND_MAX);
    //for (auto& v : img1) v = rand() / float(RAND_MAX);

    //std::map<std::string, std::vector<float>> inputs = {
    //    {"img0", img0},
    //    {"img1", img1}
    //};

    //auto outputs = trt.infer(inputs);

    //for (auto& kv : outputs) {
    //    std::cout << kv.first << " size = " << kv.second.size() << std::endl;
    //}
}
