// dml相关的逻辑封装
#pragma once
#include <iostream>
#include <videoEncoder.h>
#include <videoDecoder.h>
#include <opencv2/opencv.hpp>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#include <dml_provider_factory.h>
//#include <onnxruntime_cxx_api.h> util中已经引入了？
#include "utils.h"

class DmlInfer1{
public:
    DmlInfer1(std::string modelPath) :
        session{nullptr},
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        env(ORT_LOGGING_LEVEL_WARNING, "VideoInterpolation"){
        // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VideoInterpolation");
        int re = fileExists(modelPath);
        if(re != 0){
            throw std::runtime_error("Model file error: " + modelPath);
        }
        
        // 创建会话
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(0);
        session_options.SetInterOpNumThreads(0);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.DisableMemPattern();
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        // 启用 DML
        const OrtApi& ortApi = Ort::GetApi();
        const void* dmlApi_ptr = nullptr;
        OrtStatus* status = ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, &dmlApi_ptr);
        if (status == nullptr && dmlApi_ptr != nullptr) {
            auto dmlApi = reinterpret_cast<const OrtDmlApi*>(dmlApi_ptr);
            auto re = dmlApi->SessionOptionsAppendExecutionProvider_DML(session_options, 0);
            if (re != nullptr) {
                const char* msg = ortApi.GetErrorMessage(re);
                // 此处可以打印日志，说明为什么 DML 失败了
                // std::cerr << "Error: Failed to set DML execution provider. Reason: " << msg << std::endl;
                ortApi.ReleaseStatus(re);
                // 回退回cpu执行，继续创建session
                // return -1;
                // session_options.AppendExecutionProvider_CPU(1);
                //todo 开启cpu算子内的多线程
                session_options.SetIntraOpNumThreads(0);
                session_options.SetInterOpNumThreads(0);
            }
            else {
                std::cout << "use dml" << std::endl;
            }
        }

        std::wstring widestr = std::wstring(modelPath.begin(), modelPath.end());
        session = Ort::Session(env, widestr.c_str(), session_options);

        // 输入输出字段名获取
        Ort::AllocatorWithDefaultOptions allocator;
        // input
        size_t input_count = session.GetInputCount();
        for (size_t i = 0; i < input_count; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_name_strs.push_back(name.get());
        }

        for (auto& s : input_name_strs) {
            input_names.push_back(s.c_str());
        }

        // output
        size_t output_count = session.GetOutputCount();
        for (size_t i = 0; i < output_count; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_name_strs.push_back(name.get());
        }

        for (auto& s : output_name_strs) {
            output_names.push_back(s.c_str());
        }

        // 外部传入
        /*dims = { 1, 3, height, width };*/
        input_tensors.reserve(2); // 预分配空间，提升性能
    }
    std::vector<cv::Mat> infer_(cv::Mat& frame, int n){
        int width = frame.cols;
        int height = frame.rows;
        // 需要填充到的宽度
        int pad_w = ((width + 32 - 1) / 32) * 32;
        int pad_h = ((height + 32 - 1) / 32) * 32;
        if (dims.empty()) {
            dims = { 1, 3, pad_h, pad_w };
        }

        double small_w = 32.0;
        double small_h = pad_h * (small_w / pad_w);
        //cv::Mat curr_small;
        // cv::Mat curr_small(small_h, small_w, frameMat.type()); // 分配大小，稍微快一点
        // auto& curr_small = small_buffers[buf_idx]; // resize自动创建内存 / 重新分配大小, 不需要初始化
        cv::Mat curr_small;
        cv::resize(frame, curr_small, cv::Size(small_w, small_h), 0, 0, cv::INTER_LINEAR);
        curr_small.convertTo(curr_small, CV_32FC3, 1.0f / 255.0f);
        // 第一帧特殊处理，直接返回空，等待下一帧
        if(last_buffer.empty()){
            // 预处理数据, 写入lastBuffer
            getBufferByMat(frame, last_buffer);
            last_small = std::move(curr_small);
            return std::vector<cv::Mat>{};
        }

        // 有lastBuffer, 取currentBuffer
        std::vector<float> curr_buffer;
        getBufferByMat(frame, curr_buffer);

        double ssim =compute_ssim_rgb_f32(last_small, curr_small);
        if(ssim > 0.996){
            std::cout << "same copy, return " << n << " frames" << std::endl;
            std::vector<cv::Mat> out;

            // cv::Mat fp32;
            // frame.convertTo(fp32, CV_32FC3, 1.0f / 255.0f);
            // cv::cvtColor(frame, fp32, cv::COLOR_BGR2RGB);
            out.reserve(n);  // 小优化

            // 相似度高，直接复制上一帧，减少推理次数
            for(int i = 0; i < n; i++){
                // cv::Mat fp32;
                // frame.convertTo(fp32, CV_32FC3, 1.0f / 255.0f);
                // cv::cvtColor(fp32, fp32, cv::COLOR_BGR2RGB);
                // out.push_back(std::move(fp32));
                out.push_back(frame.clone()); // 外面不会修改，只会写入，这里选择不克隆
            }
            last_buffer = std::move(curr_buffer);
            last_small = std::move(curr_small);
            return out;
        }else if(ssim < 0.2){
            std::cout << "diff copy, return " << n << " frames" << std::endl;
            std::vector<cv::Mat> out;
            // 相似度极低，直接写入当前帧，减少推理次数
            // cv::Mat fp32;
            // frame.convertTo(fp32, CV_32FC3, 1.0f / 255.0f);
            // cv::cvtColor(frame, fp32, cv::COLOR_BGR2RGB);
            out.reserve(n);  // 小优化
            for(int i = 0; i < n; i++){
                // cv::Mat fp32;
                // frame.convertTo(fp32, CV_32FC3, 1.0f / 255.0f);
                // cv::cvtColor(fp32, fp32, cv::COLOR_BGR2RGB);
                // out.push_back(std::move(fp32));
                out.push_back(frame.clone());
            }

            last_buffer = std::move(curr_buffer);
            last_small = std::move(curr_small);
            return out;
        }
        
        // 传入的图片需要做的预处理
        input_tensors.clear();
        auto tenser0 = CreateTensor(last_buffer.data(), last_buffer.size());
        auto tenser1 = CreateTensor(curr_buffer.data(), curr_buffer.size());
        // std::vector<Ort::Value> input_tensors;
        // Ort::Value 禁止拷贝, 其内部移除了 拷贝构造函数
        // std::vector::push_back会调用 拷贝构造函数 ，因此需要使用 std::move 来转移所有权，避免编译错误
        // 使用std::move, vector 会调用 移动构造函数
        input_tensors.push_back(std::move(tenser0));
        input_tensors.push_back(std::move(tenser1));
        Ort::Value* input_tensors_ptr = input_tensors.data();

        auto results = session.Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensors_ptr,   // 注意这里传入的是指针数组，每个元素是 Ort::Value*
            input_names.size(),
            output_names.data(),
            output_names.size()
        );
        auto middle = std::move(results[2]);

        if(n == 1){
            std::vector<cv::Mat> out;
            cv::Mat mid_frame = TensorToMat(middle, width, height);
            out.push_back(std::move(mid_frame));
            last_buffer = std::move(curr_buffer);
            last_small = std::move(curr_small);
            return out;
        }
        //todo 这里会发生一次拷贝, 目前没有找到更优解
        const float* middle_data = middle.GetTensorData<float>();
        size_t middle_size = middle.GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> middle_vec(middle_data, middle_data + middle_size);
        
        auto left = infer2(last_buffer, middle_vec, n / 2);
        auto right = infer2(middle_vec, curr_buffer, n / 2);
        
        std::vector<cv::Mat> out;
        out.reserve(left.size() + right.size() + (n % 2 ? 1 : 0));

        for(auto& v : left){
            // 递归里的结果是 Ort::Value, 需要转换成 cv::Mat
            out.push_back(std::move(TensorToMat(v, width, height)));
        }
        if(n % 2){
            cv::Mat mid_frame = TensorToMat(middle, width, height);
            out.push_back(std::move(mid_frame));
        }
        for(auto& v : right){
            out.push_back(std::move(TensorToMat(v, width, height)));
        }

        last_buffer = std::move(curr_buffer);
        last_small = std::move(curr_small);

        return out;
    }
    Ort::Value infer(Ort::Value& I0, Ort::Value& I1, int n) {
        // 清空并重新添加输入张量
        input_tensors.clear();
        // 使用 std::move 转移所有权，避免拷贝。 外部定义局部变量，这里可以直接转移
        // todo 工程化里外部是否定义为可以转移的局部变量是未知的，需要修改
        // Ort::Value 禁止拷贝, 其内部移除了 拷贝构造函数
        // std::vector::push_back会调用 拷贝构造函数 ，因此需要使用 std::move 来转移所有权，避免编译错误
        // 使用std::move, vector 会调用 移动构造函数
        input_tensors.push_back(std::move(I0));
        input_tensors.push_back(std::move(I1));
        // 获取指向 vector 内部数据的指针，类型为 Ort::Value*
        Ort::Value* input_tensors_ptr = input_tensors.data();

        auto results = session.Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensors_ptr,   // 注意这里传入的是指针数组，每个元素是 Ort::Value*
            input_names.size(),
            // 2,
            output_names.data(),
            // 3
            output_names.size()
        );
        return std::move(results[2]); // 编译器无法对容器内的特定元素应用返回值优化（RVO/NRVO）
    }
    Ort::Value CreateTensor(float* data, size_t size) {
        // 编译器会自动优化，直接在目标位置构造对象
        return Ort::Value::CreateTensor<float>(
            memory_info,
            data,
            size,
            dims.data(),
            dims.size()
        );
    }

    cv::Mat getLastBuffer(int width, int height){
        if(last_buffer.empty()){
            return cv::Mat();
        }

        int64_t out_h = dims[2];
        int64_t out_w = dims[3];

        int hw = out_h * out_w;

        cv::Mat result(out_h, out_w, CV_8UC3);

        for (int h = 0; h < out_h; ++h) {
            // 取行指针（比 at<> 快 5~10 倍）
            cv::Vec3b* row = result.ptr<cv::Vec3b>(h);

            for (int w = 0; w < out_w; ++w) {
                // 极简索引，无冗余计算（和你逻辑完全一致）
                float r = last_buffer[0 * hw + h * out_w + w];
                float g = last_buffer[1 * hw + h * out_w + w];
                float b = last_buffer[2 * hw + h * out_w + w];

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
private:
    Ort::Session session;
    Ort::Env env; 

    std::vector<std::string> input_name_strs;
    std::vector<const char*> input_names;

    std::vector<std::string> output_name_strs;
    std::vector<const char*> output_names;
    Ort::MemoryInfo memory_info;
    std::vector<int64_t> dims;
    std::vector<Ort::Value> input_tensors;
    std::vector<float> last_buffer;
    cv::Mat last_small;
    // 私有递归函数
    // 确保 I0, I1 数据不会在函数内有任何修改
    std::vector<Ort::Value> infer2(std::vector<float>& I0, std::vector<float>& I1, int n) {
        input_tensors.clear();
        auto tenser0 = CreateTensor(I0.data(), I0.size());
        auto tenser1 = CreateTensor(I1.data(), I1.size());
        // std::vector<Ort::Value> input_tensors;
        // Ort::Value 禁止拷贝, 其内部移除了 拷贝构造函数
        // std::vector::push_back会调用 拷贝构造函数 ，因此需要使用 std::move 来转移所有权，避免编译错误
        // 使用std::move, vector 会调用 移动构造函数
        input_tensors.push_back(std::move(tenser0));
        input_tensors.push_back(std::move(tenser1));
        Ort::Value* input_tensors_ptr = input_tensors.data();

        auto results = session.Run(
            Ort::RunOptions{ nullptr },
            input_names.data(),
            input_tensors_ptr,   // 注意这里传入的是指针数组，每个元素是 Ort::Value*
            input_names.size(),
            output_names.data(),
            output_names.size()
        );
        auto middle = std::move(results[2]);

        if(n == 1){
            std::vector<Ort::Value> out;
            // out.push_back(std::move(results[2])); // 编译器无法对容器内的特定元素应用返回值优化（RVO/NRVO）
            out.push_back(std::move(middle));
            return out;
        }
        //todo 这里会发生一次拷贝, 目前没有找到更优解
        const float* middle_data = middle.GetTensorData<float>();
        size_t middle_size = middle.GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> middle_vec(middle_data, middle_data + middle_size);
        
        auto left = infer2(I0, middle_vec, n / 2);
        auto right = infer2(middle_vec, I1, n / 2);
        
        std::vector<Ort::Value> out;
        out.reserve(left.size() + right.size() + (n % 2 ? 1 : 0));

        for(auto& v : left){
            out.push_back(std::move(v));
        }
        if(n % 2){
            out.push_back(std::move(middle));
        }
        for(auto& v : right){
            out.push_back(std::move(v));
        }

        return out;
    }
};