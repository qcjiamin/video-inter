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
#include <onnxruntime_cxx_api.h>
#include "utils.h"

class DmlInfer{
public:
    DmlInfer(std::string modelPath, int width, int height) :
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
        dims = { 1, 3, height, width };
        input_tensors.reserve(2); // 预分配空间，提升性能
    }

    // 确保 I0, I1 数据不会在函数内有任何修改
    std::vector<Ort::Value> infer2(std::vector<float>& I0, std::vector<float>& I1, int n) {
        // 第一帧特殊处理，直接返回空，等待下一帧
        if(I1.empty()){
            // 假设输入的数据没有被预处理过

            lastBuffer = std::move(I0); // 这样用的话demo的逻辑就不能使用了，因为数据在这里被转移了
            return std::vector<Ort::Value>{};
        }

        
        // 如果传入的是frame? 每张图片都需要转rgb、chw、归一化、填充
        // 如果外面已经处理好了，传入的是浮点数数组，即buffer
        
        // 传入的图片需要做的预处理
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
    std::vector<float> lastBuffer;
    std::vector<float> lastSmall;
};