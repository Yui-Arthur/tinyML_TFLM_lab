/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "model_data.h"
#include "model_data_quant.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mbed.h"


struct result{
    uint32_t infer_time;
    float input;
    float pred;
    float ans;
    float diff;
    int quant;
    result(): infer_time(0) , input(0) , pred(0) , ans(0) , diff(0) , quant(0) {;};
};

Queue<bool, 4> in_queue;
Queue<struct result, 20> out_queue;

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  
  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 5000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  bool *quant_ptr;
  in_queue.try_get(&quant_ptr);
  bool quant = *quant_ptr;
//   MicroPrintf("Quant Flag %d" , quant);

  const unsigned char *model_data_ptr;
  if(quant)
    model_data_ptr = model_data_quant;
  else
    model_data_ptr = model_data;
  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(model_data_ptr), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);

  if(quant)
    interpreter.input(0)->data.int8[0] = 1;
  else
    interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());


  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  MicroPrintf("ProfileMemoryAndLatency() OK");
  return kTfLiteOk;
}

TfLiteStatus LoadModelAndPerformInference() {
//   MicroPrintf("LoadModelAndPerformInference() Start");
  const unsigned char *model_data_ptr;
  bool *quant_ptr;
  in_queue.try_get(&quant_ptr);
  bool quant = *quant_ptr;
//   MicroPrintf("Quant Flag %d" , quant);
  if(quant)
    model_data_ptr = model_data_quant;
  else
    model_data_ptr = model_data;
  const tflite::Model* model = tflite::GetModel(model_data_ptr);

  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 5000;
  uint8_t tensor_arena[kTensorArenaSize]; 


  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

//   MicroPrintf("Test");

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  float output_scale = 0, input_scale = 0;
  int output_zero_point = 0 , input_zero_point = 0;
  
  if(quant){
    output_scale = output->params.scale , output_zero_point = output->params.zero_point;
    input_scale = input->params.scale , input_zero_point = input->params.zero_point;
    // MicroPrintf("Input quant info : %f , %d",input->params.scale , input->params.zero_point);
    // MicroPrintf("Output quant info : %f , %d",output->params.scale , output->params.zero_point);
  }

  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};
  
  

  for (int i = 0; i < kNumTestValues; ++i) {
    if(quant)
        interpreter.input(0)->data.int8[0] = golden_inputs[i] / input_scale + input_zero_point;
    else
        interpreter.input(0)->data.f[0] = golden_inputs[i];
    mbed::Timer t;
    t.start();
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    t.stop();

    float y_pred = 0.f;

    if(quant)
        y_pred = (interpreter.output(0)->data.int8[0] - output_zero_point) * output_scale;
    else
        y_pred = interpreter.output(0)->data.f[0];
    
    struct result *infer_result = new struct result;
    uint32_t infer_time = std::chrono::duration_cast<std::chrono::microseconds>(t.elapsed_time()).count();
    // auto infer_time = t.elapsed_time().count();
    infer_result->infer_time = infer_time;
    infer_result->input = golden_inputs[i];
    infer_result->pred = y_pred;
    infer_result->ans = float(sin(golden_inputs[i]));
    infer_result->diff = float(abs(sin(golden_inputs[i]) - y_pred));
    infer_result->quant = quant;
    out_queue.try_put(infer_result);
    
    // MicroPrintf("inference with %ld (ns)", infer_time);
    // MicroPrintf("  input\t%f\n  pred\t%f\n  ans\t%f\n  diff\t%f\n", golden_inputs[i] , y_pred , sin(golden_inputs[i]) , abs(sin(golden_inputs[i]) - y_pred));
    // delete infer_result;

  }
//   MicroPrintf("LoadFloatModelAndPerformInference() OK");
  return kTfLiteOk;
}


Thread infer1(osPriorityNormal, 50000, nullptr, nullptr);
Thread infer2(osPriorityNormal, 50000, nullptr, nullptr);
// Queue<tflite::MicroInterpreter, 2> queue;



int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  bool a = 0,b = 1;

  in_queue.try_put(&a);
  in_queue.try_put(&b);
  infer1.start(LoadModelAndPerformInference);
  infer1.join();
  infer2.start(LoadModelAndPerformInference);
  infer2.join();

//   TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(false));
//   TF_LITE_ENSURE_STATUS(LoadModelAndPerformInference());

//   TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(true));
//   TF_LITE_ENSURE_STATUS(LoadModelAndPerformInference(true));
  struct result *infer_result = new result;
  while(out_queue.try_get(&infer_result)){
    MicroPrintf("inference with %ld (ns) and quant flag %d", infer_result->infer_time , infer_result->quant);
    MicroPrintf("  input\t%f\n  pred\t%f\n  ans\t%f\n  diff\t%f\n", infer_result->input , infer_result->pred , infer_result->ans , infer_result->diff);
  }
  
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  
  return kTfLiteOk;
}

