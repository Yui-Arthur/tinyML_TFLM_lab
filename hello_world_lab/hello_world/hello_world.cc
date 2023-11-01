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

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<1>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency(bool quant) {
  tflite::MicroProfiler profiler;
  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 5000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

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

TfLiteStatus LoadModelAndPerformInference(bool quant) {
  MicroPrintf("LoadModelAndPerformInference() Start");
  
  const unsigned char *model_data_ptr;
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

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  float output_scale = 0;
  int output_zero_point = 0;

  if(quant)
    output_scale = output->params.scale , output_zero_point = output->params.zero_point; 

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());



  constexpr int kNumTestValues = 4;
  float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f};

  for (int i = 0; i < kNumTestValues; ++i) {
    if(quant)
        interpreter.input(0)->data.int8[0] = (int)golden_inputs[i];
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
        interpreter.output(0)->data.f[0];
    
    MicroPrintf("inference with %ld (ns)", std::chrono::duration_cast<std::chrono::microseconds>(t.elapsed_time()).count());
    MicroPrintf("  input\t%f\n  pred\t%f\n  ans\t%f\n  diff\t%f\n", golden_inputs[i] , y_pred , sin(golden_inputs[i]) , abs(sin(golden_inputs[i]) - y_pred));

  }
  MicroPrintf("LoadFloatModelAndPerformInference() OK");
  return kTfLiteOk;
}


int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(false));
  TF_LITE_ENSURE_STATUS(LoadModelAndPerformInference(false));

  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency(true));
  TF_LITE_ENSURE_STATUS(LoadModelAndPerformInference(true));
  
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
