/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C++ implementations of submitter_implemented.h

#include "api/submitter_implemented.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "api/internally_implemented.h"
#include "mbed.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
#include "ic/ic_inputs.h"
#include "ic/ic_model_data.h"
#include "ic/ic_model_settings.h"

#include "img.h"

UnbufferedSerial pc(USBTX, USBRX);
DigitalOut timestampPin(D7);

// AAML tinyML Lab: ARENA_SIZE is defined in tensorflow/lite/c/common.h
constexpr int kTensorArenaSize = ARENA_SIZE;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// #define QUANT_MODEL false
// #define IN_IO_TYPE float
// #define OUT_IO_TYPE float

tflite::MicroModelRunner<IN_IO_TYPE, OUT_IO_TYPE, OP_NUM> *runner;

// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
//   runner->SetZeroInput();
    int input_length = runner->SetInput((IN_IO_TYPE*)img_bin);
}

// Add to this method to return real inference results.
void th_results() {
  const int nresults = 10;
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[");
  int kCategoryCount = 10;
  float maxx = -1;
  int maxx_idx = -1;
  for (size_t i = 0; i < kCategoryCount; i++) {
    float converted =
    #if QUANT_MODEL
        DequantizeInt8ToFloat(runner->GetOutput()[i], runner->output_scale(),
                              runner->output_zero_point());
    #else
        runner->GetOutput()[i];
    #endif // IO_TYPE == int8_t
    if(converted > maxx) maxx = converted , maxx_idx = i;
    th_printf("%0.6f", converted);
    if (i < (nresults - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
  th_printf("max class : %d , conf : %f\r\n" , maxx_idx , maxx);
  th_printf("m-arena-size : %d bytes\r\n" , runner->arena_size() );
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() { runner->Invoke(); }

/// \brief optional API.
void th_final_initialize(void) {
  static tflite::MicroMutableOpResolver<OP_NUM> resolver;
//   if(QUANT_MODEL)
//     op_size = 8;


  resolver.AddAdd();
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddAveragePool2D();
  if(QUANT_MODEL)
    resolver.AddQuantize();

  static tflite::MicroModelRunner<IN_IO_TYPE, OUT_IO_TYPE, OP_NUM> model_runner(
      model, resolver, tensor_arena, kTensorArenaSize);
  runner = &model_runner;
}
void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }
void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}

char th_getchar() { return getchar(); }

void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE==1
  pc.baud(9600);
#else
  pc.baud(115200);
#endif
}

uint32_t th_timestamp(void) {
  # if EE_CFG_ENERGY_MODE==1
  timestampPin = 0;
  for (int i=0; i<100'000; ++i) {
    asm("nop");
  }
  timestampPin = 1;
 #else
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  microSeconds = us_ticker_read();
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
  return microSeconds;
  #endif
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
