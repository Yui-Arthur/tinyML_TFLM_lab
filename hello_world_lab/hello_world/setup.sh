git clone https://github.com/tensorflow/tflite-micro
cd tflite-micro
# git am ../patches/0001-profiler.patch

# Third party files need to be downloaded to build tflite-micro.
# For instance, Flatbuffers is not natively included in tflite-micro.
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads

# Returns all necessary header and source files
OPTIONS="OPTIMIZED_KERNEL_DIR=cmsis_nn"
if [ "$CMSIS" = false ] ; then
  OPTIONS=""
fi
HEADERS=$(make -f tensorflow/lite/micro/tools/make/Makefile $OPTIONS list_library_headers)
SOURCES=$(make -f tensorflow/lite/micro/tools/make/Makefile $OPTIONS list_library_sources)
TP_HEADERS=$(make -f tensorflow/lite/micro/tools/make/Makefile $OPTIONS list_third_party_headers)
TP_SOURCES=$(make -f tensorflow/lite/micro/tools/make/Makefile $OPTIONS list_third_party_sources)
HEADERS_SEP=$(echo $HEADERS | tr " " "\n")
SOURCES_SEP=$(echo $SOURCES | tr " " "\n")
TP_HEADERS_SEP=$(echo $TP_HEADERS | tr " " "\n")
TP_SOURCES_SEP=$(echo $TP_SOURCES | tr " " "\n")
cd ..

# Copy files into project folder
for file in $HEADERS_SEP
do
  DIR=$(dirname $file)
  BASENAME=$(basename $file)
  mkdir -p $DIR
  cp tflite-micro/$file $DIR
done

for file in $SOURCES_SEP
do
  DIR=$(dirname $file)
  BASENAME=$(basename $file)
  mkdir -p $DIR
  cp tflite-micro/$file $DIR
done

for file in $TP_HEADERS_SEP
do
  DIR=$(dirname $file)
  BASENAME=$(basename $file)
  mkdir -p $DIR
  cp tflite-micro/$file $DIR
done

for file in $TP_SOURCES_SEP
do
  DIR=$(dirname $file)
  BASENAME=$(basename $file)
  mkdir -p $DIR
  cp tflite-micro/$file $DIR
done

rm -r tflite-micro