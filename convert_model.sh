model_path=$1
project_path=$2

xxd -i $model_path > $project_path/model.cc
echo -ne "#include \"ic/ic_model_data.h\"\nalignas(8)\n" > $project_path/ic/ic_model_data.cc
cat $project_path/model.cc >> $project_path/ic/ic_model_data.cc
sed -i -E 's/(unsigned\s.*\s).*(_len|\[\])/const \1model\2/g' $project_path/ic/ic_model_data.cc