target_file=model_data
while getopts 'p:qh' flag; do
case $flag in
    h) 
        echo "-q to set the Quant Model"
        echo "-p to set the Input Model Path"
        exit 0
    ;;
    q)
        target_file=model_data_quant
    ;; 
    p)
        path=$OPTARG
    ;;
    ?)
        echo "Argument Error"
        exit 1
    ;;
    esac
done

echo $path
xxd -i $path > model.cc
echo -ne "#include \"$target_file.h\"\nalignas(8)\n" > $target_file.cc
cat model.cc >> $target_file.cc
sed -i -E "s/(unsigned\s.*\s).*(_len|\[\])/const \1${target_file}\2/g" $target_file.cc
rm model.cc