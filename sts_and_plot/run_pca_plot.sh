
cd ${SCRIPT_DIR}
vector_path=${VECTOR_PATH}
output_path=${OUTPUT_PATH}

for iso_type in base whitening
do
vector_type=static
output_directory=${output_path}/${vector_type}-${iso_type}
mkdir -p ${output_directory}
python whitening_effect.py --vector_type ${vector_type} --vector_path ${vector_path} --dim 300 --sample_size 100000 --output_path ${output_directory} --iso_type ${iso_type}
done

data_path=${DATA_PATH}
for iso_type in base whitening
do
vector_type=contextualized
output_directory=${output_path}/${vector_type}-${iso_type}
mkdir -p ${output_directory}
python whitening_effect.py --vector_type ${vector_type} --model_name_or_path bert-base-cased --dim 768 --data_path ${data_path} --sample_size 100000 --output_path ${output_directory} --remove_selected_tokens --batch_size 128 --reg_same_word  --iso_type ${iso_type}
done


