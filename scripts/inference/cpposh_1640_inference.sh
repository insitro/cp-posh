# CP-DINO 1640
CHANNEL_ORDER_IN_MODEL_INPUT_ORDER=(0 1 2 3 4)
NORMALIZE_MEAN_IN_MODEL_INPUT_ORDER=(4.89883638 7.92730405 9.47751658 10.16663979 4.09869388)
NORMALIZE_STD_IN_MODEL_INPUT_ORDER=(15.66504555 18.897298 19.45658873 20.57687824 15.01226514)

python cp_posh/ssl/inference.py \
    --cell_image_dataframe_path /path/to/cell_image_dataframe \
    --model_name cp-dino-1640 \
    --channel_order_in_model_input_order "${CHANNEL_ORDER_IN_MODEL_INPUT_ORDER[@]}" \
    --normalize_mean_in_model_input_order "${NORMALIZE_MEAN_IN_MODEL_INPUT_ORDER[@]}" \
    --normalize_std_in_model_input_order "${NORMALIZE_STD_IN_MODEL_INPUT_ORDER[@]}" \
    --output_path /path/to/output/parquet