python -m data.download --dataset all
python -m data.process_gt_linemod
python -m data.process_gt_tless
python -m data.render_templates --dataset linemod --disable_output --num_workers 4
python -m data.render_templates --dataset tless --disable_output --num_workers 4
python -m data.crop_image_linemod
python -m data.create_dataframe_linemod
python -m data.create_dataframe_tless --split train
python -m data.create_dataframe_tless --split test