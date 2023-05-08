python -m src.scripts.download
python -m src.scripts.process_lm_gt
python -m src.scripts.render_template dataset_to_render=lm
python -m src.scripts.render_template dataset_to_render=tless
python -m src.scripts.render_template dataset_to_render=hb
python -m src.scripts.compute_neighbors