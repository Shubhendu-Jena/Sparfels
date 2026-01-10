export PYTHONPATH="${PYTHONPATH}:./utils"
export CUDA_VISIBLE_DEVICES=0
set=0
data_dir="./dtu"
scenes=(scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122)
n_view=3
iter=1000

for scene in ${scenes[@]}
do  
python evaluation/save_depth_normals_render.py --set $set --dtu_input_dir ${data_dir}_${set}/$scene --mesh_dir ./output_set_${set}_full/$scene/train/ours_$iter 
done
