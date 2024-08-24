# Kitti360 nighttime
python render.py --config configs/kitti/kitti_687_743.txt \
     --light configs/light/kitti_687_743.yaml --relight night --add_flares
# python render.py --config configs/kitti/kitti_1538_1601.txt \
#      --light configs/light/kitti_1538_1601.yaml --relight night --add_flares
# python render.py --config configs/kitti/kitti_1720_1783.txt \
#      --light configs/light/kitti_1720_1783.yaml --relight night --add_flares
# python render.py --config configs/kitti/kitti_3970_4010.txt \
#      --light configs/light/kitti_3970_4010.yaml --relight night --add_flares
# python render.py --config configs/kitti/kitti_6040_6103.txt \
#      --light configs/light/kitti_6040_6103.yaml --relight night --add_flares

# Waymo nighttime
# python render.py --config configs/waymo/waymo_seq0.txt \
#      --light configs/light/waymo_seq0.yaml --relight night --add_flares
# python render.py --config configs/waymo/waymo_seq2.txt \
#      --light configs/light/waymo_seq2.yaml --relight night --add_flares
# python render.py --config configs/waymo/waymo_seq3.txt \
#      --light configs/light/waymo_seq3.yaml --relight night --add_flares
# python render.py --config configs/waymo/waymo_seq4.txt \
#      --light configs/light/waymo_seq4.yaml --relight night --add_flares
# python render.py --config configs/waymo/waymo_seq5.txt \
#      --light configs/light/waymo_seq5.yaml --relight night --add_flares

# Sun move
# python render_static.py --config configs/kitti/kitti_4474_4537.txt \
#      --light configs/light/kitti_4474_4537.yaml --relight sun_move --video_len 300
