singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/dvae.sif \
                python /mnt/xbie/Code/dvae-speech/train_model.py \
                /mnt/xbie/Code/dvae-speech/config/cfg_dsae.ini

