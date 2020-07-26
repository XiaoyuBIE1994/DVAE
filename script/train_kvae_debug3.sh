singularity exec \
                --nv \
                --bind /scratch/virgo/xbie/:/mnt/xbie/ \
                /scratch/virgo/xbie/Simgs/test \
                python /mnt/xbie/Code/dvae-speech/train_model_kvae.py \
                /mnt/xbie/Code/dvae-speech/config/kvae_debug/cfg_kvae_50-30.ini
