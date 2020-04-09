oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script/train_vrnn.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent
