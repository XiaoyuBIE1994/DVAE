oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script_train.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' " \
        -t besteffort \
        -t idempotent