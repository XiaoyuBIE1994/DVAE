oarsub -S /scratch/virgo/xbie/Code/rvae-speech/script_RNN.sh \
        -l/host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' AND not host like 'gpu5-perception.inrialpes.fr' AND not host like 'gpu6-perception.inrialpes.fr' " \
        -t besteffort \
        -t idempotent