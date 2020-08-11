oarsub -S /scratch/virgo/xbie/Code/dvae-speech/script/train_kvae.sh \
        -l /host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis' AND not host like 'gpu1-perception.inrialpes.fr' AND not host like 'gpu2-perception.inrialpes.fr' AND not host like 'gpu3-perception.inrialpes.fr'" \
        -t besteffort \
        -t idempotent

# oarsub -S /scratch/virgo/xbie/Code/dvae-speech/script/train_kvae_debug2.sh \
#         -l /host=1/gpudevice=1,walltime=90:00:00 \
#         -p "cluster='perception' OR cluster='kinovis' AND not host like 'gpu1-perception.inrialpes.fr' AND not host like 'gpu2-perception.inrialpes.fr' AND not host like 'gpu3-perception.inrialpes.fr'" \
#         -t besteffort \
#         -t idempotent

# oarsub -S /scratch/virgo/xbie/Code/dvae-speech/script/train_kvae_debug3.sh \
#         -l /host=1/gpudevice=1,walltime=90:00:00 \
#         -p "cluster='perception' OR cluster='kinovis' AND not host like 'gpu1-perception.inrialpes.fr' AND not host like 'gpu2-perception.inrialpes.fr' AND not host like 'gpu3-perception.inrialpes.fr'" \
#         -t besteffort \
#         -t idempotent

# oarsub -S /scratch/virgo/xbie/Code/dvae-speech/script/train_kvae_debug4.sh \
#         -l /host=1/gpudevice=1,walltime=90:00:00 \
#         -p "cluster='perception' OR cluster='kinovis' AND not host like 'gpu1-perception.inrialpes.fr' AND not host like 'gpu2-perception.inrialpes.fr' AND not host like 'gpu3-perception.inrialpes.fr'" \
#         -t besteffort \
#         -t idempotent

oarsub -S /scratch/virgo/xbie/Code/dvae-speech/script/train_dmm.sh \
        -l /host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' AND host like 'gpu5-perception.inrialpes.fr' OR host like 'gpu6-perception.inrialpes.fr' OR host like 'gpu7-perception.inrialpes.fr'" \
        -t besteffort \
        -t idempotent

oarsub -S /scratch/virgo/xbie/Code/dvae-speech/script/train_storn.sh \
        -l /host=1/gpudevice=1,walltime=90:00:00 \
        -p "cluster='perception' OR cluster='kinovis'" \
        -t besteffort \
        -t idempotent

oarsub -I -l /host=1/gpudevice=1 -p "cluster='perception' AND not host like 'gpu5-perception.inrialpes.fr' AND not host like 'gpu6-perception.inrialpes.fr' AND not host like 'gpu7-perception.inrialpes.fr'"

oarsub -I -l /host=1/gpudevice=1 -p "cluster='perception' AND not host like 'gpu5-perception.inrialpes.fr'"