#! /bin/bash

for filename in /home/frederik/Dropbox/Astro/3DPDR/3D-PDR/frederik_output/*_3D-PDR.txt
do
    file=${filename#"/home/frederik/Dropbox/Astro/3DPDR/3D-PDR/frederik_output/"}
    cp ${filename} ${file}
done
