# Performance tests
-------------------

For Feautrier solver

To get assembly code:
```
icc -qopenmp -std=c++11 -I /home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src -o example_feautrier.exe example_feautrier.cpp -fcode-asm -Faasm.s
```

```
icc -O3 -qopenmp -std=c++11 -qopt-report3 -I /home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/src -o example_feautrier.exe example_feautrier.cpp
```
