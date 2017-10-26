## Magritte plot scripts
------------------------

### How to use:

Run the python script to plot the respective output, e.g.

```
python plot_abun.py date_stamp tag
```
where ``date_stamp`` is of the form YYYY-mm-dd_HH_mm_ss and ``tag`` is the last word before .txt in the file name (default is final).

The plot will be saved in the plots directory in the output folder of the run indicated by the date stamp. For the example above this will be ``
YYYY-mm-dd_HH_mm_ss_output/plots/abundances_tag.png
``.

To plot all output of a certain run with a certain tag one can use
```
bash plot_all.sh date_stamp tag
```
