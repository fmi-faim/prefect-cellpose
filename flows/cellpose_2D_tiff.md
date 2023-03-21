# Cellpose for 2D tiff files
Applies a [cellpose](https://github.com/MouseLand/cellpose) model to each tiff file in a directory.

## Input Format
The provided tiff-files must be 2D and can have an optional channel axis.

## Flow Parameters
* `user`:
    * `name`: Name of the user.
    * `group`: Group name of the user.
    * `run_name`: Name of processing run.
* `input_data`:
    * `input_dir`: Input directory containing the 2D+Channel tiff files.
    * `pattern`: A pattern to filter the tiff files.
    * `axes`: String indicating the axes order of the tiff files.
    * `xy_pixelsize_um`: The pixel-size in micrometers.
* `output_format`:
    * `output_dir`: Path to the output directory.
    * `imagej_compatible`: If the output images should be written ImageJ compatible.
* `cellpose_parameters`: See [cellpose-documentation](https://cellpose.readthedocs.io/en/latest/settings.html#settings) for more details.
    * `model`: Name of the cellpose model.
    * `seg_channel`: Channel to segment
    * `nuclei_channel`: Channel with the nuclei signal. If no nuclei-information should be used, set this channel equal to `seg_channel`.
    * `diameter`: Average object diameter.
    * `flow_threshold`: Maximum allowed error of the flows for each mask.
    * `cell_probability_threshold`: Threshold used to determine ROIs.
    * `resample`: Setting this to `true` will create smoother segmentations.
    * `remove_touching_border`: Remove ROIs touching the image border.
* `parallelization`: How many cellpose jobs are running in parallel. This number if optimized for our setup. __Do not change this.__
* `n_imgs_per_job`: Number of images processed by each cellpose job. This number if optimized for our setup. __Do not change this.__

## Output Format
The cellpose segmentations are saved as 16bit unsigned integer tiff file. The file-names have a 17 character long suffix appended which is used by Prefect to identify unique results.

## Citation
If you use this flow please cite the [Cellpose paper](https://www.nature.com/articles/s41592-020-01018-x.epdf?sharing_token=yrCA1mB-y9TR8-RC8w_CPdRgN0jAjWel9jnR3ZoTv0Ms-A3TbUG5N7s_6d3I7lMImMDE6cyl-17ubiknffX50r-dX1un0XSIQ2PGYWsCV1du16fIaipcHNxste8FMByEL75Ek_S2_UEVkSk7lCFllWEVogGWJwmQkBC9uKq9UEA%3D):
```text
@article{stringer2021cellpose,
  title={Cellpose: a generalist algorithm for cellular segmentation},
  author={Stringer, Carsen and Wang, Tim and Michaelos, Michalis and Pachitariu, Marius},
  journal={Nature methods},
  volume={18},
  number={1},
  pages={100--106},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```
