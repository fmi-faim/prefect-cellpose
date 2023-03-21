# Build
```shell
prefect deployment build flows/cellpose_2D_tiff.py:cellpose_2D_tiff -n "default" -q slurm -sb github/prefect-cellpose --skip-upload -o deployments/cellpose_2D_tiff.yaml  -ib process/cellpose-orchestration -t segmentation -t 2D

prefect deployment build flows/cellpose_2D_tiff.py:run_cellpose_2D_tiff -n "default" -q slurm -sb github/prefect-cellpose --skip-upload -o deployments/run_cellpose_2D_tiff.yaml  -ib process/cellpose-gpu
```

# Apply
```shell
prefect deployment apply deployments/*.yaml
```
