# Build
```shell
prefect deployment build flows/cellpose_tiff.py:cellpose_tiff -n "default" -q slurm -sb github/prefect-cellpose --skip-upload -o deployments/cellpose_tiff.yaml  -ib process/prefect-cellpose-orchestration -t segmentation -t 2D

prefect deployment build flows/cellpose_tiff.py:run_cellpose_tiff -n "default" -q slurm -sb github/prefect-cellpose --skip-upload -o deployments/run_cellpose_tiff.yaml  -ib process/prefect-cellpose-gpu
```

# Apply
```shell
prefect deployment apply deployments/*.yaml
```
