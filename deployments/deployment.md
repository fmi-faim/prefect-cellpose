# Build
```shell
prefect deployment build flows/cellpose_tiff.py:cellpose_tiff -n "default" -q slurm -sb github/prefect-cellpose --skip-upload -o deployments/cellpose_tiff.yaml  -ib process/prefect-cellpose-orchestration -t segmentation -t orchestration

prefect deployment build flows/cellpose_tiff.py:run_cellpose_tiff -n "default" -q slurm -sb github/prefect-cellpose --skip-upload -o deployments/run_cellpose_tiff.yaml  -ib process/prefect-cellpose-gpu -t segmentation
```

# Apply
```shell
prefect deployment apply deployments/*.yaml
```
