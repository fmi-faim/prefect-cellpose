import gc
import json
import os
import re
import threading
from copy import copy
from os.path import exists, join
from typing import Any, Optional

import cellpose.models
import numpy as np
from cellpose import models
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.numpy.NumpyTarget import NumpyTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
from faim_prefect.mamba import log_infrastructure
from faim_prefect.parallelization.utils import wait_for_task_run
from faim_prefect.parameter import User
from prefect import flow, get_client, get_run_logger, task
from prefect.client.schemas import FlowRun
from prefect.context import TaskRunContext
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from pydantic import BaseModel


class InputData(BaseModel):
    input_dir: str = "/home/tibuch/Data/broad/nuclei_U2OS/images/"
    pattern: str = ".*.tif"
    axes: str = "YX"
    xy_pixelsize_um: float = 0.134
    z_spacing_um: float = 1.0


class OutputFormat(BaseModel):
    output_dir: str = "/home/tibuch/Gitrepos/prefect-cellpose/test-output/"
    imagej_compatible: bool = True


class Cellpose(BaseModel):
    model: str = "nuclei"
    seg_channel: int = 0
    nuclei_channel: int = 0
    diameter: float = 40.0
    flow_threshold: float = 0.4
    cell_probability_threshold: float = 0.0
    resample: bool = True
    save_labeling: bool = True
    save_flows: bool = False
    do_3D: bool = False


@task(cache_key_fn=task_input_hash, refresh_cache=True)
def list_images(
    input_dir: str = "/path/to/input_dir",
    pattern: str = ".*.tif",
    pixel_resolution_um: float = 0.134,
    z_spacing_um: float = 0.5,
    axes="CYX",
):
    pattern_re = re.compile(pattern)
    images: list[ImageSource] = []
    for entry in os.scandir(input_dir):
        if entry.is_file():
            if pattern_re.fullmatch(entry.name):
                metadata = {
                    "axes": axes,
                    "unit": "micron",
                }

                if "Z" in axes:
                    metadata["spacing"] = z_spacing_um

                images.append(
                    ImageSource.from_path(
                        entry.path,
                        metadata=metadata,
                        resolution=[
                            1e4 / pixel_resolution_um,
                            1e4 / pixel_resolution_um,
                        ],
                    )
                )

    return images


def exlude_semaphore_and_model_task_input_hash(
    context: "TaskRunContext", arguments: dict[str, Any]
) -> Optional[str]:
    hash_args = {}
    for k, item in arguments.items():
        if (not isinstance(item, threading.Semaphore)) and (
            not isinstance(item, cellpose.models.CellposeModel)
        ):
            hash_args[k] = item

    return task_input_hash(context, hash_args)


def prepare_data_for_single_segmentation_channel(
    img: ImageSource,
    seg_channel: int,
):
    axes = img.get_metadata()["axes"]

    if "C" in axes:
        assert 3 <= len(axes) <= 4, f"{axes} not supported."
        img_data = np.moveaxis(img.get_data(), axes.index("C"), 0)
        return img_data[seg_channel : seg_channel + 1]
    else:
        assert 2 <= len(axes) <= 3, f"{axes} not supported."
        return img.get_data()[np.newaxis]


def prepare_data_for_two_segmentation_channels(
    img: ImageSource,
    seg_channel: int,
    nuc_channel: int,
):
    axes = img.get_metadata()["axes"]

    if "C" in axes:
        assert 3 <= len(axes) <= 4, f"{axes} not supported."
        img_data = np.moveaxis(img.get_data(), axes.index("C"), 0)
        return np.concatenate(
            [
                img_data[seg_channel : seg_channel + 1],
                img_data[nuc_channel : nuc_channel + 1],
            ],
            axis=0,
        )
    else:
        raise RuntimeError(f"Channel axis is missing: {axes}")


@task(cache_key_fn=exlude_semaphore_and_model_task_input_hash)
def predict(
    img: ImageSource,
    model,
    cellpose_parameter: Cellpose,
    output_format: OutputFormat,
    gpu_sem: threading.Semaphore,
):
    logger = get_run_logger()

    metadata = img.get_metadata()

    if cellpose_parameter.seg_channel == cellpose_parameter.nuclei_channel:
        img_data = prepare_data_for_single_segmentation_channel(
            img, seg_channel=cellpose_parameter.seg_channel
        )
    else:
        img_data = prepare_data_for_two_segmentation_channels(
            img,
            seg_channel=cellpose_parameter.seg_channel,
            nuc_channel=cellpose_parameter.nuclei_channel,
        )

    try:
        gpu_sem.acquire()
        mask, flows, _ = model.eval(
            img_data,
            diameter=cellpose_parameter.diameter,
            flow_threshold=cellpose_parameter.flow_threshold,
            cellprob_threshold=cellpose_parameter.cell_probability_threshold,
            channels=[0, 1],
            channel_axis=0,
            compute_masks=cellpose_parameter.save_labeling,
            do_3D=cellpose_parameter.do_3D,
        )
    except Exception as e:
        raise e
    finally:
        gpu_sem.release()

    output_metadata = copy(metadata)
    if "C" in output_metadata["axes"]:
        mask = mask[np.newaxis]
        mask = np.moveaxis(mask, 0, output_metadata["axes"].index("C"))

    if cellpose_parameter.save_labeling and cellpose_parameter.save_flows:
        pred_mask = ImageTarget.from_path(
            join(output_format.output_dir, img.get_name() + ".tif"),
            metadata=output_metadata,
            resolution=img.get_resolution(),
            imagej=output_format.imagej_compatible,
        )
        pred_mask.set_data(mask.astype(np.uint16))

        pred_flows = []
        for i, cellpose_flow in enumerate(flows):
            pred_flow = NumpyTarget.from_path(
                join(output_format.output_dir, img.get_name() + f"_flow-" f"{i}.npy")
            )
            pred_flow.set_data(cellpose_flow)
            pred_flows.append(pred_flow)

        return pred_mask, pred_flows
    elif cellpose_parameter.save_labeling and not cellpose_parameter.save_flows:
        pred_mask = ImageTarget.from_path(
            join(output_format.output_dir, img.get_name() + ".tif"),
            metadata=output_metadata,
            resolution=img.get_resolution(),
            imagej=output_format.imagej_compatible,
        )
        pred_mask.set_data(mask.astype(np.uint16))

        return pred_mask, None
    elif not cellpose_parameter.save_labeling and cellpose_parameter.save_flows:
        pred_flows = []
        for i, cellpose_flow in enumerate(flows):
            pred_flow = NumpyTarget.from_path(
                join(output_format.output_dir, img.get_name() + f"_flow-" f"{i}.npy"),
                imagej=output_format.imagej_compatible,
            )
            pred_flow.set_data(cellpose_flow)
            pred_flows.append(pred_flow)

        return None, pred_flows
    else:
        logger.error(
            "No prediction is saved. Please select 'save_labeling' or 'save_flows'."
        )


@flow(
    name="Run cellpose inference",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage=LocalFileSystem.load("prefect-cellpose"),
)
def run_cellpose_tiff(
    image_dicts: list[dict],
    cellpose_parameter: Cellpose = Cellpose(),
    output_format: OutputFormat = OutputFormat(),
):
    if "data_hash" in image_dicts[0].keys():
        images = [ImageTarget(**d) for d in image_dicts]
    else:
        images = [ImageSource(**d) for d in image_dicts]
    gpu_sem = threading.Semaphore(1)

    model = models.CellposeModel(gpu=True, pretrained_model=cellpose_parameter.model)

    def unpack(future):
        labeling, flows = future.result()
        return {"mask": labeling, "flows": flows}

    predictions: list[ImageTarget] = []
    buffer = []
    for img in images:
        gc.collect()
        buffer.append(
            predict.submit(
                img=img,
                model=model,
                cellpose_parameter=cellpose_parameter,
                output_format=output_format,
                gpu_sem=gpu_sem,
            )
        )

        wait_for_task_run(
            results=predictions,
            buffer=buffer,
            max_buffer_length=2,
            result_insert_fn=unpack,
        )

    wait_for_task_run(
        results=predictions,
        buffer=buffer,
        max_buffer_length=0,
        result_insert_fn=unpack,
    )

    return predictions


@task(cache_key_fn=task_input_hash)
def submit_flows(
    images: list[ImageSource],
    cellpose_parameter: Cellpose = Cellpose(),
    output_format: OutputFormat = OutputFormat(),
    chunk_size: int = 500,
):
    image_dicts = [i.serialize() for i in images]

    predictions = []
    for i in range(0, len(images), chunk_size):
        run: FlowRun = run_deployment(
            name="Run cellpose inference/default",
            parameters={
                "image_dicts": image_dicts[i : min(i + chunk_size, len(images))],
                "cellpose_parameter": cellpose_parameter,
                "output_format": output_format,
            },
            client=get_client(),
        )
        predictions.extend(run.state.result())

    return predictions


def validate_parameters(
    user: User,
    run_name: str,
    input_data: InputData,
    output_format: OutputFormat,
    cellpose_parameters: Cellpose,
    parallelization: int,
    n_imgs_per_job: int,
):
    logger = get_run_logger()
    base_dir = LocalFileSystem.load("base-output-directory").basepath
    group = user.group.value
    if not exists(join(base_dir, group)):
        logger.error(f"Group '{group}' does not exist in '{base_dir}'.")

    if not exists(input_data.input_dir):
        logger.error(f"Input directory '{input_data.input_dir}' does not " f"exist.")

    if not bool(re.match("[XYZC]+", input_data.axes)):
        logger.error("Axes is only allowed to contain 'XYZC'.")

    os.makedirs(output_format.output_dir, exist_ok=True)

    if parallelization < 1:
        logger.error(f"parallelization = {parallelization}. Must be >= 1.")

    if n_imgs_per_job < 1:
        logger.error(f"n_imgs_per_job = {n_imgs_per_job}. Must be >= 1.")

    run_dir = join(
        base_dir,
        group,
        user.name,
        "prefect-runs",
        "cellpose",
        run_name.replace(" ", "-"),
    )

    parameters = {
        "user": {
            "name": user.name,
            "group": group,
        },
        "input_data": input_data.dict(),
        "output_format": output_format.dict(),
        "cellpose_parameters": cellpose_parameters.dict(),
        "parallelization": parallelization,
        "n_imgs_per_job": n_imgs_per_job,
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(join(run_dir, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters, indent=4))

    return run_dir


with open(
    join("flows/cellpose_tiff.md"),
    encoding="UTF-8",
) as f:
    description = f.read()


@flow(
    name="Cellpose inference [tiff]",
    description=description,
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage=LocalFileSystem.load("prefect-cellpose"),
)
def cellpose_tiff(
    user: User,
    run_name: str,
    input_data: InputData = InputData(),
    output_format: OutputFormat = OutputFormat(),
    cellpose_parameters: Cellpose = Cellpose(),
    parallelization: int = 1,
    n_imgs_per_job: int = 500,
) -> list[ImageTarget]:
    run_dir = validate_parameters(
        user=user,
        run_name=run_name,
        input_data=input_data,
        output_format=output_format,
        cellpose_parameters=cellpose_parameters,
        parallelization=parallelization,
        n_imgs_per_job=n_imgs_per_job,
    )

    logger = get_run_logger()
    logger.info(f"Run logs are written to: {run_dir}")
    logger.info(f"Segmentations are saved in:" f" {output_format.output_dir}")

    imgs = list_images(
        input_dir=input_data.input_dir,
        pattern=input_data.pattern,
        pixel_resolution_um=input_data.xy_pixelsize_um,
        z_spacing_um=input_data.z_spacing_um,
        axes=input_data.axes,
    )

    n_imgs = len(imgs)
    split = n_imgs // parallelization
    start = 0

    runs = []
    for i in range(parallelization - 1):
        runs.append(
            submit_flows.submit(
                images=imgs[start : start + split],
                cellpose_parameter=cellpose_parameters,
                output_format=output_format,
                chunk_size=n_imgs_per_job,
            )
        )
        start = start + split

    runs.append(
        submit_flows.submit(
            images=imgs[start:],
            cellpose_parameter=cellpose_parameters,
            output_format=output_format,
            chunk_size=n_imgs_per_job,
        )
    )

    segmentations = []
    for run in runs:
        segmentations.extend(run.result())

    log_infrastructure(run_dir)

    return segmentations


if __name__ == "__main__":
    cellpose_tiff()
