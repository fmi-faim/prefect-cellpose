import os
import re
import threading
from os.path import join
from typing import Any, Optional

import cellpose.models
import numpy as np
import skimage
from cellpose import models
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.Serializer import cpr_serializer
from cpr.utilities.utilities import task_input_hash
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
    remove_touching_border: bool = False


def remove_border_objects(image: np.ndarray) -> np.ndarray:
    """Remove objects touching the border of the image."""
    shape = {0, *image.shape}
    for prop in skimage.measure.regionprops(image):
        if bool(shape & {*prop.bbox}):
            image = np.where(image == prop.label, 0, image)
    return image


@task(cache_key_fn=task_input_hash)
def list_images(
    input_dir: str = "/path/to/input_dir",
    pattern: str = ".*.tif",
    pixel_resolution_um: float = 0.134,
    axes="CYX",
):
    pattern_re = re.compile(pattern)
    images: list[ImageSource] = []
    for entry in os.scandir(input_dir):
        if entry.is_file():
            if pattern_re.fullmatch(entry.name):
                images.append(
                    ImageSource.from_path(
                        entry.path,
                        metadata={
                            "axes": axes,
                            "unit": "micron",
                        },
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
            not isinstance(item, cellpose.models.Cellpose)
        ):
            hash_args[k] = item

    return task_input_hash(context, hash_args)


@task(cache_key_fn=exlude_semaphore_and_model_task_input_hash)
def predict(
    img: ImageSource,
    model,
    cellpose_parameter,
    output_format,
    gpu_sem: threading.Semaphore,
):
    logger = get_run_logger()

    metadata = img.get_metadata()
    axes = metadata["axes"]

    if cellpose_parameter.seg_channel == cellpose_parameter.nuclei_channel:
        img_data = img.get_data()
        if img_data.ndim == 3:
            img_data = np.moveaxis(img_data, axes.index("C"), 0)
            img_data = img_data[cellpose_parameter.seg_channel]
            img_data = img_data[np.newaxis]
        elif img_data.ndim == 2:
            img_data = img_data[np.newaxis]
        else:
            logger.error(f"{img_data.ndim} not supported.")
    else:
        img_data = img.get_data()
        if img_data.ndim == 3:
            img_data = np.moveaxis(img_data, axes.index("C"), 0)
            img_data = np.concatenate(
                [
                    img_data[cellpose_parameter.seg_channel],
                    img_data[cellpose_parameter.nuclei_channel],
                ],
                axis=0,
            )
        else:
            logger.error(f"{img_data.ndim} not supported.")

    metadata["imagej"] = output_format.imagej_compatible

    try:
        gpu_sem.acquire()
        mask, _, _, _ = model.eval(
            img_data,
            diameter=cellpose_parameter.diameter,
            flow_threshold=cellpose_parameter.flow_threshold,
            cellprob_threshold=cellpose_parameter.cell_probability_threshold,
            channels=[0, 1],
            channel_axis=0,
        )
    except Exception as e:
        raise e
    finally:
        gpu_sem.release()

    pred = ImageTarget.from_path(
        join(output_format.output_dir, img.get_name()),
        metadata=metadata,
        resolution=img.get_resolution(),
    )

    if cellpose_parameter.remove_touching_border:
        mask = remove_border_objects(mask)

    pred.set_data(mask.astype(np.uint16))

    return pred


@flow(
    name="Run cellpose inference 2D",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage=LocalFileSystem.load("cellpose"),
)
def run_cellpose_2D_tiff(
    image_dicts: list[dict],
    cellpose_parameter: Cellpose = Cellpose(),
    output_format: OutputFormat = OutputFormat(),
):
    images = [ImageSource(**d) for d in image_dicts]
    gpu_sem = threading.Semaphore(1)

    model = models.Cellpose(gpu=True, model_type=cellpose_parameter.model)

    predictions: list[ImageTarget] = []
    buffer = []
    for img in images:
        buffer.append(
            predict.submit(
                img=img,
                model=model,
                cellpose_parameter=cellpose_parameter,
                output_format=output_format,
                gpu_sem=gpu_sem,
            )
        )

        while len(buffer) >= 4:
            predictions.append(buffer.pop(0).result())

    while len(buffer) > 0:
        predictions.append(buffer.pop(0).result())

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
            name="Run cellpose inference 2D/default",
            parameters={
                "image_dicts": image_dicts[i : min(i + chunk_size, len(images))],
                "cellpose_parameter": cellpose_parameter,
                "output_format": output_format,
            },
            client=get_client(),
        )
        predictions.extend(run.state.result())

    return predictions


@flow(
    name="Cellpose inference 2D [tiff]",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    result_storage=LocalFileSystem.load("cellpose"),
)
def cellpose_2D_tiff(
    input_parameter: InputData = InputData(),
    output_format: OutputFormat = OutputFormat(),
    cellpose_parameter: Cellpose = Cellpose(),
    parallel_jobs: int = 1,
    n_imgs_per_job: int = 500,
) -> list[ImageTarget]:
    imgs = list_images(
        input_dir=input_parameter.input_dir,
        pattern=input_parameter.pattern,
        pixel_resolution_um=input_parameter.xy_pixelsize_um,
        axes=input_parameter.axes,
    )

    n_imgs = len(imgs)
    split = n_imgs // parallel_jobs
    start = 0

    runs = []
    for i in range(parallel_jobs - 1):
        runs.append(
            submit_flows.submit(
                images=imgs[start : start + split],
                cellpose_parameter=cellpose_parameter,
                output_format=output_format,
                chunk_size=n_imgs_per_job,
            )
        )
        start = start + split

    runs.append(
        submit_flows.submit(
            images=imgs[start:],
            cellpose_parameter=cellpose_parameter,
            output_format=output_format,
            chunk_size=n_imgs_per_job,
        )
    )

    segmentations = []
    for run in runs:
        segmentations.extend(run.result())

    return segmentations


if __name__ == "__main__":
    cellpose_2D_tiff()
