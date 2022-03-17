import logging
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go

from dicom_image_tools.roi.square_roi import SquareRoi

from ..constants.plotting import (
    PLOTLY_FONT_COLOUR,
    PLOTLY_FONT_FAMILY,
    PLOTLY_HEATMAP_AXIS_LINE_COLOUR,
    PLOTLY_HEATMAP_HOVERLABEL_BACKGROUND_COLOUR,
    PLOTLY_HEATMAP_HOVERLABEL_BORDER_COLOUR,
    PLOTLY_PAPER_BACKGROUND_COLOUR,
    PLOTLY_PLOT_BACKGROUND_COLOUR,
    PLOTLY_TICK_COLOUR,
)
from .colour_scales import plotly_colour_scales

logger = logging.getLogger(__name__)


def _plotly_layout(x_scale: float, y_scale: float):
    axis_template = {
        "showgrid": False,
        "zeroline": False,
        "linecolor": PLOTLY_HEATMAP_AXIS_LINE_COLOUR,
        "showticklabels": False,
        "ticks": "",
    }

    x_axis = axis_template.copy()
    x_axis["scaleanchor"] = "y"
    x_axis["scaleratio"] = x_scale / y_scale

    y_axis = axis_template.copy()
    y_axis["scaleanchor"] = "x"
    y_axis["scaleratio"] = y_scale / x_scale
    y_axis["autorange"] = "reversed"

    return go.Layout(
        autosize=True,
        plot_bgcolor=PLOTLY_PLOT_BACKGROUND_COLOUR,
        paper_bgcolor=PLOTLY_PAPER_BACKGROUND_COLOUR,
        xaxis=x_axis,
        yaxis=y_axis,
    )


def _get_square_roi_border_trace(roi: SquareRoi, roi_colour: str, roi_border_width: float, roi_only_border: bool):
    return go.Scatter(
        x=[roi.UpperLeft.x, roi.UpperRight.x, roi.LowerRight.x, roi.LowerLeft.x, roi.UpperLeft.x],
        y=[roi.UpperLeft.y, roi.UpperRight.y, roi.LowerRight.y, roi.LowerLeft.y, roi.UpperLeft.y],
        mode="lines",
        line={"color": roi_colour, "width": roi_border_width, "shape": "linear"},
        showlegend=False,
        name="ROI",
        fill=None if roi_only_border else "toself",
        hoverinfo="skip",
    )


def _create_image_heatmap_trace(
    image: np.ndarray, colour_map: str, colour_bar: bool, window: tuple[float, float]
) -> go.Heatmap:
    return go.Heatmap(
        z=image,
        colorscale=plotly_colour_scales.get(colour_map),
        colorbar=None
        if not colour_bar
        else {
            "thickness": 20,
            "ticklen": 4,
            "tickfont": {"family": PLOTLY_FONT_FAMILY, "color": PLOTLY_FONT_COLOUR},
            "tickcolor": PLOTLY_TICK_COLOUR,
        },
        zmin=window[0],
        zmax=window[1],
        showlegend=False,
        hoverlabel={
            "bgcolor": PLOTLY_HEATMAP_HOVERLABEL_BACKGROUND_COLOUR,
            "bordercolor": PLOTLY_HEATMAP_HOVERLABEL_BORDER_COLOUR,
            "font": {"color": PLOTLY_FONT_COLOUR, "family": PLOTLY_FONT_FAMILY},
        },
    )


def get_image_and_roi_traces_and_layout(
    image: np.ndarray,
    x_scale: float,
    y_scale: float,
    colour_map: str = "bone",
    colour_bar: bool = True,
    rois: Optional[Union[SquareRoi, list[SquareRoi]]] = None,
    roi_colour: Optional[str] = "#33a652",
    roi_only_border: Optional[bool] = True,
    roi_border_width: Optional[int] = 2,
    window: Optional[tuple[float, float]] = None,
) -> tuple[list[Union[go.Heatmap, go.Scatter]], go.Layout]:
    """Creates a graph objects heatmap and scatter for visualizing DICOM image and related ROIs

    Args:
        image:
        x_scale:
        y_scale:
        colour_map:
        colour_bar:
        rois:
        roi_colour:
        roi_only_border:
        roi_border_width:
        window:

    Returns:

    """
    if window is None:
        window = (float(np.min(image)), float(np.max(image)))

    im_plot = [_create_image_heatmap_trace(image=image, colour_map=colour_map, colour_bar=colour_bar, window=window)]

    if rois:
        if not isinstance(rois, list) or not all([isinstance(roi, SquareRoi) for roi in rois]):
            raise TypeError("The ROIs must be given as a list of SquareRoi instances")

        im_plot += [
            _get_square_roi_border_trace(
                roi=roi, roi_colour=roi_colour, roi_border_width=roi_border_width, roi_only_border=roi_only_border
            )
            for roi in rois
        ]

    return im_plot, _plotly_layout(x_scale=x_scale, y_scale=y_scale)


def show_image(
    image: np.ndarray,
    x_scale: float,
    y_scale: float,
    colour_map: str = "bone",
    colour_bar: bool = True,
    rois: Optional[SquareRoi] = None,
    roi_colour: Optional[str] = "#33a652",
    roi_only_border: Optional[bool] = True,
    roi_border_width: Optional[int] = 2,
    window: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """Creates and shows a 2D image as an interactive image in the browser. The colour map, windowing

    Args:
        image:
        x_scale:
        y_scale:
        colour_map:
        colour_bar:
        rois:
        roi_colour:
        roi_only_border:
        roi_border_width:
        window:

    Returns:

    """
    im_plot, layout = get_image_and_roi_traces_and_layout(
        image=image,
        x_scale=x_scale,
        y_scale=y_scale,
        colour_map=colour_map,
        colour_bar=colour_bar,
        rois=rois,
        roi_colour=roi_colour,
        roi_only_border=roi_only_border,
        roi_border_width=roi_border_width,
        window=window,
    )

    fig = go.Figure(data=im_plot, layout=layout)
    fig.show()

    return fig


def _validate_scale_list(scale: list[Union[int, float]]) -> bool:
    return scale is None or not isinstance(scale, list) or any([not isinstance(el, (float, int)) for el in scale])


def create_stack_plot(
    image: np.ndarray,
    x_scale: list[float],
    y_scale: list[float],
    colour_map: str = "bone",
    colour_bar: bool = True,
    rois: Optional[list[list[SquareRoi]]] = None,
    roi_colour: Optional[str] = "#33a652",
    roi_only_border: Optional[bool] = True,
    roi_border_width: Optional[int] = 2,
    window: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """

    Args:
        image: THe 3D numpy ndarray containing the pixel/voxel values to plot/show
        x_scale: Voxel size in x-direction
        y_scale: Voxel size in y-direction
        colour_map: The colour map to use
        colour_bar:
        rois:
        roi_colour:
        roi_only_border:
        roi_border_width:
        window:

    Returns:

    """
    if rois is None:
        rois = [None] * image.shape[2]

    if _validate_scale_list(x_scale) or _validate_scale_list(y_scale):
        raise ValueError("The x- and y-scale must be specified as a list of floats")

    if (
        not isinstance(rois, list)
        or len(rois) != image.shape[2]
        or not all(
            [
                roi is None
                or (isinstance(roi, list) and all([isinstance(roi_instance, SquareRoi) for roi_instance in roi]))
                for roi in rois
            ]
        )
    ):
        raise TypeError(
            "ROIs, if specified, must be given as a list of same length as the image slices to display containing lists of SquareRoi instances"
        )

    heatmaps, layouts = map(
        list,
        zip(
            *[
                get_image_and_roi_traces_and_layout(
                    image=image[:, :, ind],
                    x_scale=x_scale[ind],
                    y_scale=y_scale[ind],
                    window=window,
                    rois=rois[ind],
                    roi_colour=roi_colour,
                    roi_only_border=roi_only_border,
                    roi_border_width=roi_border_width,
                    colour_map=colour_map,
                    colour_bar=colour_bar,
                )
                for ind in range(image.shape[2])
            ]
        ),
    )

    fig = go.Figure(
        data=heatmaps[0],
        frames=[go.Frame(data=heatmap) for heatmap in heatmaps],
        layout=layouts[0],
    )

    fig.show()

    return fig
