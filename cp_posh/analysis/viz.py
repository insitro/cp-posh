from typing import Dict
from matplotlib.colors import Colormap
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def create_composite_image(
    image: np.ndarray,
    cmap: str = "gist_rainbow",
    channel_map: Dict[int, int] = {0: 5, 1: 2, 2: 0, 3: 1, 4: 4, 5: 3},
):
    """
    Create a composite image from a multi-channel image using a colormap

    Parameters
    ----------
    image : np.ndarray
        input multi-channel image of shape (H, W, C)
    cmap : str, optional
        color map, by default 'gist_rainbow'
    channel_map : Optional[Dict[int, int]], optional
        mapping of channels to colors in cmap, by default None

    Returns
    -------
    np.ndarray
        RGB composite image
    """

    # convert to 3 channel image
    image = image.astype(np.float32)

    # create colormap
    colormap: Colormap = plt.get_cmap(cmap, image.shape[2])

    composite_image = np.zeros((image.shape[0], image.shape[1], 3))
    for ch in range(image.shape[2]):
        composite_image[:, :, 0] += image[:, :, ch] * colormap(channel_map[ch])[0]
        composite_image[:, :, 1] += image[:, :, ch] * colormap(channel_map[ch])[1]
        composite_image[:, :, 2] += image[:, :, ch] * colormap(channel_map[ch])[2]

    return np.clip(composite_image, 0, 1)


def interactive_image_scatter_plot(
    img: np.ndarray,
    scatter_x: np.ndarray,
    scatter_y: np.ndarray,
    hover_text: np.ndarray,
):
    """
    plot an interactive scatter plot with hover text over an image

    Parameters
    ----------
    img : np.ndarray
        image to plot
    scatter_x : np.ndarray
        x-coordinates of scatter points
    scatter_y : np.ndarray
        y-coordinates of scatter points
    hover_text : np.ndarray
        hover text for scatter points (in order as scatter_x, scatter_y)
    """
    fig = go.Figure(layout=go.Layout(width=img.shape[1], height=img.shape[0]))

    fig.add_trace(go.Image(z=img))
    fig.add_trace(
        go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            marker=dict(size=5, color="red"),
            text=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white"
    )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        autorange="reversed",  # This reverses the y-axis as images have origin at top-left
    )

    fig.show()
