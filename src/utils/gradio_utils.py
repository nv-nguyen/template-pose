# credit: https://github.com/cvlab-columbia/zero123
import fire
import hydra
from omegaconf import DictConfig
import gradio as gr
from functools import partial
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
from lovely_numpy import lo


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    """
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    """
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array(
        [
            [
                np.cos(azimuth_rad) * np.cos(polar_rad),
                -np.sin(azimuth_rad),
                -np.cos(azimuth_rad) * np.sin(polar_rad),
            ],
            [
                np.sin(azimuth_rad) * np.cos(polar_rad),
                np.cos(azimuth_rad),
                -np.sin(azimuth_rad) * np.sin(polar_rad),
            ],
            [np.sin(polar_rad), 0.0, np.cos(polar_rad)],
        ]
    )
    # print('camera_R:', lo(camera_R).v)

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T


class CameraVisualizer:
    def __init__(self, gradio_plot, num_neighbors=1):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polars = np.zeros(num_neighbors)
        self._azimuths = np.zeros(num_neighbors)
        self._radius = np.zeros(num_neighbors)
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None
        self.num_neighbors = num_neighbors

    def polar_change(self, values):
        self._polars = values
        # return self.update_figure()

    def neighbors_change(self, values):
        self.num_neighbors = values
        self._polars = np.zeros(values)
        self._azimuths = np.zeros(values)
        self._radius = np.zeros(values)
        # return self.update_figure()

    def azimuth_change(self, values):
        self._azimuths = values
        # return self.update_figure()

    def radius_change(self, values):
        self._radius = values
        # return self.update_figure()

    def encode_image(self, raw_image):
        """
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        """
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype="uint8")).convert(
            "P", palette="WEB"
        )
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert(
            "P", palette="WEB", dither=None
        )
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, "rgb({}, {}, {})".format(*rgb)]
            for i, rgb in enumerate(idx_to_color)  #
        ]

        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(
                np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W
            )
            print("x:", lo(x))
            print("y:", lo(y))
            print("z:", lo(z))
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=self._8bit_image,
                    cmin=0,
                    cmax=255,
                    colorscale=self._image_colorscale,
                    showscale=False,
                    lighting_diffuse=1.0,
                    lighting_ambient=1.0,
                    lighting_fresnel=1.0,
                    lighting_roughness=1.0,
                    lighting_specular=0.3,
                )
            )

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            cones = {}
            for idx_neighbor in range(self.num_neighbors):
                color = "red"
                cones[f"Query view, Top {idx_neighbor+1}"] = {
                    "color": color,
                    "width": self.num_neighbors - idx_neighbor,
                }  # To do
                cones[f"Query view, Top {idx_neighbor+1}"][
                    "cone"
                ] = calc_cam_cone_pts_3d(
                    self._polars[idx_neighbor],
                    self._azimuths[idx_neighbor],
                    base_radius + self._radius[idx_neighbor] * zoom_scale,
                    fov_deg,
                )  # (5, 3).

            for legend in cones:
                clr = cones[legend]["color"]
                cone = cones[legend]["cone"]
                width = cones[legend]["width"]
                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x1, x2],
                            y=[y1, y2],
                            z=[z1, z2],
                            mode="lines",
                            line=dict(color=clr, width=width),
                            name=legend,
                            showlegend=(i == 0),
                        )
                    )

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[cone[0, 0]],
                            y=[cone[0, 1]],
                            z=[cone[0, 2] - 0.05],
                            showlegend=False,
                            mode="text",
                            text=legend,
                            textposition="bottom center",
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[cone[0, 0]],
                            y=[cone[0, 1]],
                            z=[cone[0, 2] + 0.05],
                            showlegend=False,
                            mode="text",
                            text=legend,
                            textposition="top center",
                        )
                    )

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                # height=400,
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=True,
                legend=dict(
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99,
                ),
                scene=dict(
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0),
                    ),
                    xaxis_title="",
                    yaxis_title="",
                    zaxis_title="",
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks="",
                    ),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks="",
                    ),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks="",
                    ),
                ),
            )

        self._fig = fig
        return fig
