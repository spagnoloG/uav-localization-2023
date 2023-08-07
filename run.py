#!/usr/bin/env python3
from logger import logger
from joined_dataset import JoinedDataset
import torch
from torch.utils.data import DataLoader
from model import CrossViewLocalizationModel
import os
from tqdm import tqdm
import yaml
import argparse
from torchviz import make_dot
import torchvision.transforms as transforms
from criterion import HanningLoss, RDS
from map_utils import MapUtils
import numpy as np
import plotly.graph_objects as go
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import queue
import threading
import rasterio
import plotly.express as px
import pandas as pd


class DashApp:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.data = queue.Queue()
        self.pd_lat_lon = pd.DataFrame(columns=["lat", "lon", "type"])

        self.app.layout = html.Div(
            children=[
                html.H1(children="Live run"),
                html.Div(
                    [
                        dcc.Graph(id="live-heatmap"),
                    ],
                ),
                html.Div(
                    [
                        dcc.Graph(id="3d-heatmap"),
                    ],
                ),
                html.Div(
                    [
                        dcc.Graph(id="live-map"),
                    ],
                ),
                dcc.Interval(
                    id="graph-update",
                    interval=1 * 3000,
                    n_intervals=0,
                ),
            ]
        )

        @self.app.callback(
            Output("3d-heatmap", "figure"),
            Output("live-heatmap", "figure"),
            Output("live-map", "figure"),
            Input("graph-update", "n_intervals"),
        )
        def update_3d_heatmap(n):
            if not self.data.empty():
                data = self.data.get()
                heatmap = data["heatmap"]
                xGrid, yGrid = np.meshgrid(data["x_3d_hm"], data["y_3d_hm"])

                ## 3D heatmap ##
                hm_3d_fig = go.Figure(
                    data=[go.Surface(z=heatmap, x=xGrid, y=yGrid, colorscale="Viridis")]
                )
                hm_3d_fig.update_layout(
                    title=f"Predicted Heatmap",
                    autosize=False,
                    scene=dict(
                        xaxis_title="X Axis",
                        yaxis_title="Y Axis",
                        zaxis_title="Heatmap Intensity",
                        aspectratio=dict(x=1, y=1, z=0.7),
                    ),
                    width=500,
                    height=500,
                    margin=dict(l=65, r=50, b=65, t=90),
                )
                ##  3D heatmap ##

                ## 2D heatmap ##
                hm_2d_fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=heatmap,
                            colorscale="Viridis",
                        )
                    ]
                )

                hm_2d_fig.update_layout(
                    title=f"Predicted Heatmap",
                    autosize=False,
                    scene=dict(
                        xaxis_title="X Axis",
                        yaxis_title="Y Axis",
                        zaxis_title="Heatmap Intensity",
                        aspectratio=dict(x=1, y=1, z=0.7),
                    ),
                    width=400,
                    height=400,
                )

                lat_pred = data["lat_pred"]
                lon_pred = data["lon_pred"]
                lat_gt = data["lat_gt"]
                lon_gt = data["lon_gt"]

                # map_fig = px.line_mapbox(
                #    self.pd_lat_lon,
                #    lat="lat",
                #    lon="lon",
                #    color="type",
                #    zoom=10,
                #    height=400,
                # )

                # map_fig.update_layout(
                #    mapbox_style="open-street-map",
                #    mapbox_zoom=16,
                #    mapbox_center_lat=lat_pred,
                #    mapbox_center_lon=lon_pred,
                #    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                # )
                map_fig = go.Figure(
                    data=[
                        go.Scattermapbox(
                            lat=[lat_pred],
                            lon=[lon_pred],
                            mode="markers",
                            marker=go.scattermapbox.Marker(size=14),
                            text=["Predicted"],
                        ),
                        go.Scattermapbox(
                            lat=[lat_gt],
                            lon=[lon_gt],
                            mode="markers",
                            marker=go.scattermapbox.Marker(size=14),
                            text=["Ground Truth"],
                        ),
                    ]
                )

                map_fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_zoom=16,
                    mapbox_center_lat=lat_pred,
                    mapbox_center_lon=lon_pred,
                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                )

                ##  2D heatmap ##
                return [hm_3d_fig, hm_2d_fig, map_fig]
            else:
                return [go.Figure(), go.Figure(), go.Figure()]

    def plot_data(self, heatmap, metadata):
        shape_y, shape_x = heatmap.shape

        x_3d_hm = np.linspace(0, shape_x - 1, shape_x)
        y_3d_hm = np.linspace(0, shape_y - 1, shape_y)

        lat_gt = metadata["lat_gt"]
        lon_gt = metadata["lon_gt"]
        lat_pred = metadata["lat_pred"]
        lon_pred = metadata["lon_pred"]

        self.pd_lat_lon = pd.concat(
            [
                self.pd_lat_lon,
                pd.DataFrame({"lat": [lat_pred], "lon": [lon_pred], "type": ["pred"]}),
            ],
            ignore_index=True,
        )

        self.pd_lat_lon = pd.concat(
            [
                self.pd_lat_lon,
                pd.DataFrame({"lat": [lat_gt], "lon": [lon_gt], "type": ["gt"]}),
            ],
            ignore_index=True,
        )

        self.data.put(
            {
                "heatmap": heatmap,
                "x_3d_hm": x_3d_hm,
                "y_3d_hm": y_3d_hm,
                "lat_gt": lat_gt,
                "lon_gt": lon_gt,
                "lat_pred": lat_pred,
                "lon_pred": lon_pred,
            }
        )

    def run(self, debug=False):
        self.app.run_server(debug=debug)


class Runner:
    def __init__(self, config=None):
        self.config = config
        self.device = self.config["run"]["device"]
        self.num_workers = self.config["run"]["num_workers"]
        self.val_hash = self.config["run"]["checkpoint_hash"]
        self.val_subset_size = self.config["run"]["run_subset_size"]
        self.batch_size = config["run"]["batch_size"]
        self.heatmap_kernel_size = config["dataset"]["heatmap_kernel_size"]
        self.RDS = RDS()
        self.dash_app = DashApp()

        self.criterion = HanningLoss(
            kernel_size=self.heatmap_kernel_size, device=self.device
        )

        self.prepare_dataloaders(config)
        self.map_utils = MapUtils()

        self.load_model()

    def prepare_dataloaders(self, config):
        if self.val_subset_size is not None:
            logger.info(f"Using val subset of size {self.val_subset_size}")
            subset_dataset = torch.utils.data.Subset(
                JoinedDataset(
                    dataset="test",
                    config=config,
                    tiffs=[16],
                ),
                indices=range(self.val_subset_size),
            )
            self.val_dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
        else:
            logger.info("Using full val dataset")
            subset_dataset = JoinedDataset(
                dataset="test",
                config=config,
                tiffs=[16],
            )
            self.val_dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

    def load_model(self):
        """
        Load the model for the validation phase.

        This function will load the model state dict from a saved checkpoint.
        The epoch used for loading the model is stored in the config file.
        """

        epoch = self.config["run"]["checkpoint_epoch"]

        # construct the path of the saved checkpoint
        load_path = f"./checkpoints/{self.val_hash}/checkpoint-{epoch}.pt"

        if not os.path.isfile(load_path):
            logger.error(f"No checkpoint found at '{load_path}'")
            raise FileNotFoundError(f"No checkpoint found at '{load_path}'")

        checkpoint = torch.load(load_path)

        self.model = torch.nn.DataParallel(
            CrossViewLocalizationModel(
                satellite_resolution=(
                    self.config["dataset"]["sat_patch_w"],
                    self.config["dataset"]["sat_patch_h"],
                )
            )
        )
        # load the state dict into the model
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # move the model to the correct device
        self.model.to(self.device)

        logger.info(f"Model loaded from '{load_path}' for validation.")

    def run_simulation(self):
        """
        Run the validation phase.

        This function will run the validation phase for the specified number of
        epochs. The validation loss will be printed after each epoch.
        """
        logger.info("Starting validation...")

        self.simulate()

        logger.info("Validation done.")

    def simulate(self):
        """
        Perform one epoch of validation.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        running_RDS = 0.0
        with torch.no_grad():
            for i, (
                drone_images,
                drone_infos,
                sat_images,
                heatmaps_gt,
            ) in tqdm(
                enumerate(self.val_dataloader),
                total=len(self.val_dataloader),
            ):
                drone_images = drone_images.to(self.device)
                sat_images = sat_images.to(self.device)
                heatmap_gt = heatmaps_gt.to(self.device)
                # Forward pass
                outputs = self.model(drone_images, sat_images)
                # Calculate loss
                loss = self.criterion(outputs, heatmap_gt)
                # Accumulate the loss
                running_loss += loss.item() * drone_images.size(0)

                x_sat = drone_infos["x_sat"]
                y_sat = drone_infos["y_sat"]

                ### RDS ###
                running_RDS += self.RDS(
                    outputs,
                    x_sat,
                    y_sat,
                    heatmaps_gt[0].shape[-1],
                    heatmaps_gt[0].shape[-2],
                )

                ### RDS ###
                for j in range(len(outputs)):
                    metadata = {
                        "x_sat": drone_infos["x_sat"][j].item(),
                        "y_sat": drone_infos["y_sat"][j].item(),
                        "x_offset": drone_infos["x_offset"][j].item(),
                        "y_offset": drone_infos["y_offset"][j].item(),
                        "zoom_level": drone_infos["zoom_level"][j].item(),
                        "lat_gt": drone_infos["coordinate"]["latitude"][j].item(),
                        "lon_gt": drone_infos["coordinate"]["longitude"][j].item(),
                        "filename": drone_infos["filename"][j],
                        "scale": drone_infos["scale"][j].item(),
                    }

                    self.plot_results(
                        drone_images[j].detach(),
                        sat_images[j].detach(),
                        heatmap_gt[j].detach(),
                        outputs[j].detach(),
                        metadata,
                        i,
                        j,
                    )

            total_samples += len(self.val_dataloader)

        epoch_loss = running_loss / total_samples

        self.val_loss = epoch_loss

        logger.info(f"Validation loss: {epoch_loss}")
        logger.info(f"Validation RDS: {running_RDS.cpu().item() / total_samples}")

    def visualize_model(self):
        tensor_uav = torch.randn(1, 128, 128, 3)
        tensor_sat = torch.randn(1, 512, 512, 3)

        fused_heatmap = self.model(tensor_uav, tensor_sat)

        dot = make_dot(fused_heatmap, params=dict(self.model.named_parameters()))
        dot.format = "png"
        os.makedirs("./vis", exist_ok=True)
        dot.render("model", "./vis", view=True)

    def plot_results(
        self,
        drone_image,
        sat_image,
        heatmap_gt,
        heatmap_pred,
        metadata,
        i,
        j,
    ):
        """
        Plot the validation results.

        This function will plot the validation results for the specified number of epochs.
        """
        # Inverse transform for images
        inverse_transforms = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[
                        -m / s
                        for m, s in zip(
                            self.config["dataset"]["mean"],
                            self.config["dataset"]["std"],
                        )
                    ],
                    std=[1 / s for s in self.config["dataset"]["std"]],
                ),
                transforms.ToPILImage(),
            ]
        )

        # Compute prediction, ground truth positions, and the distance
        heatmap_pred_np = heatmap_pred.cpu().numpy()
        y_pred, x_pred = np.unravel_index(
            np.argmax(heatmap_pred_np), heatmap_pred_np.shape
        )

        sat_image_path = metadata["filename"]
        zoom_level = metadata["zoom_level"]
        x_offset = metadata["x_offset"]
        y_offset = metadata["y_offset"]

        with rasterio.open(f"{sat_image_path}_sat_{zoom_level}.tiff") as s_image:
            sat_transform = s_image.transform
            lon_pred, lat_pred = rasterio.transform.xy(
                sat_transform, y_pred + y_offset, x_pred + x_offset
            )

        metadata["lat_pred"] = lat_pred
        metadata["lon_pred"] = lon_pred

        metadata["rds"] = self.map_utils.RDS(
            10,
            np.abs(metadata["x_sat"] - x_pred),
            np.abs(metadata["y_sat"] - y_pred),
            heatmap_gt.shape[-1],
            heatmap_gt.shape[-2],
        )

        self.dash_app.plot_data(heatmap_pred_np, metadata)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run the validation phase for the cross-view model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configuration",
        help="The path to the configuration file.",
    )

    args = parser.parse_args()

    config = load_config(f"./conf/{args.config}.yaml")

    runner = Runner(config=config)
    dash_thread = threading.Thread(target=runner.dash_app.run, kwargs={"debug": False})
    dash_thread.start()
    runner.run_simulation()
    dash_thread.join()


if __name__ == "__main__":
    main()
