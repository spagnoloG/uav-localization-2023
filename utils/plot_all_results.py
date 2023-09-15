import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import os
import re

# Directory where your images are stored
IMG_DIR = "../vis/12af867d77781eb93fec4dd41a9e43ff537ff5be/"

def load_images_from_directory():
    overlay_regex = re.compile(r"sat_overlay_([a-f0-9]+)-(\d+)-(\d+)\.png")
    drone_regex = re.compile(r"drone_image_([a-f0-9]+)-(\d+)-(\d+)\.png")
    satellite_regex = re.compile(r"satellite_image_([a-f0-9]+)-(\d+)-(\d+)\.png")

    images_dict = {}
    
    for file in os.listdir(IMG_DIR):
        overlay_match = overlay_regex.match(file)
        drone_match = drone_regex.match(file)
        satellite_match = satellite_regex.match(file)
        
        if overlay_match:
            hash_id, x, y = overlay_match.groups()
            category = f"{hash_id}_{x}_{y}"
            if category not in images_dict:
                images_dict[category] = {}
            images_dict[category]['overlay'] = os.path.join(IMG_DIR, file)
        elif drone_match:
            hash_id, x, y = drone_match.groups()
            category = f"{hash_id}_{x}_{y}"
            if category not in images_dict:
                images_dict[category] = {}
            images_dict[category]['drone'] = os.path.join(IMG_DIR, file)
        elif satellite_match:
            hash_id, x, y = satellite_match.groups()
            category = f"{hash_id}_{x}_{y}"
            if category not in images_dict:
                images_dict[category] = {}
            images_dict[category]['satellite'] = os.path.join(IMG_DIR, file)

    return images_dict

def main():

    def plot_quadruple(
        drone_path1,
        sat_path1,
        overlay_path1,
        drone_path2,
        sat_path2,
        overlay_path2,
        count,
    ):
        # Create the figure
        fig = plt.figure(figsize=(30, 8))

        gs = gridspec.GridSpec(1, 7, width_ratios=[1, 2, 2, 0.5, 1, 2, 2])

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(mpimg.imread(drone_path1))
        ax1.set_title("Drone Image", fontsize=16)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(mpimg.imread(sat_path1))
        ax2.set_title("Satellite Image", fontsize=16)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(mpimg.imread(overlay_path1))
        ax3.set_title("Satellite Overlay", fontsize=16)
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[4])
        ax4.imshow(mpimg.imread(drone_path2))
        ax4.set_title("Drone Image", fontsize=16)
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[5])
        ax5.imshow(mpimg.imread(sat_path2))
        ax5.set_title("Satellite Image", fontsize=16)
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[6])
        ax6.imshow(mpimg.imread(overlay_path2))
        ax6.set_title("Satellite Overlay", fontsize=16)
        ax6.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"./res_s/drone_net_example_{count}.png",
            dpi=100,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

    count = 0
    images_dict = load_images_from_directory()
    matched_paths = []
    
    for category, paths in images_dict.items():
        if 'drone' in paths and 'satellite' in paths and 'overlay' in paths:
            matched_paths.append((paths['drone'], paths['satellite'], paths['overlay']))
            
            if len(matched_paths) == 2:
                drone_path1, sat_path1, overlay_path1 = matched_paths[0]
                drone_path2, sat_path2, overlay_path2 = matched_paths[1]
                
                print('drone1', drone_path1)
                print('satellite1', sat_path1)
                print('overlay1', overlay_path1)
                print('drone2', drone_path2)
                print('satellite2', sat_path2)
                print('overlay2', overlay_path2)
    
                plot_quadruple(
                    drone_path1,
                    sat_path1,
                    overlay_path1,
                    drone_path2,
                    sat_path2,
                    overlay_path2,
                    count
                )
    
                count += 1
                matched_paths.clear() 

if __name__ == "__main__":
    main()
