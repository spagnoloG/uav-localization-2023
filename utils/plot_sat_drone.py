import matplotlib.pyplot as plt
import os


def main():

    res_directory = "./res"  # path to the res directory
    drone_image_path = os.path.join(res_directory, "validation_drone_59e881f6e2da029497840f7a81f809f0cf6224b3-0-0.png")
    sat_image_path = os.path.join(res_directory, "validation_sat_59e881f6e2da029497840f7a81f809f0cf6224b3-0-0.png")
    
    # Load images
    drone_image = plt.imread(drone_image_path)
    sat_image = plt.imread(sat_image_path)
    
    # Plot images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display drone image
    ax[0].imshow(drone_image)
    ax[0].axis('off')  # Hide axis
    ax[0].set_title("Slika brezpilotnega letalnika")
    
    # Display satellite image
    ax[1].imshow(sat_image)
    ax[1].axis('off')  # Hide axis
    ax[1].set_title("Pripadajoča satelitska slika")
    
    #plt.tight_layout()
    plt.savefig(os.path.join(res_directory, "sat_drone.png"), dpi=200)

if __name__ == "__main__":
    main()
