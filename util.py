import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, random_split


def show_img(img, boxes, labels, filename, class_names=None):
    """
    Display an image with bounding boxes drawn around detected traffic signs,
    and display the image filename on top.

    Args:
        img (Tensor): The image tensor (C, H, W), typically from DataLoader.
        boxes (Tensor): Bounding boxes of shape (N, 4) where N is the number of boxes.
                        Each box is (center_x, center_y, width, height) in normalized coordinates.
        labels (Tensor): Class labels for the boxes, shape (N,)
        filename (str): The filename of the image, to be displayed on top of the image.
        class_names (list of str, optional): List of class names to map label ids to.
                                             If not provided, just label ids will be used.
    """
    # Convert the image tensor back to a NumPy array for display
    img = img.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

    # Plot the image
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    # Display the filename on top of the image
    ax.text(0.5, 1.02, filename, color='red', fontsize=14, weight='bold', ha='center', transform=ax.transAxes)

    # Draw the bounding boxes
    for i in range(boxes.shape[0]):
        # Convert from normalized coordinates to pixel coordinates
        h, w, _ = img.shape
        center_x, center_y, width, height = boxes[i]
        x = (center_x - width / 2) * w
        y = (center_y - height / 2) * h
        box_width = width * w
        box_height = height * h

        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Annotate with class label
        label = labels[i].item()
        if class_names:
            label_name = class_names[label]
        else:
            label_name = str(label)  # Just show label id if class names are not provided

        ax.text(x, y - 5, label_name, color='r', fontsize=12, weight='bold')

    # Display the plot
    plt.axis('off')
    plt.show()


def split_dataset(dataset, train_ratio=0.8, batch_size=4, shuffle=False, collate_fn=None):
    """
    Split the dataset into training and testing sets based on the given ratio.

    Args:
        dataset (Dataset): The full dataset to be split.
        train_ratio (float): The ratio of the dataset to be used for training. The rest will be used for testing.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset before splitting.
        collate_fn (function, optional): A custom collate function to merge samples into a batch.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the testing set.
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # Split the dataset into train and test subsets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print("Random Split Finished.")
    # Create DataLoader for the training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    print("Train Set Split Finished.")
    # Create DataLoader for the testing set
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    print("Test Set Split Finished.")
    return train_loader, test_loader


def slidingWindow(image_size, init_size=(64, 64), x_overlap=0.5, y_step=0.05,
                  x_range=(0, 1), y_range=(0, 1), scale=1.5):
    """
    Run a sliding window across an input image and return a list of the
    coordinates of each window.

    Window travels the width of the image (in the +x direction) at a range of
    heights (toward the bottom of the image in the +y direction). At each
    successive y, the size of the window is increased by a factor equal to
    @param scale. The horizontal search area is limited by @param x_range
    and the vertical search area by @param y_range.

    @param image_size (int, int): Size of the image (width, height) in pixels.
    @param init_size (int, int): Initial size of of the window (width, height)
        in pixels at the initial y, given by @param y_range[0].
    @param x_overlap (float): Overlap between adjacent windows at a given y
        as a float in the interval [0, 1), where 0 represents no overlap
        and 1 represents 100% overlap.
    @param y_step (float): Distance between successive heights y as a
        fraction between (0, 1) of the total height of the image.
    @param x_range (float, float): (min, max) bounds of the horizontal search
        area as a fraction of the total width of the image.
    @param y_range (float, float) (min, max) bounds of the vertical search
        area as a fraction of the total height of the image.
    @param scale (float): Factor by which to scale up window size at each y.
    @return windows: List of tuples, where each tuple represents the
        coordinates of a window in the following order: (upper left corner
        x coord, upper left corner y coord, lower right corner x coord,
        lower right corner y coord).
    """

    windows = []
    h, w = image_size[1], image_size[0]
    for y in range(int(y_range[0] * h), int(y_range[1] * h), int(y_step * h)):
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * h))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        if y + win_height > int(y_range[1] * h) or win_width > w:
            break
        x_step = int((1 - x_overlap) * win_width)
        for x in range(int(x_range[0] * w), int(x_range[1] * w), x_step):
            windows.append((x, y, x + win_width, y + win_height))

    return windows
