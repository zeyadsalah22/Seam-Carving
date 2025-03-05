import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def convolve(image, kernel):
  height, width = image.shape
  kernel_size = kernel.shape[0]

  pad = kernel_size // 2
  padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

  output = np.zeros((height, width), dtype=image.dtype)

  for i in range(height):
    for j in range(width):
      region = padded_image[i:i+kernel_size, j:j+kernel_size]
      output[i][j] = np.sum(region * kernel)
  return output


def get_energy(image):
  image_gray = ski.color.rgb2gray(image)

  sx = np.array([[-0.125, 0, 0.125],[-0.25, 0, 0.25],[-0.125, 0, 0.125]])
  sy = np.array([[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]])

  x_edges = convolve(image_gray, sx)
  y_edges = convolve(image_gray, sy)

  x_edges = np.abs(x_edges)
  y_edges = np.abs(y_edges)
  energy = x_edges + y_edges
  
  return energy

def greedy_approach(energy, horizontal=True):
    if not horizontal:
        energy = energy.T  # Transpose for vertical seam processing

    n, m = energy.shape
    path = np.zeros(n, dtype=np.int32)
    path[0] = np.argmin(energy[0])  # Get index of min energy in first row

    for i in range(1, n):
        prev = path[i - 1]
        path[i] = prev

        # Check left and right (or top and bottom when transposed)
        if prev < m - 1 and energy[i, prev + 1] < energy[i, prev]:
            path[i] = prev + 1
        if prev > 0 and energy[i, prev - 1] < energy[i, path[i]]:
            path[i] = prev - 1

    return path


def dp_approach(energy, horizontal=True):
    
    if not horizontal:
        energy = energy.T # Transpose for vertical seam processing
    
    n, m = energy.shape
    dp = np.zeros((n, m), dtype=np.float32)
    dp[-1, :] = energy[-1, :]  # Initialize last row with energy values

    for i in range(n - 2, -1, -1):  # Iterate from second last row to top
        for j in range(m):
            dp[i, j] = dp[i + 1, j] + energy[i, j]
            if j > 0:
                dp[i, j] = min(dp[i, j], dp[i + 1, j - 1] + energy[i, j])
            if j < m - 1:
                dp[i, j] = min(dp[i, j], dp[i + 1, j + 1] + energy[i, j])
    return dp if horizontal else dp.T

def mark_seam(image, seam, color=(148, 0, 211), horizontal=True):  
    marked_image = image.copy()

    if horizontal:
        for i in range(len(seam)):
            marked_image[i, seam[i]] = color  # Mark vertical seam
    else:
        for j in range(len(seam)):
            marked_image[seam[j], j] = color  # Mark horizontal seam

    return marked_image

def remove_seam(image, seam, horizontal=True):
    height, width, channels = image.shape
    if horizontal:
        new_image = np.zeros((height, width - 1, channels), dtype=image.dtype)

        for i in range(height):
            j = seam[i]
            new_image[i, :, :] = np.concatenate((image[i, :j, :], image[i, j+1:, :]), axis=0)

    else:
        new_image = np.zeros((height - 1, width, channels), dtype=image.dtype)

        for j in range(width):
            i = seam[j]
            new_image[:, j, :] = np.concatenate((image[:i, j, :], image[i+1:, j, :]), axis=0)

    return new_image


def get_seam(image, horizontal = True):
  energy = get_energy(image)
  return greedy_approach(dp_approach(energy, horizontal), horizontal)


# Load the image
image = ski.io.imread("boy.png")

# Number of seams to remove
num_seams = 100

# Create a figure
fig, ax = plt.subplots()

# Store images for animation
images = [image]
horizontal = False

# Seam carving loop
for _ in range(num_seams):
    seam = get_seam(images[-1], horizontal)  # Get the seam
    marked_image = mark_seam(images[-1], seam, horizontal= horizontal)
    images.append(marked_image)  # Store marked image
    new_image = remove_seam(images[-1], seam, horizontal = horizontal)
    images.append(new_image)  # Store the new image

# Function to update frames
def update(frame):
    ax.clear()
    ax.imshow(images[frame])
    ax.set_xticks([])  # Hide X-axis
    ax.set_yticks([])  # Hide Y-axis
    ax.set_title(f"Seam {frame // 2} Removed")  # Update title

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(images), interval=300)

ani.save("boy_animation_ver.gif", writer="pillow", fps=10)  # Save as GIF
ski.io.imsave("scaled_boy_100_ver.png", images[-1])  # Save the final resized image

plt.show()


