import numpy as np

import matplotlib.pyplot as plt

def visualize_lines(mask, axis_points_list):
   """
   Visualize the anatomical axes overlaid on the femur mask.
   
   Parameters:
      mask (np.ndarray): Binary mask of the femur.
      axis_points_list (list): List of tuples, each containing start and end points of an anatomical axis as (start, end).
   """
   # Define a list of colors for different lines
   colors = plt.cm.get_cmap('hsv', len(axis_points_list) + 1)
   
   # Plot the mask
   plt.figure(figsize=(8, 8))
   plt.imshow(mask, cmap='gray')
   
   for i, axis_points in enumerate(axis_points_list):
      # Unpack the axis points
      start, end = axis_points
      
      # Plot the line
      plt.plot([start[1], end[1]], [start[0], end[0]], color=colors(i), linewidth=2, label=f"Axis {i+1}")
      
      # Plot the axis points
      plt.scatter([start[1], end[1]], [start[0], end[0]], color=colors(i), zorder=5)
   
   # Add labels and a legend
   plt.title('Anatomical Axes Visualization')
   plt.legend()
   plt.axis('off')  # Hide axes for better visualization
   plt.show()


def visualize_lines_v2(mask1, mask2, axis_points_list1, axis_points_list2, output_file="Sample1.png"):
   # Create a figure with two subplots
   fig, axes = plt.subplots(1, 2, figsize=(16, 8))

   # Plot the left mask with its axes
   axes[0].imshow(mask1, cmap='gray')
   axes[0].set_title('Left Leg Axes')
   axes[0].axis('off')
   for i, axis_points in enumerate(axis_points_list1):
      start, end = axis_points
      axes[0].plot([start[1], end[1]], [start[0], end[0]], linewidth=2, label=f"Axis {i+1}")
      axes[0].scatter([start[1], end[1]], [start[0], end[0]], zorder=5)
   axes[0].legend()

   # Plot the right mask with its axes
   axes[1].imshow(mask2, cmap='gray')
   axes[1].set_title('Right Leg Axes')
   axes[1].axis('off')
   for i, axis_points in enumerate(axis_points_list2):
      start, end = axis_points
      axes[1].plot([start[1], end[1]], [start[0], end[0]], linewidth=2, label=f"Axis {i+1}")
      axes[1].scatter([start[1], end[1]], [start[0], end[0]], zorder=5)
   axes[1].legend()

   # Save the plot as a PNG file
   plt.savefig(output_file, format='png')
   plt.close()