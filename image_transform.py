# ==========================================================================
# Eses Image Transform
# ==========================================================================
#
# Description:
# The 'Eses Image Transform' node provides tools for applying 2D transformations
# to an image and an optional mask. It allows for adjustments to the position,
# scale, and rotation of the input content. This version relies on the
# Pillow (PIL) library for all image processing, performing operations on the CPU.
#
# Key Features:
#
# - Core Transformations:
#   - Rotation: Rotates the image around its center.
#   - Zoom: Scales the image in or out from the center.
#   - Local Scale: Squashes or stretches the image along its local X or Y axis.
#   - Offset: Shifts the image horizontally and vertically.
#   - Flip: Flips the image on its X or Y axis.
#
# - Tiling and Fill:
#   - Tiling: Fills the canvas with repeating copies of the source image when zoomed out.
#   - Fill Color: Sets the background color for areas exposed by transformations.
#     Supports RGB (e.g., 255,128,0) or RGBA (e.g., 0,255,0,128) values.
#
# - Masking Controls:
#   - Apply Mask to RGB Image: Uses the mask to define the transparency of the output image.
#   - Invert Mask Input: Inverts the incoming mask before transformations are applied.
#   - Invert Mask Output: Inverts the final generated mask before it is output.
#
# - Quality:
#   - Resampling Filter: Sets the algorithm (e.g., bicubic, nearest neighbor)
#     used for scaling and rotation operations.
#
# Usage:
# Connect an image and an optional mask. Adjust the transformation parameters
# as needed. The node outputs the transformed image and mask.
#
# Version: 1.1.0
# License: See LICENSE.txt
#
# ==========================================================================

import os
import torch
from PIL import Image, ImageOps
import numpy as np
import math

SMALL_STEP = 1
ZOOM_STEP = 0.001
NODE_PATH = os.path.dirname(__file__)


class EsesImageTransform:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flip_x": ("BOOLEAN", {"default": False}),
                "flip_y": ("BOOLEAN", {"default": False}),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -2048.0, "max": 2048.0, "step": SMALL_STEP}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -2048.0, "max": 2048.0, "step": SMALL_STEP}),
                "zoom_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": ZOOM_STEP}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.01}),
                "local_scale_x": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "local_scale_y": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "tiling_mode": (["Off", "On"],),
                "max_tiling_grid_size": ("INT", {"default": 3, "min": 1, "max": 111, "step": 2}),
                "tiling_pixel_overlap": ("INT", {"default": 4, "min": 0, "max": 64, "step": 2}),
                "apply_mask_to_rgb_image": ("BOOLEAN", {"default": False}),
                "invert_mask_input": ("BOOLEAN", {"default": False}),
                "invert_mask_output": ("BOOLEAN", {"default": False}),
                "resample_filter": (["bicubic", "bilinear", "nearest"],),
                "fill_color": ("STRING", {"default": "0,0,0"}),
            },
            "optional": { "image": ("IMAGE",), "mask": ("MASK",), }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INFO")
    RETURN_NAMES = ("IMAGE", "MASK", "info")
    FUNCTION = "apply_transformations"
    CATEGORY = "Eses Nodes/Image"
    
    def _round_to_nearest_odd(self, num):
        if num == 0:
            return 1
        rounded = round(num)
        if rounded % 2 == 0:
            if abs(num - (rounded - 1)) < abs(num - (rounded + 1)): return max(1, rounded - 1)
            else: return rounded + 1
        return rounded

    def _create_tiled_pil(self, source_pil, canvas_width, canvas_height, tile_source_width, tile_source_height, zoom_factor, max_tiling_grid_size, tiling_pixel_overlap, resampling_filter, fill_color_or_value):
        pixel_overlap = tiling_pixel_overlap
        step_x_float = tile_source_width * zoom_factor
        step_y_float = tile_source_height * zoom_factor
        
        if step_x_float <= 0 or step_y_float <= 0: 
            return Image.new(source_pil.mode, (canvas_width, canvas_height), fill_color_or_value)
        
        padded_width = max(1, int(round(step_x_float + pixel_overlap)))
        padded_height = max(1, int(round(step_y_float + pixel_overlap)))
        num_tiles_x = min(max_tiling_grid_size, math.ceil(canvas_width / step_x_float) + 2 | 1)
        num_tiles_y = min(max_tiling_grid_size, math.ceil(canvas_height / step_y_float) + 2 | 1)
        total_grid_w = num_tiles_x * step_x_float
        total_grid_h = num_tiles_y * step_y_float
        start_x, start_y = (canvas_width - total_grid_w) / 2.0, (canvas_height - total_grid_h) / 2.0
        scaled_padded_source = source_pil.resize((padded_width, padded_height), resampling_filter)
        temp_img = Image.new(source_pil.mode, (canvas_width, canvas_height), fill_color_or_value)
        
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                paste_x = int(round(start_x + x * step_x_float - pixel_overlap / 2.0))
                paste_y = int(round(start_y + y * step_y_float - pixel_overlap / 2.0))
                paste_mask = scaled_padded_source if scaled_padded_source.mode in ['RGBA', 'LA'] else None
                temp_img.paste(scaled_padded_source, (paste_x, paste_y), paste_mask)
        
        return temp_img

    def _parse_color_string(self, color_string):
        color_string = color_string.strip()
        if color_string.startswith("#"):
            color_string = color_string[1:]
            if len(color_string) == 6: return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4)) + (255,)
            elif len(color_string) == 8: return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4, 6))
            else: return (0, 0, 0, 0)
        else:
            try:
                parts = [int(p.strip()) for p in color_string.split(",")]
                if len(parts) == 3: return tuple(parts) + (255,)
                elif len(parts) == 4: return tuple(parts)
                else: return (0, 0, 0, 0)
            except ValueError: return (0, 0, 0, 0)

    def _get_perspective_coeffs(self, width, height, rotation_angle, zoom_factor, offset_x, offset_y, local_scale_x, local_scale_y):
        center_x, center_y = width / 2, height / 2
        angle_rad = np.deg2rad(rotation_angle)
        
        T1 = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]], dtype=np.float32)
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0], [np.sin(angle_rad), np.cos(angle_rad), 0], [0, 0, 1]], dtype=np.float32)
        
        # The non-uniform scaling matrix combines zoom and local scale. This is applied BEFORE rotation.
        S = np.array([[zoom_factor * local_scale_x, 0, 0], 
                      [0, zoom_factor * local_scale_y, 0], 
                      [0, 0, 1]], dtype=np.float32)

        T2_restore_center = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]], dtype=np.float32)
        M_obj_src_to_dest = T2_restore_center @ R @ S @ T1
        M_obj_src_to_dest_inv = np.linalg.inv(M_obj_src_to_dest)
        T_world_offset_inv = np.array([[1, 0, -offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
        transform_matrix = M_obj_src_to_dest_inv @ T_world_offset_inv
        
        if transform_matrix[2, 2] != 0: transform_matrix /= transform_matrix[2, 2]
        
        a, b, c = transform_matrix[0, 0], transform_matrix[0, 1], transform_matrix[0, 2]
        d, e, f = transform_matrix[1, 0], transform_matrix[1, 1], transform_matrix[1, 2]
        g, h = transform_matrix[2, 0], transform_matrix[2, 1]
        coeffs = (a, b, c, d, e, f, g, h)
        
        return coeffs


    def apply_transformations(self, rotation_angle, zoom_factor, offset_x, offset_y,
                            local_scale_x, local_scale_y,
                            tiling_mode, max_tiling_grid_size, tiling_pixel_overlap,
                            flip_x, flip_y, resample_filter, fill_color,
                            invert_mask_input, apply_mask_to_rgb_image, invert_mask_output,
                            image=None, mask=None):

        # -- Step 1: Initialize variables and load inputs into PIL Images --
        
        # Start with empty placeholder variables.
        img_pil, mask_pil, current_width, current_height = None, None, 0, 0
        
        # Check if an image tensor was connected to the node's input.
        if image is not None:
            # Convert the torch tensor to a NumPy array, scale it from 0-1 to 0-255, and create a PIL Image.
            img_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))
            # Get the width and height from this image.
            current_width, current_height = img_pil.size
        
        # Check if a mask tensor was connected.
        if mask is not None:
            # Squeeze the tensor to remove unnecessary dimensions (e.g., a 1xHxWx1 mask becomes HxW).
            mask_squeezed = mask.squeeze()
            # If the mask has 3 dimensions (HxWxChannels), it's likely an RGB image used as a mask.
            if mask_squeezed.ndim == 3:
                # We convert it to a single-channel grayscale mask by averaging the color channels.
                mask_squeezed = mask_squeezed.mean(dim=-1)
            # Convert the mask tensor to a single-channel grayscale ('L') PIL Image.
            mask_pil = Image.fromarray(np.clip(255. * mask_squeezed.cpu().numpy(), 0, 255).astype(np.uint8), 'L')
            # If no image was provided, use the mask's dimensions as the main dimensions.
            if not img_pil:
                current_width, current_height = mask_pil.size
        
        # If an image was provided but no mask was, we need a mask to work with.
        if img_pil and not mask_pil:
            # Create a new, default mask that is completely white (value 255), meaning it's fully opaque.
            mask_pil = Image.new('L', (current_width, current_height), 255)
        
        # Check if the user wants to invert the input mask.
        if mask_pil and invert_mask_input:
            mask_pil = ImageOps.invert(mask_pil)

        # If after all that, we have no dimensions, we can't proceed.
        if current_width == 0 or current_height == 0:
            # Return empty/zeroed-out tensors.
            return (torch.zeros(1, 64, 64, 4), torch.zeros(1, 64, 64), 0.0, 1.0, "No valid input.")


        # -- Step 2: Prepare transformation parameters --

        # Store the original dimensions, needed at the end for cropping and final output.
        original_width, original_height = current_width, current_height
        
        # Make sure the tiling grid size is an odd number (1, 3, 5...) for symmetrical tiling.
        max_tiling_grid_size = self._round_to_nearest_odd(max_tiling_grid_size)
        
        # A dictionary to map the user-friendly string from the UI to the actual Pillow library constant.
        resampling_method_map = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR, "nearest": Image.Resampling.NEAREST}
        # Get the chosen resampling method, defaulting to BICUBIC if something goes wrong.
        resampling_method = resampling_method_map.get(resample_filter, Image.Resampling.BICUBIC)

        # Apply flips first, as they are simple operations before the complex perspective transform.
        if flip_x:
            if img_pil: img_pil = ImageOps.mirror(img_pil)
            if mask_pil: mask_pil = ImageOps.mirror(mask_pil)
        if flip_y:
            if img_pil: img_pil = ImageOps.flip(img_pil)
            if mask_pil: mask_pil = ImageOps.flip(mask_pil)

        # Convert the user's color string (e.g., "0,255,0,128") into a usable RGBA integer tuple.
        fill_rgba = self._parse_color_string(fill_color)
        

        # -- Step 3: Main Transformation Logic --
        
        # Determine if we need to run the tiling logic.
        is_tiling_active = tiling_mode == "On" and zoom_factor != 1.0

        # Handle the "oversize-and-crop" method for tiling to prevent cutoff artifacts.
        if is_tiling_active:
            # Create a temporary canvas that is much larger than the original image.
            SUPER_SIZE_FACTOR = 3
            large_width, large_height = original_width * SUPER_SIZE_FACTOR, original_height * SUPER_SIZE_FACTOR
            
            # Group all parameters for the tiling function into a dictionary for a cleaner call.
            tiling_common_args = {"canvas_width": large_width, "canvas_height": large_height, "tile_source_width": original_width, "tile_source_height": original_height, "zoom_factor": zoom_factor, "max_tiling_grid_size": max_tiling_grid_size, "tiling_pixel_overlap": tiling_pixel_overlap}
            
            # If an image exists, replace it with the new, large tiled version.
            if img_pil:
                img_pil = self._create_tiled_pil(source_pil=img_pil.convert('RGBA'), resampling_filter=resampling_method, fill_color_or_value=(0,0,0,0), **tiling_common_args)
            # Do the same for the mask.
            if mask_pil:
                 mask_pil = self._create_tiled_pil(source_pil=mask_pil, resampling_filter=Image.Resampling.NEAREST, fill_color_or_value=0, **tiling_common_args)

            # Update our working dimensions to the large canvas size.
            current_width, current_height = large_width, large_height
            
            # Calculate the transformation matrix for the tiled plane.
            # Zoom is set to 1.0 because the tiling has already handled the scaling.
            # The local_scale is now correctly applied to the entire tiled result.
            coeffs = self._get_perspective_coeffs(current_width, current_height, rotation_angle, 1.0, offset_x, offset_y, local_scale_x, local_scale_y)
        else:
            # If not tiling, calculate the transformation matrix with all parameters.
            coeffs = self._get_perspective_coeffs(current_width, current_height, rotation_angle, zoom_factor, offset_x, offset_y, local_scale_x, local_scale_y)

        # Initialize placeholder variables for the transformed results.
        transformed_img, transformed_mask = None, None
        
        # Apply the calculated perspective transformation to the image.
        if img_pil:
            if img_pil.mode != 'RGBA': img_pil = img_pil.convert('RGBA')
            transformed_img = img_pil.transform((current_width, current_height), Image.PERSPECTIVE, coeffs, resample=resampling_method, fillcolor=(0,0,0,0))
        
        # Apply the same transformation to the mask.
        if mask_pil:
            transformed_mask = mask_pil.transform((current_width, current_height), Image.PERSPECTIVE, coeffs, resample=Image.Resampling.NEAREST, fillcolor=0)

        # If we used the tiling method, we now need to crop back to the original size.
        if is_tiling_active:
            # Define a cropping box in the center of the large canvas.
            crop_box = ((current_width - original_width) // 2, (current_height - original_height) // 2, (current_width + original_width) // 2, (current_height + original_height) // 2)
            # Crop both the image and the mask.
            if transformed_img: transformed_img = transformed_img.crop(crop_box)
            if transformed_mask: transformed_mask = transformed_mask.crop(crop_box)
        
        
        # --- Step 4: Final Compositing and Output Preparation ---
        
        # If we have a transformed image, prepare it for output.
        if transformed_img:
            # Start with the transformed image.
            image_for_compositing = transformed_img
            
            # Check if the user wants to apply the mask to the image's transparency.
            if apply_mask_to_rgb_image and transformed_mask:
                # Split the image into its R, G, B, and temporary Alpha channels.
                r, g, b, a = transformed_img.split()
                # Re-merge the R, G, B channels with our final processed mask as the new Alpha channel.
                image_for_compositing = Image.merge('RGBA', (r, g, b, transformed_mask))
            
            # Create the solid color background.
            background = Image.new('RGBA', (original_width, original_height), fill_rgba)
            # Composite the image (with its new alpha) onto the background.
            final_composited_image = Image.alpha_composite(background, image_for_compositing)
            
            # Convert the final PIL Image to a NumPy array, normalize values to 0-1, and then to a Torch tensor.
            output_image = np.array(final_composited_image.convert("RGBA")).astype(np.float32) / 255.0
            output_image = torch.from_numpy(output_image)[None,]
        else:
            # If there was no input image, create a blank transparent output.
            output_image = torch.zeros(1, original_height, original_width, 4)


        # If a transformed mask exists, prepare it for output.
        if transformed_mask:
            # Start with the final processed mask.
            output_mask_pil = transformed_mask
            
            # Check if the user wants to invert the final mask output.
            if invert_mask_output:
                output_mask_pil = ImageOps.invert(output_mask_pil)
            
            # Convert the final mask to a NumPy array and then a Torch tensor.
            output_mask_np = np.array(output_mask_pil).astype(np.float32) / 255.0
            output_mask = torch.from_numpy(output_mask_np)[None,]
        else:
            # If no mask was ever created, output a blank black mask.
            output_mask = torch.zeros(1, original_height, original_width)


        # Create a helpful info string with a summary of the transformations applied.
        output_info_string = f"Rot: {rotation_angle:.1f}, Zoom: {zoom_factor:.3f}, Scale(x,y): ({local_scale_x:.2f}, {local_scale_y:.2f}), Offset: ({offset_x:.1f}, {offset_y:.1f}), Tiling: {tiling_mode}"
        
        # Return all the final values.
        return (output_image, output_mask, output_info_string)