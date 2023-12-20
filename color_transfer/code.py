"""
BBM415 Image Processing Lab. Assignment 2.
by Mehmet Giray Nacakci, 21989009.


If I submitted the "data" folder empty (due to submit system upload size limit),
input image files (same as the "data2.zip" given in piazza resources)
can be found in this drive link:

https://drive.google.com/drive/folders/1Ant6Nr17sObcwJ17a7MdYshZR1JsPHBa?usp=sharing

The "data" folder containing input (source and target) images should be put in the same
location(directory) as all other files I submitted, so that /code/code.py can read the images.



After running the code.py (runs main() function),

The "/output/" folder will be filled with Resulting image-histogram plots,
each subfolder (/part1, /part2_mxn divisions ...) will be filled during code execution.

Results will also be displayed in the Pycharm or any IDE.


"""

import numpy as np
import skimage
from skimage.color import rgb2lab, lab2rgb
from matplotlib import pyplot
import time
import warnings


def draw_images_and_histograms(img1, img2, img3, input_file_name, part_number, subfolder):

    display_3_images = pyplot.figure(figsize=(19, 6))
    pyplot.title(label=" " + part_number.replace("_part", "PART ") + " :  " + input_file_name.replace("../data/", "") + "   Color Transfer Results", loc="center", y=1.08, fontsize=20)
    pyplot.axis('off')

    # Source image
    display_3_images.add_subplot(2, 3, 1)
    pyplot.imshow(img1)
    pyplot.axis('off')
    pyplot.title("Source", fontsize=18)

    # Target image
    display_3_images.add_subplot(2, 3, 2)
    pyplot.imshow(img2)
    pyplot.axis('off')
    pyplot.title("Target", fontsize=18)

    # Result image
    display_3_images.add_subplot(2, 3, 3)
    pyplot.imshow(img3)
    pyplot.axis('off')
    pyplot.title("Result", fontsize=18)

    # histogram for Source image
    display_3_images.add_subplot(2, 3, 4)
    pyplot.hist(img1[:, :, 0].flatten(), bins=range(5, 255), color='red', alpha=0.4)
    pyplot.hist(img1[:, :, 1].flatten(), bins=range(5, 255), color='green', alpha=0.4)
    pyplot.hist(img1[:, :, 2].flatten(), bins=range(5, 255), color='blue', alpha=0.4)
    pyplot.xlabel('intensity', fontsize=16)
    pyplot.ylabel('pixel Count', fontsize=16)
    pyplot.legend(['Red', 'Green', 'Blue'])

    # histogram for Target image
    display_3_images.add_subplot(2, 3, 5)
    pyplot.hist(img2[:, :, 0].flatten(), bins=range(5, 255), color='red', alpha=0.4)
    pyplot.hist(img2[:, :, 1].flatten(), bins=range(5, 255), color='green', alpha=0.4)
    pyplot.hist(img2[:, :, 2].flatten(), bins=range(5, 255), color='blue', alpha=0.4)
    pyplot.xlabel('intensity', fontsize=16)
    pyplot.legend(['Red', 'Green', 'Blue'])

    # histogram for Result image
    display_3_images.add_subplot(2, 3, 6)
    pyplot.hist(img3[:, :, 0].flatten(), bins=range(5, 255), color='red', alpha=0.4)
    pyplot.hist(img3[:, :, 1].flatten(), bins=range(5, 255), color='green', alpha=0.4)
    pyplot.hist(img3[:, :, 2].flatten(), bins=range(5, 255), color='blue', alpha=0.4)
    pyplot.xlabel('intensity', fontsize=16)
    pyplot.legend(['Red', 'Green', 'Blue'])

    # display results in IDE as a plot
    display_3_images.show()

    # save results as images in output folder
    new_name = input_file_name.replace("../data/in", "../output/" + subfolder + "/output")
    new_name = new_name.replace(".png", part_number + ".png")
    display_3_images.savefig(new_name)


def transfer_image_color_of_target_to_source(source_rgb, target_rgb):

    """ Reinhard's Algorithm "Color Transfer Between Images" """

    source_lab = rgb2lab(source_rgb)
    l_source, a_source, b_source = source_lab[:, :, 0], source_lab[:, :, 1], source_lab[:, :, 2]

    target_lab = rgb2lab(target_rgb)
    l_target, a_target, b_target = target_lab[:, :, 0], target_lab[:, :, 1], target_lab[:, :, 2]

    l_source_subtracted = l_source - np.mean(l_source)
    a_source_subtracted = a_source - np.mean(a_source)
    b_source_subtracted = b_source - np.mean(b_source)

    l_scaled = l_source_subtracted * (np.std(l_source) / np.std(l_target))
    a_scaled = a_source_subtracted * (np.std(a_source) / np.std(a_target))
    b_scaled = b_source_subtracted * (np.std(b_source) / np.std(b_target))

    l_result = l_scaled + np.mean(l_target)
    a_result = a_scaled + np.mean(a_target)
    b_result = b_scaled + np.mean(b_target)

    # some values are out of range (code below still does not fully solve the problem)
    l_clipped = np.clip(l_result, 0, 100)
    a_clipped = np.clip(a_result, -127, 127)
    b_clipped = np.clip(b_result, -127, 127)

    result_lab = np.dstack((l_clipped, a_clipped, b_clipped))
    result_rgb = lab2rgb(result_lab)
    result_rgb = (result_rgb * 255).astype(int)
    result_rgb = np.clip(result_rgb, 0, 255)

    return result_rgb


def part_1(source_and_target_images):
    """ global: Apply Reinhard's algorithm to the entire Source image """

    for source_and_target in source_and_target_images:

        start = time.time()
        source_rgb = skimage.io.imread(source_and_target[0])
        target_rgb = skimage.io.imread(source_and_target[1])

        result_rgb = transfer_image_color_of_target_to_source(source_rgb, target_rgb)

        """ Display in IDE, also save the output as image"""
        draw_images_and_histograms(source_rgb, target_rgb, result_rgb, source_and_target[0], "_part1", "part1")

        # elapsed time
        print("PART 1 : Results of input  " + source_and_target[0] + "  took  " + str(round(time.time() - start)) + "  seconds to process.")

    print("\n")


def crop_to_equalize_size(source_rgb, target_rgb):
    """ If any of Target or Source image is bigger than other in width or height, crop edges to equalize size. """

    source_width = np.shape(source_rgb)[1]
    source_height = np.shape(source_rgb)[0]
    target_width = np.shape(target_rgb)[1]
    target_height = np.shape(target_rgb)[0]
    cropped_width = min(source_width, target_width)
    cropped_height = min(source_height, target_height)

    source_height_crop = (source_height - cropped_height) // 2 + 1
    source_width_crop = (source_width - cropped_width) // 2 + 1
    source_cropped = source_rgb[source_height_crop: -source_height_crop, source_width_crop: -source_width_crop]

    target_height_crop = (target_height - cropped_height) // 2 + 1
    target_width_crop = (target_width - cropped_width) // 2 + 1
    target_cropped = target_rgb[target_height_crop: -target_height_crop, target_width_crop: -target_width_crop]

    # a few pixels of inequality can still exist (due to rounding and odd even differences)
    crop_source_width_again = np.shape(source_cropped)[1] - (cropped_width - 2)
    if crop_source_width_again > 0:
        source_cropped = source_cropped[:, : -crop_source_width_again]
    crop_source_height_again = np.shape(source_cropped)[0] - (cropped_height - 2)
    if crop_source_height_again > 0:
        source_cropped = source_cropped[: -crop_source_height_again, :]
    crop_target_width_again = np.shape(target_cropped)[1] - (cropped_width - 2)
    if crop_target_width_again > 0:
        target_cropped = target_cropped[:, : -crop_target_width_again]
    crop_target_height_again = np.shape(target_cropped)[0] - (cropped_height - 2)
    if crop_target_height_again > 0:
        target_cropped = target_cropped[: -crop_target_height_again, :]

    return source_cropped, target_cropped


def part_2(source_and_target_images, approach_1_is_true_else_approach_2_is_false, heightDivide, widthDivide):
    """ local: Apply Reinhard's algorithm to local regions in the Source image to obtain more precise results. """

    for source_and_target in source_and_target_images:

        start = time.time()
        source_rgb = skimage.io.imread(source_and_target[0])
        target_rgb = skimage.io.imread(source_and_target[1])

        # Sizes need to be equalized so that firstly SSD comparison, then color transfer can be applied to same size regions.
        source_cropped, target_cropped = crop_to_equalize_size(source_rgb, target_rgb)

        # Divide both images to equal size rectangular regions
        rectangle_height = np.shape(source_cropped)[0] // heightDivide
        crop_height_again = np.shape(source_cropped)[0] % heightDivide
        rectangle_width = np.shape(source_cropped)[1] // widthDivide
        crop_width_again = np.shape(source_cropped)[0] % heightDivide

        source_regions = []
        target_regions = []
        for h in range(heightDivide):
            for w in range(widthDivide):
                source_region = source_cropped[h * rectangle_height: (h+1) * rectangle_height, w * rectangle_width: (w+1) * rectangle_width]
                if np.shape(source_region)[0] > 0 and np.shape(source_region)[1] > 0:
                    source_regions.append(source_region)

                    target_region = target_cropped[h * rectangle_height: (h+1) * rectangle_height, w * rectangle_width: (w+1) * rectangle_width]
                    target_regions.append(target_region)

        # crop again the little bits left unequal after dividing to regions.
        if crop_height_again > 0:
            source_cropped = source_cropped[: -crop_height_again, :]
            target_cropped = target_cropped[: -crop_height_again, :]
        if crop_width_again > 0:
            source_cropped = source_cropped[:, : -crop_width_again]
            target_cropped = target_cropped[:, : -crop_width_again]

        if approach_1_is_true_else_approach_2_is_false:

            """
            APPROACH 1: 
            
            For each region of Source image: 
                compare this Source region with all Target regions, choose ("most similar") region with smallest SSD as best match.
                Then, transfer this matched Target region's color to Source region. 
                
                Sometimes chaotically different regions can get color transferred.
                Thus, this does not always look good. Some images can almost look like their colors are inverted.
            """
            region_results = []
            for source_region in source_regions:
                smallest_ssd = float("inf")
                best_match = None

                for j, target_region in enumerate(target_regions):

                    """ Different variations to calculate SSD """

                    """ a) """
                    # Greyscale  # best version, industry standard
                    source_grayscale = source_region[:, :, 0] + source_region[:, :, 1] + source_region[:, :, 2] // 3
                    target_grayscale = target_region[:, :, 0] + target_region[:, :, 1] + target_region[:, :, 2] // 3
                    sum_of_square_differences = np.sum((source_grayscale - target_grayscale).astype(np.float32) ** 2)

                    """ b) """
                    # R G B separately
                    # sum_of_square_differences = np.sum((source_cropped - target_cropped).astype(np.float32) ** 2)

                    """ c) """
                    # L A B separately
                    # source_lab = rgb2lab(source_region)
                    # target_lab = rgb2lab(target_region)
                    # sum_of_square_differences = np.sum((source_lab - target_lab).astype(np.float32) ** 2)

                    """ d) """
                    # L (only intensity) channel of LAB
                    # source_l = rgb2lab(source_region)[:, :, 0]
                    # target_l = rgb2lab(target_region)[:, :, 0]
                    # sum_of_square_differences = np.sum((source_l - target_l).astype(np.float32) ** 2)

                    # find minimum SSD
                    if sum_of_square_differences < smallest_ssd:
                        smallest_ssd = sum_of_square_differences
                        best_match = target_region

                # transfer matched target region's color to source region.
                region_result = transfer_image_color_of_target_to_source(source_region, best_match)

                region_results.append(region_result)

        else:
            """ 
            APPROACH 2:  
            
            Skip SSD, just assign each rectangular Source region to same position Target region. 
            Color transfer is done between same position regions. 

            In my opinion, this approach enabled me to transform colors more meaningfully since the Resulting image's 
            spacial color distribution looks much more like the Target image, compared to Approach 1. 
            """
            region_results = []
            for source_region, target_region in zip(source_regions, target_regions):
                region_result = transfer_image_color_of_target_to_source(source_region, target_region)
                region_results.append(region_result)

        # Glue the seperated regions together to form the Resulting image.
        rows_of_regions = []
        for h in range(heightDivide):
            this_row_of_regions = region_results[h * widthDivide: (h+1) * widthDivide]
            rows_of_regions.append(np.hstack(tuple(this_row_of_regions)))
        result_rgb = np.vstack(tuple(rows_of_regions))

        """ Display in IDE, also save the output as image"""
        approach_number = "1" if approach_1_is_true_else_approach_2_is_false else "2"
        subfolder = "part2_" + str(heightDivide) + "x" + str(widthDivide)
        draw_images_and_histograms(source_cropped, target_cropped, result_rgb, source_and_target[0], "_part2" + "_approach" + approach_number, subfolder)

        # elapsed time
        print("PART 2 :  " + "Approach " + approach_number + ", Division = " + subfolder.replace("part2_", "") + " ," + "  Results of input  " + source_and_target[0] + "  took  " + str(round(time.time() - start)) + "  seconds to process.")

    print("\n")


def main():

    global_start = time.time()

    source_images = ["in_01.png", "in_02.png", "in_03.png", "in_05.png", "in_06.png", "in_08.png", "in_09.png",
                     "in_11.png", "in_15.png", "in_20.png", "in_24.png", "in_29.png", "in_50.png", "in_51.png",
                     "in_56.png"]
    target_images = ["tar_01.png", "tar_02.png", "tar_03.png", "tar_05.png", "tar_06.png", "tar_08.png", "tar_09.png",
                     "tar_11.png", "tar_15.png", "tar_20.png", "tar_24.png", "tar_29.png", "tar_50.png", "tar_51.png",
                     "tar_56.png"]

    source_and_target_images = []
    for i, j in zip(source_images, target_images):
        source_and_target_images.append(("../data/" + i, "../data/" + j))

    # Part 1
    part_1(source_and_target_images)

    """ Try to optimize the Color Transfer Results with different region division amounts. """
    try_these_divisions = [(4, 7), (15, 20), (40, 70), (80, 140)]  # region counts, not pixels
    for try_this in try_these_divisions:

        # Approach 1
        part_2(source_and_target_images, approach_1_is_true_else_approach_2_is_false=True, heightDivide=try_this[0], widthDivide=try_this[1])
        # Approach 2
        part_2(source_and_target_images, approach_1_is_true_else_approach_2_is_false=False, heightDivide=try_this[0], widthDivide=try_this[1])

    # time elapsed
    seconds = time.time() - global_start
    minutes = seconds//60
    seconds -= 60*minutes
    print("\n\nThis is the end. Check 'output' folder for Results. Thank you for your patience.   Total Time elapsed:  ")
    print('Total Time elapsed:   %d:%d   minutes:seconds' % (minutes, seconds))


# skimage was printing distracting warnings I could not solve.
# while converting from lab to rgb, out of range warning. I tried value clipping,
# experimenting data type and range changing, and so on. Yet, no avail.
# This might have affected the results and not good enough colors in result images.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    main()  # run main() without showing warnings
