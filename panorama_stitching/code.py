"""

    BBM418 Computer Vision Lab. Assignment 2.
    by Mehmet Giray Nacakci, 21989009.


    * * * REQUIRED: python 3.7.8  , or any version between (3.4 <= this <= 3.7)
    * * * REQUIRED: pip install opencv-contrib-python==3.4.2.17
        These are because, the only way to get all SIFT, SURF, ORB algorithms working
        is by using OpenCV 3.4.2.17, due to licensing/patenting reasons.


    The /dataset folder containing image sets should be put in the same location as
    /code folder, so that /code/code.py can read the images.


    After running the code.py (runs main() function),

    "/output/" folder will be created and filled with Matching Results and Panorama Results
    separately for each imageset, for each Feature Extraction Algorithm: SIFT, SURF, ORB

    Results will also be displayed in Pycharm or any IDE.

"""

import cv2
import os
import time
import numpy as np
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 450

saving_directory = "../output/"

image_sets = []  # [folder name, 1.png, 2.png, ...], [folder name, 1.png, 2.png, ...], ...

# feature extraction algorithm performance comparison
sift_passed_time = 0
sift_total_features = 0
surf_passed_time = 0
surf_total_features = 0
orb_passed_time = 0
orb_total_features = 0


def read_image_dataset():
    folders = [i + "/" for i in os.listdir("../dataset/") if i != ".DS_Store"]
    for image_set in folders:

        images_in_set = [image_set.strip("/")]
        img_files = [i for i in os.listdir("../dataset/" + image_set) if (i != ".DS_Store" and ("png" in i.lower() or "jpg" in i.lower() or "jpeg" in i.lower()))]
        for img_file in sorted(img_files):  # "1.png"
            images_in_set.append("../dataset/" + image_set + img_file)

        image_sets.append(images_in_set)


def pairwise_matching_and_panorama_stitching():

    for feature_extraction_algorithm in ["SIFT", "SURF", "ORB"]:

        for image_set in image_sets:

            # Initialize Panorama canvas
            first_image = cv2.imread(image_set[1])
            stitched_canvas = np.copy(first_image)
            min_x_of_canvas = 0
            min_y_of_canvas = 0
            max_x_of_canvas = np.shape(first_image)[1]
            max_y_of_canvas = np.shape(first_image)[0]

            """ FEATURE MATCHING Image Pairs """

            match_plots = []
            for i in range(1, len(image_set)-1):
                """ matching 1.png with 2.png, ..., 1.png with n.png """
                img_path_1 = image_set[1]
                img_path_2 = image_set[i+1]

                img_gray_1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
                img_1 = cv2.imread(img_path_1)
                img_1_name = img_path_1.strip("/").split("/")[-1]
                img_gray_2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
                img_2 = cv2.imread(img_path_2)
                img_2_name = img_path_2.strip("/").split("/")[-1]

                key_points_1 = None
                img_1_features = None
                descriptors_1 = None
                key_points_2 = None
                img_2_features = None
                descriptors_2 = None
                brute_force_matcher = None

                start_time = time.time()

                if feature_extraction_algorithm == "SIFT":

                    extractor = cv2.xfeatures2d.SIFT_create(1500)
                    key_points_1, descriptors_1 = extractor.detectAndCompute(img_gray_1, None)
                    img_1_features = cv2.drawKeypoints(img_1, key_points_1, None)

                    key_points_2, descriptors_2 = extractor.detectAndCompute(img_gray_2, None)
                    img_2_features = cv2.drawKeypoints(img_2, key_points_2, None)

                    brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

                    global sift_passed_time
                    sift_passed_time += time.time() - start_time
                    global sift_total_features
                    sift_total_features += np.shape(key_points_1)[0] + np.shape(key_points_2)[0]

                if feature_extraction_algorithm == "SURF":

                    extractor = cv2.xfeatures2d.SURF_create(400)  # hessianThreshold parameter, 400 is the cv recommended value
                    key_points_1, descriptors_1 = extractor.detectAndCompute(img_gray_1, None)
                    img_1_features = cv2.drawKeypoints(img_1, key_points_1, None)

                    key_points_2, descriptors_2 = extractor.detectAndCompute(img_gray_2, None)
                    img_2_features = cv2.drawKeypoints(img_2, key_points_2, None)

                    brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

                    global surf_passed_time
                    surf_passed_time += time.time() - start_time
                    global surf_total_features
                    surf_total_features += np.shape(key_points_1)[0] + np.shape(key_points_2)[0]

                if feature_extraction_algorithm == "ORB":

                    extractor = cv2.ORB_create(nfeatures=1500)
                    key_points_1, descriptors_1 = extractor.detectAndCompute(img_gray_1, None)
                    img_1_features = cv2.drawKeypoints(img_1, key_points_1, None)

                    key_points_2, descriptors_2 = extractor.detectAndCompute(img_gray_2, None)
                    img_2_features = cv2.drawKeypoints(img_2, key_points_2, None)

                    brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                    global orb_passed_time
                    orb_passed_time += time.time() - start_time
                    global orb_total_features
                    orb_total_features += np.shape(key_points_1)[0] + np.shape(key_points_2)[0]

                """ FEATURE MATCHING """

                feature_matches = brute_force_matcher.match(descriptors_1, descriptors_2)

                # get only top 80% matches, (disregard noise):
                feature_matches = list(feature_matches)
                feature_matches.sort(key=lambda m: m.distance)
                feature_matches = feature_matches[: int(len(feature_matches) * 0.8)]

                # Plotting Matched Points
                feature_matched = cv2.drawMatches(img_1, key_points_1, img_2, key_points_2, feature_matches, None)
                # Concatenating plots to draw them later
                total_height_for_concatenated_plot = max(np.shape(img_1_features)[0], np.shape(img_2_features)[0], np.shape(feature_matched)[0])
                white_space = np.full((total_height_for_concatenated_plot, 100, 3), 255, dtype=np.uint8)
                # make them all the same height to concatenate and display side by side: img_1 features, img_2 features, matching lines.
                aaa = total_height_for_concatenated_plot - np.shape(img_1_features)[0]
                if aaa > 0:
                    img_1_features = np.pad(img_1_features, ((aaa, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)
                aaa = total_height_for_concatenated_plot - np.shape(img_2_features)[0]
                if aaa > 0:
                    img_2_features = np.pad(img_2_features, ((aaa, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)
                aaa = total_height_for_concatenated_plot - np.shape(feature_matched)[0]
                if aaa > 0:
                    feature_matched = np.pad(feature_matched, ((aaa, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)
                two_images_and_their_match = np.concatenate((img_1_features, white_space, img_2_features, white_space, feature_matched), axis=1, dtype=np.uint8)
                match_plots.append((two_images_and_their_match, image_set[0], img_1_name, img_2_name))

                """ FIND HOMOGRAPHY (TRANSFORMATION)"""

                matched_points_of_img_1 = np.float32([key_points_1[m.queryIdx].pt for m in feature_matches]).reshape(-1, 1, 2)
                matched_points_of_img_2 = np.float32([key_points_2[m.trainIdx].pt for m in feature_matches]).reshape(-1, 1, 2)
                homography_matrix_to_transform_img_2_to_img_1, matched_region_mask = cv2.findHomography(matched_points_of_img_2, matched_points_of_img_1, cv2.RANSAC)

                """  STITCHING  THE PANORAMA """

                # Obtain the area that the transformed img_2 will reside on panorama canvas.
                img_2_width = np.shape(img_2)[1]
                img_2_height = np.shape(img_2)[0]
                corners_of_img_2 = np.float32([[0, 0], [0, img_2_height - 1], [img_2_width - 1, img_2_height - 1], [img_2_width - 1, 0]]).reshape(-1, 1, 2)
                corners_of_img_2_transformed = np.int32(cv2.perspectiveTransform(corners_of_img_2, homography_matrix_to_transform_img_2_to_img_1))
                x_corners = corners_of_img_2_transformed[:, :, 0]
                y_corners = corners_of_img_2_transformed[:, :, 1]
                min_x = np.min(x_corners)
                max_x = np.max(x_corners)
                min_y = np.min(y_corners)
                max_y = np.max(y_corners)

                # padding panorama canvas with new ("widest") margins of transformed image, in order to fit everything.
                rows_to_bottom = max(max_y - max_y_of_canvas, 0)
                rows_to_top = max(min_y_of_canvas - min_y, 0)
                columns_to_right = max(max_x - max_x_of_canvas, 0)
                columns_to_left = max(min_x_of_canvas - min_x, 0)
                stitched_canvas = np.pad(stitched_canvas, ((rows_to_top, rows_to_bottom), (columns_to_left, columns_to_right), (0, 0)), mode='constant', constant_values=0)

                min_x_of_canvas = min(min_x_of_canvas, min_x)
                max_x_of_canvas = max(max_x, max_x_of_canvas)
                min_y_of_canvas = min(min_y_of_canvas, min_y)
                max_y_of_canvas = max(max_y, max_y_of_canvas)

                # Apply Transformation
                translate_the_homography_to_fit = np.array([[1, 0, abs(min_x_of_canvas)], [0, 1, abs(min_y_of_canvas)], [0, 0, 1]])
                h = np.dot(translate_the_homography_to_fit, homography_matrix_to_transform_img_2_to_img_1)
                only_2 = cv2.warpPerspective(img_2, h, (np.shape(stitched_canvas)[1], np.shape(stitched_canvas)[0]))

                # APPLY STITCHING new image onto the canvas
                mask_for_non_black_region = np.sum(stitched_canvas, axis=2) == 0
                stitched_canvas[mask_for_non_black_region, :] = only_2[mask_for_non_black_region, :]

            """ PLOTTING THE RESULTS"""

            # Draw all Image-Pair-Matches as 1 plot
            grid_size = (3, 2)
            # Assuming the image set has 6 images.
            # If more, they are not plotted here.
            # If less, plot will have empty space but will not give error.
            cell1 = pyplot.subplot2grid(grid_size, (0, 0))
            cell1.imshow(cv2.cvtColor(match_plots[0][0], cv2.COLOR_BGR2RGB))
            cell1.axis('off')
            cell1.text(0.5, 1.05, feature_extraction_algorithm + " MATCHED  " + match_plots[0][1] + "  " + match_plots[0][2] + " " + match_plots[0][3], ha="center", va="bottom", transform=cell1.transAxes)
            if len(match_plots) > 1:
                cell2 = pyplot.subplot2grid(grid_size, (0, 1))
                cell2.imshow(cv2.cvtColor(match_plots[1][0], cv2.COLOR_BGR2RGB))
                cell2.axis('off')
                cell2.text(0.5, 1.05, feature_extraction_algorithm + " MATCHED  " + match_plots[1][1] + "  " + match_plots[1][2] + " " + match_plots[1][3], ha="center", va="bottom", transform=cell2.transAxes)
            if len(match_plots) > 2:
                cell3 = pyplot.subplot2grid(grid_size, (1, 0))
                cell3.imshow(cv2.cvtColor(match_plots[2][0], cv2.COLOR_BGR2RGB))
                cell3.axis('off')
                cell3.text(0.5, 1.05, feature_extraction_algorithm + " MATCHED  " + match_plots[2][1] + "  " + match_plots[2][2] + " " + match_plots[2][3], ha="center", va="bottom", transform=cell3.transAxes)
            if len(match_plots) > 3:
                cell4 = pyplot.subplot2grid(grid_size, (1, 1))
                cell4.imshow(cv2.cvtColor(match_plots[3][0], cv2.COLOR_BGR2RGB))
                cell4.axis('off')
                cell4.text(0.5, 1.05, feature_extraction_algorithm + " MATCHED  " + match_plots[3][1] + "  " + match_plots[3][2] + " " + match_plots[3][3], ha="center", va="bottom", transform=cell4.transAxes)
            if len(match_plots) > 4:
                cell5 = pyplot.subplot2grid(grid_size, (2, 0))  # , colspan=1)
                cell5.imshow(cv2.cvtColor(match_plots[4][0], cv2.COLOR_BGR2RGB))
                cell5.axis('off')
                cell5.text(0.5, 1.05, feature_extraction_algorithm + " MATCHED  " + match_plots[4][1] + "  " + match_plots[4][2] + " " + match_plots[4][3], ha="center", va="bottom", transform=cell5.transAxes)
            if len(match_plots) > 5:  # if the 7th image exists, draw the 6th match plot
                cell6 = pyplot.subplot2grid(grid_size, (2, 1))
                cell6.imshow(cv2.cvtColor(match_plots[4][0], cv2.COLOR_BGR2RGB))
                cell6.axis('off')
                cell6.text(0.5, 1.05, feature_extraction_algorithm + " MATCHED  " + match_plots[5][1] + "  " + match_plots[5][2] + " " + match_plots[5][3], ha="center", va="bottom", transform=cell6.transAxes)
            pyplot.tight_layout()
            pyplot.savefig(saving_directory + "_" + match_plots[0][1] + "_" + feature_extraction_algorithm + "_" + "features.jpg", dpi=450)
            pyplot.show()

            # Draw PANORAMA
            # Does not matter how many input images.
            pyplot.imshow(cv2.cvtColor(stitched_canvas, cv2.COLOR_BGR2RGB))
            pyplot.title(feature_extraction_algorithm + " Panorama :  " + match_plots[0][1], fontsize=28, pad=20)
            pyplot.savefig(saving_directory + "_" + match_plots[0][1] + "_" + feature_extraction_algorithm + "_" + "panorama.jpg", dpi=450)
            pyplot.show()


def print_algorithm_comparison():
    print("\nSIFT detected  " + str(sift_total_features) + "  features in  " + str(int(sift_passed_time)) + " seconds.")
    print("SIFT detected  " + str(int(sift_total_features / sift_passed_time)) + "  features/second")

    print("\nSURF detected  " + str(surf_total_features) + "  features in  " + str(int(surf_passed_time)) + " seconds.")
    print("SURF detected  " + str(int(surf_total_features / surf_passed_time)) + "  features/second")

    print("\nORB detected  " + str(orb_total_features) + "  features in  " + str(int(orb_passed_time)) + " seconds.")
    print("ORB detected  " + str(int(orb_total_features / orb_passed_time)) + "  features/second")


def main():
    start = time.time()

    read_image_dataset()

    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    pairwise_matching_and_panorama_stitching()

    print_algorithm_comparison()

    print("\n Execution completed. Panorama images are rendered. Total time:  ", int(time.time() - start), " seconds.\n")


main()
