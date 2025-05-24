from PIL import Image
import os
from object_detection import detect_rois
from shape_alignment import detect_points
import matplotlib.pyplot as plt
import os

input_folder = "../../INPUT_IMAGES"
output_folder = "../../OUTPUT/OUTPUT_YOLO"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
results_txt_path = os.path.join(output_folder, "results.txt")
results_file = open(results_txt_path, "a+")
# Write header
results_file.write("Image Name, Correction Angle (Degrees), "
                    "Femur Head (x,y), OST Point (x,y), "
                    "Knee Inner (x,y), Knee Outer (x,y), "
                    "Ankle Inner (x,y), Ankle Outer (x,y)\n")
results_file.close()

# Convert all input images to gray scale
# overwite the gray scale one into the old one
def transfer_and_grayscale(directory):
    for picture in os.listdir(directory):
        image = os.path.join(directory, picture)

        img = Image.open(image).convert('L')
        img.save(f"{directory}/{picture.split('.')[0]}.png")

transfer_and_grayscale(input_folder)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):  # Process only PNG files
        input_path = os.path.join(input_folder, filename)

        #output_image_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_aligned.png")
        output_image_path = os.path.join(output_folder, os.path.splitext(filename)[0])
        try:
            # Load the image
            image = Image.open(input_path)
            detected_rois = detect_rois(image)
            aligned_points = detect_points(image, detected_rois)
            rl = 0
            for results in aligned_points.values():
                plt.figure(dpi=300)
                plt.imshow(image, cmap='gray')
                plt.scatter(results.femur_head.x, results.femur_head.y, s=1)
                plt.scatter(results.ost_point.x, results.ost_point.y, s=1)
                plt.scatter(results.knee_inner.x, results.knee_inner.y, s=1)
                plt.scatter(results.knee_outer.x, results.knee_outer.y, s=1)
                plt.scatter(results.ankle_inner.x, results.ankle_inner.y, s=1)
                plt.scatter(results.ankle_outer.x, results.ankle_outer.y, s=1)
                plt.axis('off')
                plt.title(f"Correction angle: {results.correction_angle_in_deg}\N{DEGREE SIGN}")

                results_file = open(results_txt_path, "a+")

                # Save the figure
                if(rl==0):
                    plt.savefig(output_image_path+"_out_left.png", bbox_inches='tight', pad_inches=0, dpi=300)
                    # Save results in the text file
                    results_file.write(f"{filename}_left, {results.correction_angle_in_deg:.2f}, "
                                  f"({results.femur_head.x}, {results.femur_head.y}), "
                                  f"({results.ost_point.x}, {results.ost_point.y}), "
                                  f"({results.knee_inner.x}, {results.knee_inner.y}), "
                                  f"({results.knee_outer.x}, {results.knee_outer.y}), "
                                  f"({results.ankle_inner.x}, {results.ankle_inner.y}), "
                                  f"({results.ankle_outer.x}, {results.ankle_outer.y})\n")
                    print(f"Processed: {filename}_left")
                else:
                    plt.savefig(output_image_path+"_out_right.png", bbox_inches='tight', pad_inches=0, dpi=300)
                    # Save results in the text file
                    results_file.write(f"{filename}_right, {results.correction_angle_in_deg:.2f}, "
                                  f"({results.femur_head.x}, {results.femur_head.y}), "
                                  f"({results.ost_point.x}, {results.ost_point.y}), "
                                  f"({results.knee_inner.x}, {results.knee_inner.y}), "
                                  f"({results.knee_outer.x}, {results.knee_outer.y}), "
                                  f"({results.ankle_inner.x}, {results.ankle_inner.y}), "
                                  f"({results.ankle_outer.x}, {results.ankle_outer.y})\n")
                    print(f"Processed: {filename}_right")
                rl+=1
                results_file.close()
                plt.close()
                #plt.show()


                #plt.plot([results.femur_head.x, results.ost_point.x],[results.femur_head.y, results.ost_point.y],color="blue",label="Femur Head to OST Point")
                #plt.savefig('outputs/'+str(img)+"_"+str(rl)+'_plt.png', bbox_inches='tight')



        except:
            print("some error occured at "+ input_path)
            results_file.close()

    else:
        print("No PNG files found")
