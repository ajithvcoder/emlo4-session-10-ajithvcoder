#!/bin/bash

# Add plots to the report

echo "### Accuracy" >> report.md
echo '![](./train_val_acc.png "Training and Validation Accuracy")' >> report.md

echo "### Loss" >> report.md
echo '![](./train_val_loss.png "Training and Validation loss")' >> report.md

echo "### Train Confusion matrix" >> report.md
echo '![](./train_confusion_matrix.png "Training Confusion matrix")' >> report.md

echo "### Test Confusion matrix" >> report.md
echo '![](./test_confusion_matrix.png "Test Confusion matrix")' >> report.md

echo "### Infer Images" >> report.md
# Directory containing images
image_dir="infer_images"

# Report file
report_file="report.md"

# Loop through all images in the directory
for image in "$image_dir"/*; do
    # Get the image filename
    image_filename=$(basename "$image")
    
    # Append the image to the report file
    echo "![](./$image_dir/$image_filename )" >> "$report_file"
done

echo "All images have been appended to $report_file"
