Download the dataset and put it in this folder.

There are "Collar" and "Sleeve" two datasets. And for each dataset, we have the cropped out collar and/or sleeve part. And the masked image which we called "src" image.

There are four csv files to use the dataset. We divide the dataset as train and test correspondingly. 
For the collar csv file:
	"src_imgPath" refers to the images with collar part cropped out.
	"part_imgPath" refers to color collar part images.
	"part_edgePath" refers to the collar edge images.
	"tgt_imgPath" refers to the target synthesized result images.
	"mask_imgPath" refers to the temporal mask images we used, which is no used during training or testing process.
	"landmark1_x/y" refers to the up-left point of the collar, "landmark2_x/y" means the down-right point of the collar.
	"bbox_x/y" refers to the coordinates of the cropping bounding box.
	"collar_type" refers to the label of the collar type corresponding to the paper (start from 0)
	"edge_imgPath" refers to the edge of the intergral cloth images, which is used in pre-processing step and not used during training or testing process.

For the sleeve csv file:
	"imageName" refers to the target synthesized result images.
	"orig_H" refers to the image height before we resized.
	"orig_W" refers to the image width before we resized.
	"shoudler1/2_x/y" refers to the shoulders point of the cloth.
	"sleeve1/2_x/y" refers to the cuff point of the cloth.
	"type" refers to the sleeve type, 0 means short sleeve and 1 means long sleeve.
	"cropped_img_path" refers to images with sleeve part cropped out.
	"edge_path" refers to the intergral edge of the cloth images, which also is not used during training and testing process.
	"CroppedSleeve" refers to the edge of the sleeve part images.

The experiments dataset part are randomly selected from the testSet, please use the pandas package to build your own test for leave one out experiment and sleeve test set.

If you have any questions, please email to the authors. If you think the dataset is useful, please cite our paper. Thank you!
