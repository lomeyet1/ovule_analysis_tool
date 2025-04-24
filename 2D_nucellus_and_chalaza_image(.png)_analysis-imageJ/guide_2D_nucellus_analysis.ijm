//////////According to the color distinguishing between the two tissues, 
/////////find the junction point of the outer contour, and connect as the baseline. 
////////Nucellus outline point with the biggest to the base line middle point as the tip of nucellus. 
///////After the end, the user can to check whether this is satisfied with these and can update these landmarks.
//////Then use those landmarks to get angles, distance (bending), and raito.
main();
function main() {
sourceDir = getDirectory("Contains mid-sagittal screenshots (png/jpg) of nucellus (blue) attached to chalaza (red).");

//"C:/Users/13198/2025_intern/development/Ovule_Analysis/intr/2-III_3D_screenshots/";
files = getFileList(sourceDir);
ResultArray = newArray();
for (i = 0; i < files.length; i++) {
    if (files[i].endsWith(".jpg") || files[i].endsWith(".png")) {     // choose these two types images
		roiManager("Reset"); 
		closeimages();                  // close opened images
		setBatchMode(true);             // Do not show any images

        open(sourceDir + files[i]);
		imagepath = getDirectory("image");
		imagetitle = getTitle();
		Angle = getlandmarksAndAngle();
		
		close("results");
		closeimages();
		setBatchMode(false);    // leave to here before fixing that showing wrong roi if use ture

		// Bending Analysis use cleane outline and roi with baseline and centriod
		open(imagepath + "cleaned_outline.tif");
		roiManager("Open",  imagepath + "roi.zip");
		distance = run2DBendingAnalysisModule();
		
		roiManager("Show All without labels");
		roiManager("Select", 5);

		run("Flatten");
		rotate();
		saveAs("Tiff", imagepath+ imagetitle + "_distance="+distance);
//		waitForUser("Bending Analysis result.");
		
		close("results");
		closeimages();
		roiManager("Reset");
		
		// Differential Growth Analysis
		open(imagepath + "cleaned_boundary.tif");
		pos_ant_ratioArray = run2DDifferentialGrowthAnalysisModule();
		posterior_nucellus_area = pos_ant_ratioArray[0];
		anterior_nucellus_area = pos_ant_ratioArray[1];
		pos_ant_ratio = pos_ant_ratioArray[2];
		
		roiManager("Show All without labels");
		roiManager("Select", 0);
		run("Flatten");
		rotate();
		saveAs("Tiff", imagepath+ imagetitle + "_pos_ant_ratio=" + posterior_nucellus_area +","+ anterior_nucellus_area+","+pos_ant_ratio);
//		waitForUser("Differential Growth Analysis result.");
		
		// recording the results in array
		ResultArray = Array.concat(ResultArray, newArray(imagetitle, Angle, distance, pos_ant_ratio, posterior_nucellus_area, anterior_nucellus_area));

        
    }
}
closeimages();
close("roi Manager");
close("Results");   n = 0;  setBatchMode(false); 

for (i = 0; i < lengthOf(ResultArray); i++) {
setResult("Nucellus", n, ResultArray[i]);
setResult("Angle", n, ResultArray[i+1]);
setResult("Distance", n, ResultArray[i+2]);
setResult("Pos_Ant_ratio", n, ResultArray[i+3]);
setResult("Posterior_area", n, ResultArray[i+4]);
setResult("Anteriors_area", n, ResultArray[i+5]);
i = i+5; n = n+1;
}

if (n>0) {updateResults(); saveAs("Results", imagepath+"All_Results.txt");
		  waitForUser((n) + ".png or jpg found and processed; All Results saved in txt. \n press OK to input another folder."); 
		  close("Results"); main(); }
	else {waitForUser("NO .png or jpg found !! \n press OK to input another folder."); 
			main();} 
}


function getlandmarksAndAngle() { 
	imagetitle = getTitle();
	imagepath = getDirectory("image");
	
	run("Split Channels");            // Split the row image based on RGB Channels, 
									  // new 3 images window names as imagetitle+" (green)" , imagetitle+" (red)" , and imagetitle+" (blue)"
	selectWindow(imagetitle+" (green)"); 
	close();
	
	maskname1 = "red";
	selectWindow(imagetitle+" (red)");            // usually we marked the posterior chalaza dark red in MGX
	run("Duplicate...", "title="+ maskname1);    // just copy the image
	makeOutline(0,maskname1);                   // make outline images window named as maskname1+".tif", and add it on ROI as maskname1_outline
	
	selectWindow(imagetitle+" (blue)");       // usually we marked the nucellus blue in MGX
	run("Duplicate...", "title="+"blue");
	makeOutline(1,"blue");
	roiManager("Save", imagepath+"outlines.zip");
	
	maskname3 = "blue_image";
	selectWindow(imagetitle+" (blue)"); 
	run("Duplicate...", "title="+maskname3);


	// Get overlap of the two outline images
	imageCalculator("AND create", maskname1+".tif", "blue"+".tif"); // only overlap area of outlines has intensity = 255
	rename("outline_Intersection");                             // rename the window
	run("Analyze Particles...", "clear display include add");   //make the overlap_areas on roi
	close("Results");
	
	// Number of acquisition of ROI; when do imageCalculator, the ROI reseted and added only overlap areas in
	nROIs = roiManager("count");
	// Each overlap ROI renamed as "overlap_area" + number
	for (i = 0; i < nROIs; i++) {
	    roiManager("Select", i);
	    roiManager("Rename", "overlap_area" + (i + 1));
	    roiManager("Update");
	}
	roiManager("Save", imagepath+"roi_overlap.zip");
	roiManager("Reset");      // clear for adding new roi with good order




	// Find coordinates of the outline_Intersection image with intensity = 255 (i.e. overlap points coordinates)
	selectWindow("outline_Intersection");
	getDimensions(width, height, channels, slices, frames);   // image width is the bigest x, image height is biggest y

	overlapArray = newArray();
	for (y = 0; y < height; y++) {
	    for (x = 0; x < width; x++) {     // looping every coordinates

	        v = getPixel(x, y);       // get intensity of this coordinate
	        if (v == 255) {         // only overlap area of outlines has intensity = 255
	        	overlapArray = Array.concat(overlapArray, newArray(x,y));      // store these overlap points
	        }
	    }
	}
	// to print and add those points
	//for (i = 0; i < overlapArray.length; i++) {
	//	makePoint(overlapArray[i], overlapArray[i+1]);
	//	roiManager("add");
	//    print("(" + overlapArray[i] + ", " + overlapArray[i+1] + ")");
	//    i = i+1;
	//}


	// Get coordinates close to the overlap points, and screen to obtain those on the red outine and blue inside
	// these new points are more accurate baseline points
	
if (overlapArray.length >= 4) {
	onbothArray = newArray();

	for (i = 0; i < lengthOf(overlapArray); i++) {
	X = overlapArray[i];
	Y = overlapArray[i+1];
	i = i+1;

		offsetX = newArray(-1, 1, 0, 0); // X displacement
		offsetY = newArray(0, 0, 1, -1); // Y displacement

		for (j = 0; j < lengthOf(offsetX); j++) {       // the newX, newY is the X,Y moved one picxel away
		    newX = X + offsetX[j];
		    newY = Y + offsetY[j];

			selectWindow(maskname1+".tif");    // check it on red outine
			onRed = getPixel(newX, newY);

			open(imagepath + "blue" + ".tif");// check it on blue inside (use mask of blured blue part)
			onBlue = getPixel(newX, newY);

			if (onRed == 255 && onBlue == 255) {  // both on red outine and blue inside are good baseline points
				onbothArray = Array.concat(onbothArray, newArray(newX, newY));
			}
		}
	}

	// to print and add those points
	//for (i = 0; i < onbothArray.length; i++) {
	//	makePoint(onbothArray[i], onbothArray[i+1]);
	//	roiManager("add");
	//    print("(" + onbothArray[i] + ", " + onbothArray[i+1] + ")");
	//    i = i+1;
	//}


	// If there are more than 2 coordinates in onbothArray, the calculation distance and retain the largest pair
	maxDist = 0;
	maxPair = newArray(0, 0, 0, 0);

	if (lengthOf(onbothArray) > 4) {
	    for (i = 0; i < lengthOf(onbothArray); i++) {

	        for (j = i + 2; j < lengthOf(onbothArray); j++) {
	            dist = sqrt(pow(onbothArray[i] - onbothArray[j], 2) + pow(onbothArray[i+1] - onbothArray[j+1], 2));

	            if (dist > maxDist) {
	                maxDist = dist;
	                maxPair = newArray(onbothArray[i], onbothArray[i+1], onbothArray[j], onbothArray[j+1]);
	            }
	            j = j+1;
	        }

	        i = i+1;
	    }
	} else {  maxPair = onbothArray; }

		// to print and add those points
	//	for (i = 0; i < maxPair.length; i++) {
	//		makePoint(maxPair[i], maxPair[i+1]);
	//		roiManager("add");
	//	    print("(" + maxPair[i] + ", " + maxPair[i+1] + ")");
	//	    i = i+1;
	//	}

	// now we have the baseline edge points
	x1 = maxPair[0];
	y1 = maxPair[1];
	x2 = maxPair[2];
	y2 = maxPair[3];

	// Draw a baseline and later let use to update
	makeLine(x1, y1, x2, y2);
	AddToRoi(0, "baseline");
	closeimages();

	// make a combine mask (bulred before) with different colors
    // combine two masks and show it for user update connect_line
    open(imagepath + maskname1+".tif");
    open(imagepath + "blue"+".tif");
    selectWindow(maskname1+".tif");
    run("Multiply...", "value=0.5");   // make a mask darker a bit
    imageCalculator("OR create",maskname1+".tif" ,  "blue"+ ".tif");  // combine two masks
    setBatchMode(false);
    saveAs("Tiff", imagepath + "masked"+  ".tif");
	roiManager("Show All");
	roiManager("Select", 0);
	// let user update the baseline
	waitForUser("Move the baseline to better place if needs.  \n  It will automatically move to the nearest edge point. \n press OK to update.");
	roiManager("Select", 0);
	roiManager("Update");

} else {
	// make a combine mask (bulred before) with different colors
    // combine two masks and show it for user update connect_line
    open(imagepath + maskname1+".tif");
    open(imagepath + "blue"+".tif");
    selectWindow(maskname1+".tif");
    run("Multiply...", "value=0.5");   // make a mask darker a bit
    imageCalculator("OR create",maskname1+".tif" ,  "blue"+ ".tif");  // combine two masks
    saveAs("Tiff", imagepath + "masked" +  ".tif");
	setBatchMode(false);
	// let user make the baseline if no baseline
	setTool("Multi-point Tool");
	waitForUser("Marks 2 edge points of baseline on the edge of tissue. \n  It will automatically move to the nearest edge point. \n press OK to update.");
	roiManager("Add");
	roiManager("Select", 0);
	getSelectionCoordinates(x, y);
	makeLine(x[0], y[0], x[1], y[1]);
	roiManager("Update");
	roiManager("Rename", "baseline");
}
	
//	setBatchMode(true);
	closeimages();
	open(imagepath + "blue"+".tif");
	roiManager("Select", 0);
    getSelectionCoordinates(x, y); // get updated Coordinates
    x1 = x[0];
    y1 = y[0];
    x2 = x[1];
    y2 = y[1];
	
	// make them on nearest outline point
	outlineArray = getOutlineArray();
	
	open(imagepath + "blue_outline.tif");
	if (getPixel(x1, y1) == 0) { 
	refx = x1;
	refy = y1;
	minIndex = makeItonOutline();
	x1 = outlineArray[minIndex];
	y1 = outlineArray[minIndex+1];
	}
	
	if (getPixel(x2, y2) == 0) { 
	refx = x2;
	refy = y2;
	minIndex = makeItonOutline();
	x2 = outlineArray[minIndex];
	y2 = outlineArray[minIndex+1];
	}
	
	
	roiManager("Select", 0);
	makeLine(x1, y1, x2, y2);
	roiManager("Update");
	
	waitForUser("test the edge points moved on outline.");
	
    // Calculate the straight line
    xmid = (x1 + x2) / 2;
    ymid = (y1 + y2) / 2;

    makePoint(xmid, ymid);
    AddToRoi(1, "base_midpoint");
    makePoint(x1, y1);
    AddToRoi(2, "base_point1");
    makePoint(x2, y2);
    AddToRoi(3, "base_point2");

    roiManager("Show None");
  

    // Nucellus outline point with the biggest Distance to the base line middle point is the tip of nucellus.
    closeimages();
    roiManager("Open",  imagepath + "outlines.zip");  // add outlines saved before; now you have 4 + 2 = 6 roi
    findTip();  // see the function
    closeimages();
    
	setBatchMode(false);
    open(imagepath + "masked" +  ".tif");  // open the combine mask (bulred before) with different colors made before
	roiManager("Show none");
	roiManager("Select", 6);
	waitForUser("Move the tip point to better place if needs. \n It will automatically move to the nearest edge point. \n Press OK to update.");
	roiManager("Update");
//	setBatchMode(true);
	
	getSelectionCoordinates(x_tip_arry, y_tip_arry);
	x_tip = x_tip_arry[0];
	y_tip = y_tip_arry[0];
	
	
	open(imagepath + "blue_outline.tif");
	if (getPixel(x_tip, y_tip) == 0) {     // find its nearest outline points to the tip if the tip is not on outline
	refx = x_tip;
	refy = y_tip;
	minIndex = makeItonOutline();
	x_tip = outlineArray[minIndex];
	y_tip = outlineArray[minIndex+1];
	roiManager("select",6);
	makePoint(x_tip, y_tip);
	roiManager("Update");
	}	
	
	waitForUser("test the tip moved on outline.");
	
	closeimages();
	setBatchMode(true);
	

	// make and measure the angle of tip -> midpoint of baseline -> right end of baseline
	open(imagepath + "masked" +  ".tif");
	makeSelection("angle", newArray(x_tip, xmid, x1), newArray(y_tip, ymid, y1));
	run("Measure");
	Angle = getResult("Angle", 0);
	AddToRoi(7, "angle="+Angle);

	// now save the combine masks with roi on it
	roiManager("Open",  imagepath + "roi_overlap.zip");
	
//	setBatchMode(false);
	roiManager("Show All");
	roiManager("Select", 7);
//	run("Flatten");
    saveAs("Tiff", imagepath +  imagetitle  + "_angle=" + Angle +  ".tif");
    roiManager("Save", imagepath+ imagetitle + "_angle_roi.zip");
//    waitForUser("Saved \n Press OK to next sample.");
    close("Results");
    closeimages();
    
    setBatchMode(false);
	open(imagepath + "blue.tif");
	cleanOutline();
	
	//  fill hole (The intensity of myself is 0, but the upper, bottom, left and right are 255) on outline
	getDimensions(width, height, channels, slices, frames);
	for (y = 0; y < height; y++) {
	    for (x = 0; x < width; x++) {    
	    	v = getPixel(x, y);
	        	
	        if (v == 0 && getPixel(x+1, y) == 255 && getPixel(x-1, y) == 255 
	        	&& getPixel(x, y+1) == 255 &&  getPixel(x, y-1) == 255 ) {
	        	setPixel(x, y, 255); } 
	    }
	}
	
	roiManager("Open",  imagepath+ imagetitle + "_angle_roi.zip");
	
	
	roiManager("Select", newArray(4,7,8,9));
	roiManager("delete");
	
	
	roiManager("Select", 5);
	getSelectionCoordinates(tip_x, tip_y);
	roiManager("Select", 1);
	getSelectionCoordinates(xmid, ymid);
	
	roiManager("Select", 4);
	List.setMeasurements;
	makePoint(List.getValue("X"), List.getValue("Y"));//get the coordinates of the centroid of the boundary of the image
	AddToRoi(6, "centroid");
	getSelectionCoordinates(centroid_x, centroid_y); 
	
	
	makeLine(tip_x[0], tip_y[0], centroid_x[0], centroid_y[0], xmid[0], ymid[0]);
	run("Fit Spline");  
	
	AddToRoi(7, "Fit_Spline");
	
	run("Line Width...", "line=2");  // incase cannot be splited in Differential Growth Analysis
	run("Draw");
	
	roiManager("Save", imagepath+"roi.zip");
	roiManager("reset");
	saveAs("Tiff", imagepath + "cleaned_boundary.tif");
    
    return Angle
    }


function rotate() {
roiManager("reset");
roiManager("Open",  imagepath + "roi.zip");

roiManager("Select", 0);  // baseline
getSelectionCoordinates(xLine, yLine);
roiManager("Reset");
// get rotation angle which = baseline end point B with lower y -> baseline end point A with higher y -> the Horizontal line
if (yLine[0] > yLine[1]) {
       xA = xLine[0]; yA = yLine[0];  xB = xLine[1]; yB = yLine[1];} 
 else {xA = xLine[1]; yA = yLine[1];  xB = xLine[0]; yB = yLine[0];}
//  get rotation angle
deltaX = xB - xA;  deltaY = yB - yA;
angle = atan2(deltaY, deltaX);  angle_deg = angle * (180 / PI);


//Extend the canvas to avoid cutting
getDimensions(width, height, channels, slices, frames);
diagonal = sqrt(width * width + height * height);
newSize = round(diagonal) + 50;  // give more space

run("Canvas Size...", "width=" + newSize + " height=" + newSize + " position=Center");
// Rotate the entire image
run("Rotate...", "angle=" + (-angle_deg) + " grid=0 interpolation=Bilinear");
}


function findTip() { 
open(imagepath + "blue"+".tif");
roiManager("select",1);  // choose the base_midpoint roi
getSelectionCoordinates(base_midx, base_midy);   
close();

//get Blur outline
open(imagepath + "blue"+".tif");  // use the mask of bule part saved before
run("Gaussian Blur...", "sigma=25"); 
run("8-bit");
setAutoThreshold("Default");
run("Make Binary");
run("Invert");      
run("Outline");                         
run("Create Selection");
AddToRoi(6,"blur_outline");
getSelectionCoordinates(xpoints, ypoints); // Although this cannot get not-edge coordinates of straight part of the outline,
										  //mathematically, Distance to one edge point is always the largest.							  
// Nucellus outline point with the biggest Distance to the base line middle point is the tip of nucellus. 
maxDistance = 0;
maxIndex = -1;

refX = base_midx[0] ;  // refernt porint is base_midpoint
refY = base_midy[0] ;

for (i = 0; i < xpoints.length; i++) {      // loop each outline coordinates to find the biggest distance
	delX = xpoints[i] - refX;
	delY = ypoints[i] - refY;
    distance = delX*delX + delY*delY ;
     
    if (distance > maxDistance) {
        maxDistance = distance;
        maxIndex = i;
    }
}


roiManager("select",6);  // remove the blur_outline
roiManager("delete");
tipx = xpoints[maxIndex];
tipy = ypoints[maxIndex];


	open(imagepath + "blue_outline.tif");
	if (getPixel(tipx, tipy) == 0) {     // find its nearest outline points to the tip if the tip is not on outline
	refx = tipx;
	refy = tipy;
	minIndex = makeItonOutline();
	tipx = outlineArray[minIndex];
	tipy = outlineArray[minIndex+1];
	}	
	


makePoint(tipx, tipy);
roiManager("Add", "point");
roiManager("select",6);
roiManager("rename", "tip"); 

}



function getOutlineArray() {
	//make the outline
	open(imagepath + "blue"+".tif");
	setAutoThreshold("Default");
	run("Make Binary");
	run("Invert");
	run("Outline");  
	saveAs("Tiff", imagepath + "blue_outline.tif"); 
	
	//records all points on the outline
	getDimensions(width, height, channels, slices, frames); 
	outlineArray = newArray();
	for (y = 0; y < height; y++) {
	    for (x = 0; x < width; x++) {     // looping every coordinates
	
	        v = getPixel(x, y);       // get intensity of this coordinate
	        if (v == 255) {         // only overlap area of outlines has intensity = 255
	        	outlineArray = Array.concat(outlineArray, newArray(x,y));      // store these overlap points
	        }
	    }
	}
	return outlineArray;
}


function cleanOutline() {
	// get the baseline edge points
	roiManager("Select", 2);
	getSelectionCoordinates(x1, y1);
	roiManager("Select", 3);
	getSelectionCoordinates(x2, y2);
	// use edge points to draw a drak baseline in width 2 for good split later
	setColor(0);
	setLineWidth(2);
	drawLine(x1[0], y1[0], x2[0], y2[0]);

	// split and get the roi index of the biggest area part
	run("Create Selection");
	roiManager("Split"); 
	
	nROIs = roiManager("count");  	MaxArea = 0;
	for (i = 10; i < nROIs; i++) {
    roiManager("Select", i);
    run("Measure"); 
    Area = getResult("Area", i-10);
    if (Area>MaxArea) 
    {MaxArea = Area;  index = i;}}
	
	// inerse selection and delete the rest small parts
	roiManager("Select", index);
	run("Make Inverse");
	setBackgroundColor(0, 0, 0);
	run("Clear", "slice");
	
	// redraw the baseline in width 2 but with 255 intensity
	setColor(255);
	setLineWidth(2);
	drawLine(x1[0], y1[0], x2[0], y2[0]);
	run("Fill Holes");
	
	// use the previous outline to delete extra pixels of the drawn line
	roiManager("Select", 5);
	setBackgroundColor(0, 0, 0);
	run("Clear", "slice");
	// draw the baseline in width 1 with 255 intensity and fill the holes left by last step
	setColor(255);
	setLineWidth(1);
	drawLine(x1[0], y1[0], x2[0], y2[0]);
	run("Fill Holes");
	// set the baseline to dark
	setColor(0);
	setLineWidth(1);
	drawLine(x1[0], y1[0], x2[0], y2[0]);
	roiManager("reset");
	// get and save the cleaned_outline
	run("Make Binary");
	run("Outline");
	saveAs("Tiff", imagepath+ "cleaned_outline.tif"); 
}


function makeItonOutline() {
	//get the outline point with the smallest distance to the tip
	minDistance = width*width + height*height;
	minIndex = -1;

	for (i = 0; i < outlineArray.length; i++) {     
		delX = outlineArray[i] - refx;
		delY = outlineArray[i+1] - refy;
	    distance = delX*delX + delY*delY ;
	    
	    if (distance < minDistance) {
	        minDistance = distance;
	        minIndex = i;
	    }
	    i = i+1;
	}
	return minIndex;
}


function printxy(x,y) { 
for (i = 0; i < x.length; i++) {
    print("(" + x[i] + ", " + y[i] + ")");
}
}

function closeimages() { 
while (nImages > 0) {
    selectImage(nImages);
    close();
}
}


function AddToRoi(index, name){
	run("ROI Manager...");
	roiManager("Add");
	roiManager("Select", index);
	roiManager("Rename", name);
	}


function makeOutline(outlineindex,Name) { 
	run("8-bit");
	setAutoThreshold("Default");
	run("Make Binary");
	run("Invert");
	run("Fill Holes");                        // something inside of tiusse has holes
	
		
	getDimensions(width, height, channels, slices, frames);
	for (y = 0; y < height; y++) {
	    for (x = 0; x < width; x++) {     // looping every coordinates
	    	v = getPixel(x, y);
	        	
	        if (v == 255 && getPixel(x+1, y) == 0 && getPixel(x-1, y) == 0 
	        	&& getPixel(x, y+1) == 0 &&  getPixel(x, y-1) == 0 ) {
	        	setPixel(x, y, 0); } // remove the independent point to aviod hole on outline
	    }
	}
	
	saveAs("Tiff", imagepath + Name);         // before become outline, the mask is saved in local for many use later
	run("Outline");                           // the system has a window with the outline images (only outline pixcels = 255)
	run("Create Selection");
	AddToRoi(outlineindex, Name+"_outline");
}



function run2DBendingAnalysisModule() {   
	roiManager("Select", 1);
	getSelectionCoordinates(xmid, ymid);
	roiManager("Select", 2);
	getSelectionCoordinates(x1, y1);  x1 = x1[0]; y1 = y1[0];  // baseline start
	roiManager("Select", 3);
	getSelectionCoordinates(x2, y2);  x2 = x2[0]; y2 = y2[0];  // baseline end
	roiManager("Select", 6);
	getSelectionCoordinates(centroidx, centroidy);  x0 = centroidx[0]; y0 = centroidy[0];  // centroid point
	roiManager("Select", newArray(2,3,4,5,7));
	roiManager("delete");


	//Calculate to get a vertical coordinate of centroid on baseline and draw it
	dx = x2 - x1;
	dy = y2 - y1;
	lengthSquared = dx * dx + dy * dy;
	t = ((x0 - x1) * dx + (y0 - y1) * dy) / lengthSquared;
	
	xf = x1 + t * dx; 
	yf = y1 + t * dy;  

	makeLine(x0, y0, xf, yf);
	AddToRoi(3, "centroid_line");
	
	// Calculate the vertical line length
	perp_dx = xf - x0;                
	perp_dy = yf - y0;                

	// vertical line pass middle points with the same length; make sure the end point close to centroid
	startX = xmid[0]; 
	startY = ymid[0];

	endX1 = startX + perp_dx;
	endY1 = startY + perp_dy;
	distance1 = (endX1-x0)*(endX1-x0) + (endY1-y0)* (endY1-y0);
	
	endX2 = startX - perp_dx;
	endY2 = startY - perp_dy;
	distance2 = (endX2-x0)*(endX2-x0) + (endY2-y0)* (endY2-y0);
	
	endX = endX1; endY = endY1;
	if (distance1 > distance2) {endX = endX2; endY = endY2;}

	makeLine(startX, startY, endX, endY);  
	AddToRoi(4, "middlepoint_line");

	makeLine(x0, y0, endX, endY);
	run("Measure"); 
	distance = getResult("Length", 0);
	AddToRoi(5, "distance="+distance);

////save the roiset
roiManager("save",imagepath+imagetitle+"_Bending_Analysis_roi.zip");
//roiManager("Show All");  
//roiManager("select",5);
//saveAs("Tiff", imagepath+ imagetitle + "_distance="+distance);
return distance
}  




function run2DDifferentialGrowthAnalysisModule() {      
// split nucellus based on the drawn line: tip -> centroid -> middle point of baselines
setAutoThreshold("Default dark no-reset");
run("Create Selection");
roiManager("Split");                 
//rename the rois of two splited part and get the area
roiManager("select",0);
run("Measure");    
posterior_nucellus_area = getResult("Area", 0);    
roiManager("rename", "posterior_nucellus_area="+posterior_nucellus_area);

roiManager("select",1);
run("Measure");    
anterior_nucellus_area = getResult("Area", 1);    
roiManager("rename", "anterior_nucellus_area="+anterior_nucellus_area);


roiManager("select",2);
roiManager("rename", "total_area");

//get the ration and save the image
pos_ant_ratio = posterior_nucellus_area / anterior_nucellus_area  ;        

roiManager("save",imagepath+imagetitle+"_Differential_Growth_Analysis_roi.zip");
//roiManager("Show All");  
//roiManager("select",1);
//saveAs("Tiff", imagepath+ imagetitle + "_pos_ant_ratio=" + posterior_nucellus_area +","+ anterior_nucellus_area+","+pos_ant_ratio);
return newArray(posterior_nucellus_area, anterior_nucellus_area, pos_ant_ratio);
}       

