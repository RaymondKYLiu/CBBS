### Implementation of Codebook background subtraction

Kyungnam Kima, Thanarat H. Chalidabhongse, David Harwood, Larry Davis. Real-time foreground–background segmentation
using codebook model.

http://www.umiacs.umd.edu/~knkim/paper/Kim-RTI2005-FinalPublished.pdf

### Usage

```cpp
// OpenCV example:
int main()
{
    CBModel model(50, COLOR_TYPE_BGR, 0.5, 1.5, 10, 10, true);
    model.setCBUpdateParam(300, 300);

    VideoCapture cap(0);
    int w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    Mat colorImg;
    Mat blured;
    Mat mask;

    mask.create(Size(w, h), CV_8U);

    while(1)
    {
        cap >> colorImg;

        GaussianBlur(colorImg, blured, cv::Size(7,7), 1.5,  1.5);

        if (model.process(blured.data, w, h, mask.data) != CODEBOOK_FLAG_DETECT)
            continue;
        imshow("Mask", mask);

        noiseRemove(mask.data, w, h, 3);
        imshow("Mask_noise_remove", mask);

        medianFilterBinary(mask.data, w, h, 15);
        imshow("Mask_medin", mask);

        imshow("Frame", colorImg);

        if(waitKey(10) == 27)
        break;
    }

    releaseCBModel(model);
    return 0;
}
```
