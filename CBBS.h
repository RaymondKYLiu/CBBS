#ifndef __CBBS_H__
#define __CBBS_H__

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//#define  DEBUG_MODE
#ifdef DEBUG_MODE
#include <stdio.h>
#include <iostream>
#include <iomanip>
#endif

#ifndef MAX
#define MAX(a,b) ((a) >= (b) ? (a) : (b))
#endif // MAX

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif // MIN

typedef unsigned char uchar;

class CodeWord      // pixel-based codeword object
{
public:
    enum CodeWordType { BG = 0, Cache = 1 };

    CodeWord();
    CodeWord(uchar* p, int T, int type = BG);

    ~CodeWord() {};
    CodeWord(const CodeWord& cw);
    CodeWord& operator= (const CodeWord& cw);
    void assign(CodeWord& cw);
    void update(uchar* p, int T, bool is_train_state);
    float score();

    float       m_colors[3];            // means of (R,G,B)
    int         m_frequency;            // frequency with which the codeword has occurred
    int         m_stale;                // maximum negative run-length (MNRL)
    int         m_first_update;
    int         m_last_update;          // ﬁrst and last access times,

    bool        m_is_perm;        // robust codeword permanentlly
    bool        m_is_valid;       // invalid codework not be considered
private:

};

class  CBModel
{
public:
    //************************************
    // nFrameTrain : number of frame to train
    // e1 : color threshold for training process
    // e2 : color threshold for detection process
    // update : Update model?
    // shadowRm : Remove shadow ?
    //************************************
    enum ColorType {
        COLOR_TYPE_BGR,
        COLOR_TYPE_RGB
    };

    CBModel(unsigned int num_frame_train, ColorType color_type = COLOR_TYPE_BGR, float e1 = 7.5,
            float e2 = 15,
            bool need_update = true, bool shadow_remove = false);

    CBModel() {};
    ~CBModel();


    enum {
        CODEBOOK_FLAG_TRAIN,
        CODEBOOK_FLAG_DETECT
    };


    //************************************
    // Process the Codebook background substraction model and return CodeBookFlags .
    // pColorImg: #input color(BGR) image buffer
    // pUpdateMap: #The map to determine that this pixel to be update or not
    // pFGMask: #output foreground mask buffer
    // pColorBGImg: #output background color image
    //
    //************************************
    int process(uchar* pColorImg, int width, int height, uchar* pUpdateMap, uchar* pFGMask,
                uchar* pColorBGImg = NULL);

    //************************************
    // Set update paramater.
    // updatePeriod: the period frames for update
    // timeAdd: Time for adding static object to background.
    // timeDel: Time for deleting ghost from background
    //************************************
    void setUpdateParam(int updatePeriod, int timeAdd, int timeDel);

    //************************************
    // Set shadow removal paramater in HSV color space.
    // alpha: lower bound ratio of V
    // beta: upper bound ratio of V
    // tau_h: threshold of H
    // tau_s: threshold of S
    //************************************
    void setShadowRmParam(float alpha, float beta, float tau_h, float tau_s);

    void noiseRemove(uchar* mask, int width, int height, int remove_thresh = 3);

    void medianFilterBinary(uchar* mask, int width, int height, int win_sz = 11);

private:
    ColorType   m_color_type;

    unsigned int    m_curr_frame;
    int         m_depth;

    int         m_rows;
    int         m_cols;

    int         m_num_train;
    float       m_e1, m_e2;

    bool        m_shadow_remove;
    float       m_alpha, m_beta;
    float       m_tau_h, m_tau_s;

    bool        m_need_update;
    int         m_T_update_period;

    int         m_T_add;
    int         m_T_maxstale;
    int         m_T_maxdel;
    int         m_T_delfreq;

    uchar*       m_bg;
    typedef struct {
        CodeWord* bg;
        CodeWord* cache;
    } CodeBook;

    CodeBook* m_cb;
    CodeWord* m_cw_map;

    void initialize(int width, int height);
    void trainBG(uchar* pColorImg);
    void clearBG(int stale_thresh);
    void detectFG(uchar* pColorImg, uchar* pMask, uchar* pUpdateMap);
    void getBG(uchar* pBGImg);
    int clearCache(CodeBook* cb, int staleThres);
    int addToCodeBook(CodeBook* cb,  int T, int add_thresh);
    int deleteFromCodeBook(CodeBook* cb, int T, int del_thresh);
    void shadowRemove(uchar* pColorImg,  uchar* pFGMask, uchar* pColorBGImg);
};

#endif


/*  OpenCV example:
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
*/