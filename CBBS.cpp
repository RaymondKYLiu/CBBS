#include "CBBS.h"


CodeWord::CodeWord()
{
	m_colors[0]=m_colors[1]=m_colors[2]=0;
	m_frequency = 0;
	m_stale = 0;
	m_first_update = 0;
	m_last_update = 0;
	m_is_perm = false;
	m_is_valid = false;
}

CodeWord::CodeWord( const CodeWord& cw )
{
	operator=(cw);
}

CodeWord::CodeWord( uchar* p, int T, int type)
{
	m_colors[0] = (float)p[0];
	m_colors[1] = (float)p[1];
	m_colors[2] = (float)p[2];
	m_frequency = 1;
	m_first_update = T;
	m_last_update = T;
	m_is_valid = true;

	if(type == Cache)
	{
		m_stale = 1;
		m_is_perm = false;
	}
	else
	{
		m_stale = T-1;
		m_is_perm = true;
	}
	
}

CodeWord& CodeWord::operator=( const CodeWord& cw )
{
	if(this == &cw)
		return *this;

	this->m_colors[0] = cw.m_colors[0]; this->m_colors[1] = cw.m_colors[1]; this->m_colors[2] = cw.m_colors[2];
	this->m_frequency = cw.m_frequency;
	this->m_stale = cw.m_stale;
	this->m_first_update = cw.m_first_update;
	this->m_last_update = cw.m_last_update;
	this->m_is_perm = cw.m_is_perm;
	this->m_is_valid = cw.m_is_valid;

	return *this;
}


void CodeWord::assign( CodeWord& cw )
{
	if(this == &cw)
		return;

	(this->m_colors)[0] = cw.m_colors[0]; (this->m_colors)[1] = cw.m_colors[1]; (this->m_colors)[2] = cw.m_colors[2];
	this->m_frequency = cw.m_frequency;
	this->m_stale = cw.m_stale;
	this->m_first_update = cw.m_first_update;
	this->m_last_update = cw.m_last_update;
	this->m_is_perm = cw.m_is_perm;
	this->m_is_valid = cw.m_is_valid;

}


void CodeWord::update( uchar* p, int T , bool is_train_state )
{
	float alpha = 0.01f;
	float p0=p[0], p1=p[1], p2=p[2]; 
	if(is_train_state) 
	{
		m_colors[0] = (m_frequency*m_colors[0]+p0)/(m_frequency + 1);
		m_colors[1] = (m_frequency*m_colors[1]+p1)/(m_frequency + 1);
		m_colors[2] = (m_frequency*m_colors[2]+p2)/(m_frequency + 1);
	}
	else
	{
		m_colors[0] = (1-alpha)*m_colors[0] + alpha*p0;
		m_colors[1] = (1-alpha)*m_colors[1] + alpha*p1;
		m_colors[2] = (1-alpha)*m_colors[2] + alpha*p2;
	}

	m_frequency++;
	m_stale = MAX(m_stale, T - m_last_update);
	m_last_update = T;
}

float CodeWord::score()
{
	return (float)(m_last_update - m_first_update) + m_frequency + (m_is_perm ? 100 : 0);
}

CBModel::CBModel(unsigned int num_frame_train, ColorType color_typet, float e1/*=7.5*/, float e2/*=15*/,
			     bool need_update/*=true*/, bool shadow_remove/*=true*/ )
{
	this->m_color_type = color_typet;

	this->m_curr_frame = 0;
	this->m_num_train = num_frame_train;
	this->m_depth = 3;
	this->m_rows = this->m_cols = 0;
	this->m_cb = NULL;
	
	this->m_e1 = e1;
	this->m_e2 = e2;
	
	this->m_shadow_remove = shadow_remove;
	this->m_alpha = 0.65f;
	this->m_beta = 1.15f;
	this->m_tau_h = 40.f;
	this->m_tau_s = 60.f;

	this->m_need_update = need_update;
	this->m_T_update_period = 5;

	this->m_T_add = 100;
	this->m_T_maxstale = 50;
	this->m_T_delfreq = 200;
	this->m_T_maxdel = 200;

	/*unsigned int	t;
	int			num_bins;

	int			rows;
	int			cols;

	int			num_train;
	float		e1, e2;

	bool		isShadowRm;
	float		alpha, beta;
	float		tau_h, tau_s;

	bool		isUpdate;
	int			T_updatePeriod;

	int			T_add;
	int			T_maxm_stale;
	int			T_maxdel;
	int			T_delfreq;*/
}

void CBModel::initialize( int width, int height )
{
	m_cols = width;
	m_rows = height;
	m_cb = new CodeBook[m_rows*m_cols];
	m_cw_map = new CodeWord[2*m_cols*m_rows*m_depth];
	CodeWord* p = m_cw_map;
	m_bg = (uchar*)malloc(sizeof(uchar)*width*height*3);
	// assign pointer
	for (int i = 0; i < m_cols*m_rows; i++)
	{
		m_cb[i].bg = p;
		p += m_depth;

		m_cb[i].cache = p;
		p += m_depth;
	}

// 	for (int i = 0; i < m_cols*m_rows; i++)
// 	{
// 		m_cb[i].bg = new CodeWord[m_depth];
// 		m_cb[i].cache = new CodeWord[m_depth];
// 	}
}

CBModel::~CBModel()
{
	if(m_cb) 
	{
		delete [] m_cw_map;
		delete [] m_cb;
	}
	if (m_bg)
		delete []m_bg;

	m_cw_map = NULL;
	m_cb = NULL;
}

int CBModel::process( uchar* pColorImg, int width, int height, uchar* pUpdateMap, uchar* pFGMask, uchar* pColorBGImg )
{	
	if(!m_cb)
		initialize(width, height);

	if (pFGMask)
		memset(pFGMask, 0, sizeof(uchar)*width*height);
	if(pColorBGImg)
		memset(pColorBGImg, 0, sizeof(uchar)*width*height*3);

	if (m_curr_frame < m_num_train)
	{
		trainBG(pColorImg/*, mask*/);
		return CODEBOOK_FLAG_TRAIN;
	}
	else if (m_curr_frame == m_num_train)
	{
		clearBG(m_num_train/2);
		return CODEBOOK_FLAG_DETECT;
		//return CODEBOOK_FLAG_TRAIN;
	}
	else
	{
		detectFG(pColorImg, pFGMask, pUpdateMap);
		
		if(m_shadowRm || pColorBGImg) {
			//uchar* bg = (uchar*)malloc(sizeof(uchar)*width*height*3);			
			getBG(m_bg);
			if(m_shadowRm)
				shadowRemove(pColorImg, pFGMask, m_bg);
			if(pColorBGImg)
				memcpy(pColorBGImg, m_bg, sizeof(uchar)*width*height*3);
			//free(bg);
		}
				
		return CODEBOOK_FLAG_DETECT;
	}

}

void CBModel::trainBG( uchar* pColorImg )
{
	int img_step = m_cols*3;
	int T = ++(m_curr_frame);
	int x, y, i;
	int num_add = 0;
	for (y = 0; y < m_rows; y++)
	{
		uchar* p = pColorImg + y*img_step;
		CodeBook* cb = m_cb + y*m_cols;
		for (x = 0; x < m_cols; x++, p+=3, cb++)
		{
			float p0=p[0], p1=p[1], p2=p[2]; 
			int negRun = 0;
			bool isFound = false;
			CodeWord* cw = cb->bg;
			for (i = 0; i < m_depth; i++)
			{
				if(!cw[i].m_is_valid) continue;

				if(abs(cw[i].m_colors[0] - p0) <= m_e1 && abs(cw[i].m_colors[1] - p1) <= m_e1 && abs(cw[i].m_colors[2] - p2) <= m_e1)
				{
					cw[i].update(p, T, true);
					isFound = true;
					break;
				}

				negRun = T - cw[i].m_last_update;
				cw[i].m_stale = MAX(cw[i].m_stale, negRun);
			}
			for (; i < m_depth; i++)
			{
				if(!cw[i].m_is_valid) continue;
				negRun = T - cw[i].m_last_update;
				cw[i].m_stale = MAX(cw[i].m_stale, negRun);
			}
			
			// New codeword
			if (!isFound)
			{
				CodeWord cw_new(p, T, CodeWord::BG);	// a new codeword will be added.
				float min_score = FLT_MAX; int idx;
				for (i = 0; i < m_depth; i++)	// search where to add
				{
					if(!cw[i].m_is_valid) { 
						idx = i ; 
						break;
					}
					if(cw[i].score() < min_score) {
						min_score = cw[i].score(); 
						idx = i;
					}
				}
				cw[idx] = cw_new;
				//cw[idx].assign(cw_new);
				num_add++;
			}
		}	// end for x
	}	// end for y
#ifdef DEBUG_MODE
	printf("Frame:%d creat %d\n", T, num_add);
#endif
	
}

void CBModel::clearBG( int m_stale_thresh )
{
	int num_clear = 0;
	int T = ++(m_curr_frame);
	int x, y, i;

	for (y = 0; y < m_rows; y++)
	{
		CodeBook* cb = m_cb + y*m_cols;
		for (x = 0; x < m_cols; x++, cb++)
		{
			CodeWord* cw = cb->bg;
			for (i = 0; i < m_depth; i++)
			{
				if(!cw[i].m_is_valid) continue;
				if(cw[i].m_stale > m_stale_thresh)
				{
					cw[i].m_is_valid = false;
					num_clear++;
				}
			}
		}	// end for x
	}	// end for y

#ifdef DEBUG_MODE
	printf("Frame:%d free... %d\n", T, num_clear);
#endif
}

void CBModel::detectFG( uchar* pColorImg, uchar* pMask, uchar* pUpdateMap )
{
	int T = ++(m_curr_frame);
	int x, y, i;
	int num_clear_cache=0;
	int num_new=0;
	int num_add = 0;
	int num_del = 0;
	bool T_update = T % m_T_update_period;
	int img_step = m_cols*3;
	uchar* p, *pp, *ppp = NULL;

	for (y = 0; y < m_rows; y++)
	{
		p = pColorImg + y*img_step;
		pp = pMask + y*m_cols;
		if(pUpdateMap) ppp = pUpdateMap + y*m_cols;
		CodeBook* cb = m_cb + y*m_cols;
		
		for (x = 0; x < m_cols; x++, p+=3, pp++, ppp++, cb++)
		{
			float p0=p[0], p1=p[1], p2=p[2], d=0; 
			int negRun = 0;
			bool isFound = false;
			CodeWord* cw = cb->bg;
			CodeWord* ca = cb->cache;

			for (i = 0; i < m_depth; i++)
			{
				if(!cw[i].m_is_valid) continue;
				d = MAX(abs(cw[i].m_colors[0]-p0), abs(cw[i].m_colors[1]-p1));
				d = MAX(d, abs(cw[i].m_colors[2]-p2));
				if (d <= m_e2)	// find match background codeword
				{
					cw[i].update(p, T, false);
					pp[0]=0;

					isFound = true;
					break;
				}
			}

			if (!isFound)
				pp[0]=255;
			if(!m_need_update || (T_update) != 0)
				continue;
/************************************************************************/
/*							Update                                      */
/************************************************************************/
			if (!isFound)	// foreground detected. Then find match from cache.
			{
				if( pUpdateMap)
					if(ppp[0])	// the pixel of moving object is not to updated.
						continue;

				bool is_found_cache=false;	
				
				for (i = 0; i < m_depth; i++)
				{
					if(!ca[i].m_is_valid) continue;
					if (abs(ca[i].m_colors[0] - p0) <= m_e1 && abs(ca[i].m_colors[1] - p1) <= m_e1 && abs(ca[i].m_colors[2] - p2) <= m_e1)
					{
						ca[i].update(p, T, false);
						is_found_cache = true;
						break;
					}
					negRun = T - ca[i].m_last_update;
					ca[i].m_stale = MAX(ca[i].m_stale, negRun);
				}
				for (; i < m_depth; i++)
				{
					if(!ca[i].m_is_valid) continue;
					negRun = T - ca[i].m_last_update;
					ca[i].m_stale = MAX(ca[i].m_stale, negRun);
				}

				if (!is_found_cache)
				{
					CodeWord cw_new(p, T, CodeWord::Cache);	// a new codeword will be added.
					float min_score = FLT_MAX; int idx;
					for (i = 0; i < m_depth; i++)	// search where to add
					{
						if(!ca[i].m_is_valid) { 
							idx = i ; 
							break;
						}
						if(ca[i].score() < min_score) {
							min_score = ca[i].score(); 
							idx = i;
						}
					}
					ca[idx] = cw_new;
					num_new++;
				}
			}
			else	// update all cache m_stale
			{
				for (i = 0; i < m_depth; i++)
				{
					negRun = T - ca[i].m_last_update;
					ca[i].m_stale = MAX(ca[i].m_stale, negRun);
				}
			}
 			num_clear_cache += clearCache(cb, m_T_maxstale);
 			num_add += addToCodeBook(cb, T, m_T_add);
 			//if((T % m_T_delfreq) == 0)
 				num_del += deleteFromCodeBook(cb, T, m_T_maxdel);

		}	// end for x
	}	// end for y

#ifdef DEBUG_MODE
	std::cout << "Frame:" << std::setw(5) << T << 
		" new: " << std::setw(5) << num_new <<
		" clear: " << std::setw(5) << num_clear_cache << 
		" add: " << std::setw(5) << num_add << 
		" delete: " << std::setw(5) << num_del <<"\n";
#endif

}

void CBModel::getBG( uchar* pBGImg )
{
	if (!pBGImg)
		return;

	int x, y, k;
	int img_step = m_cols*3;

//	for ( y = 0; y < m_rows; y++)
//	{
//		uchar* p = pBGImg + y*img_step;
//		CodeBook* cb = m_cb + y*m_cols;
		//for ( x = 0; x < m_cols; x++, p+=3, cb++)
	//	{
	CodeBook* cb = m_cb;
	uchar* p = pBGImg;
	memset(pBGImg, 0, m_rows*m_cols);
	for (k = 0; k < m_rows*m_cols; k++, cb++, p+=3)
	{
			CodeWord *cw = cb->bg;
			CodeWord* cw_max_freq = NULL;
			int max_freq = 0;

			for (int i = 0; i < m_depth; i++)
			{
				if(!cw[i].m_is_valid) continue;

				if(cw[i].m_frequency > max_freq)
				{
					max_freq = cw[i].m_frequency;
					cw_max_freq = &cw[i];
				}
			}

			if(cw_max_freq)
			{
				p[0] = (uchar)cw_max_freq->m_colors[0];
				p[1] = (uchar)cw_max_freq->m_colors[1];
				p[2] = (uchar)cw_max_freq->m_colors[2];
			}
	//	}
	//}
	}
}


int CBModel::addToCodeBook( CodeBook* cb, int T, int adding_thresh )
{
	int num_add = 0;
	CodeWord* cw = cb->bg;
	CodeWord* ca = cb->cache;

	CodeWord tmp;
	
	for (int i = 0; i < m_depth; i++)
	{
		if(!ca[i].m_is_valid) continue;

		if(T - ca[i].m_first_update > adding_thresh && T - ca[i].m_last_update < adding_thresh/10)
		{
			// find BG position to add
			int idx; float min_score = FLT_MAX;
			for (int j = 0; j < m_depth; j++)	// search where to add
			{
				if(!cw[j].m_is_valid) { 
					idx = j ; 
					break;
				}
				if(cw[j].score() < min_score) {
					min_score = cw[j].score(); 
					idx = j;
				}
			}
			cw[idx] = ca[i];
			ca[i].m_is_valid = false;
			num_add++;
			break;
		}
	}

	return num_add;
}

int CBModel::deleteFromCodeBook( CodeBook* cb, int T, int del_thresh )
{
	int num_del=0;
	CodeWord* cw = cb->bg;

	for (int i = 0; i < m_depth; i++)
	{
		if(!cw[i].m_is_valid) continue;

		if (T - cw[i].m_last_update > (cw[i].m_is_perm ? 2*del_thresh : del_thresh) )
		{
			cw[i].m_is_valid = false;
			num_del++;
		}
		else
		{
			if(T - cw[i].m_first_update > del_thresh*2)
				cw[i].m_is_perm = true;
		}
	}

	return num_del;
}

int CBModel::clearCache( CodeBook* cb, int m_stale_thres )
{
	int num_clear = 0;
	CodeWord* ca = cb->cache;

	for (int i = 0; i < m_depth; i++)
	{	
		if(!ca[i].m_is_valid) continue;

		if(ca[i].m_stale > m_stale_thres)
		{
			ca[i].m_is_valid = false;
			num_clear++;
		}
	}
	return num_clear;
}


static void cvtPixRGB2HSV( uchar R, uchar G, uchar B, uchar &H, uchar &S, uchar &V )
{
	float fR, fG, fB;
	float fH, fS, fV;
	const float FLOAT_TO_BYTE = 255.0f;
	const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
	int bB = B; //*(uchar*)(pRGB+0);	// Blue component
	int bG = G; //*(uchar*)(pRGB+1);	// Green component
	int bR = R; //*(uchar*)(pRGB+2);	// Red component

	// Convert from 8-bit integers to floats.
	fR = bR * BYTE_TO_FLOAT;
	fG = bG * BYTE_TO_FLOAT;
	fB = bB * BYTE_TO_FLOAT;

	// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
	float fDelta;
	float fMin, fMax;
	int iMax;
	// Get the min and max, but use integer comparisons for slight speedup.
	if (bB < bG) {
		if (bB < bR) {
			fMin = fB;
			if (bR > bG) {
				iMax = bR;
				fMax = fR;
			}
			else {
				iMax = bG;
				fMax = fG;
			}
		}
		else {
			fMin = fR;
			fMax = fG;
			iMax = bG;
		}
	}
	else {
		if (bG < bR) {
			fMin = fG;
			if (bB > bR) {
				fMax = fB;
				iMax = bB;
			}
			else {
				fMax = fR;
				iMax = bR;
			}
		}
		else {
			fMin = fR;
			fMax = fB;
			iMax = bB;
		}
	}
	fDelta = fMax - fMin;
	fV = fMax;				// Value (Brightness).
	if (iMax != 0) {			// Make sure its not pure black.
		fS = fDelta / fMax;		// Saturation.
		float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta);	// Make the Hues between 0.0 to 1.0 instead of 6.0
		if (iMax == bR) {		// between yellow and magenta.
			fH = (fG - fB) * ANGLE_TO_UNIT;
		}
		else if (iMax == bG) {		// between cyan and yellow.
			fH = (2.0f/6.0f) + ( fB - fR ) * ANGLE_TO_UNIT;
		}
		else {				// between magenta and cyan.
			fH = (4.0f/6.0f) + ( fR - fG ) * ANGLE_TO_UNIT;
		}
		// Wrap outlier Hues around the circle.
		if (fH < 0.0f)
			fH += 1.0f;
		if (fH >= 1.0f)
			fH -= 1.0f;
	}
	else {
		// color is pure Black.
		fS = 0;
		fH = 0;	// undefined hue
	}

	// Convert from floats to 8-bit integers.
	int bH = (int)(0.5f + fH * 255.0f);
	int bS = (int)(0.5f + fS * 255.0f);
	int bV = (int)(0.5f + fV * 255.0f);

	// Clip the values to make sure it fits within the 8bits.
	if (bH > 255)
		bH = 255;
	if (bH < 0)
		bH = 0;
	if (bS > 255)
		bS = 255;
	if (bS < 0)
		bS = 0;
	if (bV > 255)
		bV = 255;
	if (bV < 0)
		bV = 0;

	// Set the HSV pixel components.
	H = bH;		// H component
	S = bS;		// S component
	V = bV;		// V component
}


void CBModel::shadowRemove( uchar* pColorImg, uchar* pFGMask, uchar* pColorBGImg )
{
	uchar r_i, g_i, b_i;
	uchar r_b, g_b, b_b;
	uchar h_i, s_i, v_i;
	uchar h_b, s_b, v_b;
	int r_pos = 0, g_pos = 1, b_pos = 2; // (m_color_type == COLOR_TYPE_RGB)

	float h_diff, s_diff, v_ratio;

	int numPixels = m_cols*m_rows;
	int i, pos;

	if(m_color_type == COLOR_TYPE_BGR)
	{
		r_pos = 2; g_pos = 1; b_pos = 0;
	}

	for (i = 0; i < numPixels; i++)
	{
		if(!pFGMask[i]) continue;
		pos = i*3;
		b_i = pColorImg[pos + b_pos];
		g_i = pColorImg[pos + g_pos];
		r_i = pColorImg[pos + r_pos];
		b_b = pColorBGImg[pos + b_pos];
		g_b = pColorBGImg[pos + g_pos];
		r_b = pColorBGImg[pos + r_pos];

		cvtPixRGB2HSV(r_i, g_i, b_i, h_i, s_i, v_i);
		cvtPixRGB2HSV(r_b, g_b, b_b, h_b, s_b, v_b);

		v_ratio = (float)v_i / (float)v_b;
		s_diff = abs((int)s_i - (int)s_b);
		h_diff = MIN( abs((int)h_i - (int)h_b), 255 - abs((int)h_i - (int)h_b));

		if(	h_diff <= m_tau_h &&
			s_diff <= m_tau_s &&
			v_ratio >= m_alpha &&
			v_ratio < m_beta)
		{
			pFGMask[i] = 0;	// Shadow value :128
		}
	}
}

void CBModel::setUpdateParam( int update_period, int time_to_add, int time_to_del )
{
	m_T_update_period = update_period;
	m_T_add = time_to_add;
	m_T_maxstale = time_to_add/2;
	m_T_delfreq = time_to_del;
	m_T_maxdel = time_to_del;
}

void CBModel::setShadowRmParam( float alpha, float beta, float tau_h, float tau_s )
{
	m_alpha = alpha;
	m_beta = beta;
	m_tau_h = tau_h;
	m_tau_s = tau_s;
}


void CBBS_APInoiseRemove(uchar* mask, int width, int height, int remove_thresh)
{
	int sz = width*height;
	uchar* tmp = (uchar*)malloc(sizeof(uchar)*sz);


	remove_thresh = MAX(MIN(remove_thresh, 8), 0);
	int img_step = width;

	bool isok;

	for (int i = 0; i < 1; i++)
	{
		memcpy(tmp, mask, sz);
		isok = true;
		for (int y = 1; y < height - 1; y++)
		{
			uchar* p = mask + y*img_step + 1;
			uchar* pp = tmp + y*img_step + 1 ;

			for (int x = 1; x < width - 1; x++, p++, pp++)
			{
				if(*pp == 255)
				{
					int cnt = (int) pp[-1-img_step] + pp[-img_step] + pp[1-img_step] +
						pp[-1] +						 + pp[1] +
						pp[-1+img_step] + pp[+img_step]	 + pp[1+img_step];
					cnt /= 255;
					if (cnt < remove_thresh)
					{
						*p = 0;
						isok = false;
					}

				}

			}
		}
	} 
	free(tmp);
}

static void myIntergral(const uchar* src, int* sum, int width, int height )
{
	//* src : height by width 
	//* sum : height+1 by width+1 (padding zeros on top row and left column)
	

	int sum_widthStep = width + 1;
	const uchar* src_ptr = src ;
	int* sum_ptr = sum;
	memset(sum, 0, (width+1)*(height+1)*sizeof(int));
	
 	sum_ptr = sum + sum_widthStep +1; 
	
 	for (int y = 0 ; y < height ; y++, sum_ptr++)
 		for (int x = 0 ; x < width ; x++, src_ptr++, sum_ptr++)
		{
			sum_ptr[0] = (int)src_ptr[0] 
						+ sum_ptr[-1] 
						+ sum_ptr[-sum_widthStep] 
						- sum_ptr[-sum_widthStep-1];	
		}
}

void medianFilterBinary(uchar* mask, int width, int height, int win_sz)
{
	int offset = win_sz >> 1;
	int num = win_sz*win_sz;
	int th = (num+1)/2 * 255;

	int img_step = width;
	int sum_step = width + 1;
	int offset_sum_step = offset*sum_step;

	int *sum = (int*)malloc((width+1)*(height+1)*sizeof(int));
	myIntergral(mask, sum, width, height);

	for (int y = offset; y < height - offset; y++)
	{
		uchar* p = mask + y*img_step + offset;
		int* sum_ptr = sum + y*sum_step + offset  ; 
		for (int x = offset; x < width - offset; x++, p++, sum_ptr++)
		{
			int* p0 = /*sum_ptr - offset_sum_step - offset;*/ 			sum + (y-offset)*sum_step + x - offset;
			int* p1 = /*sum_ptr - offset_sum_step + offset + 1;	*/		sum + (y-offset)*sum_step + x + offset + 1;
			int* p2 = /*sum_ptr + offset_sum_step + sum_step - offset;*/	sum + (y+offset+1)*sum_step + x - offset;
			int* p3 = /*sum_ptr + offset_sum_step + offset + 1;	*/		sum + (y+offset+1)*sum_step + x + offset + 1;

			int cnt = *p0 + *p3 - *p1 - *p2;
			*p = (cnt > th ) ? 255 : 0;
		}
	}


	free(sum);
}

