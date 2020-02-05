const char * CL_PROGRAM_DOT_PRODUCT = R"( 
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 __kernel void DotProduct (__global float* a, __global float* b, __global float* c, int iNumElements)
{
    int i = get_global_id(0);

   for(int ind = 0; ind < 1e4; ind++)
   {
     int k = (4 * i + ind) % iNumElements;
     c[i] += sin(k * a[k]) * cos(k * b[k]);
   }

}
)";