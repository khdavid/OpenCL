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
    // find position in global arrays
    int i = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (i >= iNumElements)
    {   
        return; 
    }

    // process 
    int k = i * 4 ;
    c[i] = a[k] * b[k] 
               + a[k + 1] * b[k + 1]
               + a[k + 2] * b[k + 2]
               + a[k + 3] * b[k + 3];

   for(int ind = 0; ind < 1e4; ind++)
   {
     c[i] += sin(ind * a[ind % iNumElements]);
   }

}
)";