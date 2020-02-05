const char * CL_PROGRAM_DOT_PRODUCT = R"( 
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